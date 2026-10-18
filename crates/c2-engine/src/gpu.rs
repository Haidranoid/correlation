//! CUDA backend: cross-products in several precisions (cuBLAS / cuBLASLt) and NVRTC kernels.
//!
//! cuBLAS works column-major. Panels are `k x n` column-major with leading dimension `k`, and
//! the cross-product `C = X^T Y` is `n x n` column-major: `C[i + j * n]` pairs leader `i` with
//! target `j`.

pub use cudarc;

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::cublaslt::{result as lt, sys as ltsys};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

/// Arithmetic used for the cross-product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// cuBLAS SGEMM, full FP32.
    Fp32,
    /// FP32 inputs on tensor cores with TF32 multiplies, FP32 sums.
    Tf32,
    /// FP16 inputs, FP32 sums.
    Fp16,
    /// BF16 inputs, FP32 sums.
    Bf16,
    /// INT8 inputs (`crate::INT8_SCALE` steps per sd), exact INT32 sums.
    Int8,
}

impl Precision {
    pub const ALL: [Precision; 5] = [Self::Fp32, Self::Tf32, Self::Fp16, Self::Bf16, Self::Int8];

    pub fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32",
            Self::Tf32 => "TF32",
            Self::Fp16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Int8 => "INT8",
        }
    }

    fn lt_types(self) -> (ltsys::cudaDataType, ltsys::cudaDataType, ltsys::cublasComputeType_t, ltsys::cudaDataType) {
        use ltsys::cublasComputeType_t::*;
        use ltsys::cudaDataType_t::*;
        match self {
            Self::Fp32 => (CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            Self::Tf32 => (CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F),
            Self::Fp16 => (CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            Self::Bf16 => (CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            Self::Int8 => (CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I),
        }
    }
}

/// A panel resident on the GPU in the storage type its precision needs.
pub enum DevPanel {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
    Bf16(CudaSlice<half::bf16>),
    I8(CudaSlice<i8>),
}

/// Cross-product output: FP32 for float precisions, INT32 for INT8.
pub enum DevProduct {
    F32(CudaSlice<f32>),
    I32(CudaSlice<i32>),
}

/// Per-stock factors turning a cross-product into r: `r = (c - sx[i] * my[j]) * ax[i] * by[j]`
/// with sx = leader sum, my = target mean, ax/by = 1 / sqrt(sum of squared deviations).
pub struct PassFactors {
    pub sx: CudaSlice<f32>,
    pub my: CudaSlice<f32>,
    pub ax: CudaSlice<f32>,
    pub by: CudaSlice<f32>,
}

/// A planned cuBLASLt matmul (descriptors and chosen algorithm) for one precision and shape.
pub struct LtPlan {
    desc: ltsys::cublasLtMatmulDesc_t,
    a: ltsys::cublasLtMatrixLayout_t,
    b: ltsys::cublasLtMatrixLayout_t,
    c: ltsys::cublasLtMatrixLayout_t,
    algo: ltsys::cublasLtMatmulAlgo_t,
    integer: bool,
}

impl Drop for LtPlan {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matrix_layout(self.a);
            let _ = lt::destroy_matrix_layout(self.b);
            let _ = lt::destroy_matrix_layout(self.c);
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

/// Converts a cross-product to Pearson r, histograms |Fisher z| and compacts entries above a
/// threshold. One pass over the matrix; the matrix itself never leaves the GPU.
///
/// Layout: `c[i + j * n]` (leader i, target j). Blocks walk down columns (contiguous i) with a
/// grid-stride loop over j. Per-stock factors are precomputed on the host so each entry is
/// `r = (c - sx[i] * my[j]) * ax[i] * by[j]`, all in FP32 (FP64 runs at 1/64 rate on GeForce).
/// Histogram updates are combined per warp with `__match_any_sync` before the shared atomic.
const KERNELS: &str = r#"
#define NBINS 128
#define CENSUS_PASS(NAME, T)                                                                   \
extern "C" __global__ void NAME(const T* __restrict__ c, int n,                                 \
        const float* __restrict__ sx, const float* __restrict__ my,                             \
        const float* __restrict__ ax, const float* __restrict__ by,                             \
        float z_scale, float bins_per_z, float store_z, unsigned int* hist,                     \
        unsigned int* count, unsigned int cap, long long* out_idx, float* out_r) {              \
    __shared__ unsigned int hs[NBINS];                                                         \
    for (int b = threadIdx.x; b < NBINS; b += blockDim.x) hs[b] = 0;                           \
    __syncthreads();                                                                           \
    for (int j = blockIdx.y; j < n; j += gridDim.y) {                                          \
        const T* col = c + (long long)j * n;                                                   \
        float myj = my[j], byj = by[j];                                                        \
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { \
            if (i == j) continue;                                                              \
            float r = ((float)col[i] - sx[i] * myj) * ax[i] * byj;                             \
            r = fminf(fmaxf(r, -0.999999f), 0.999999f);                                        \
            float z = fabsf(atanhf(r)) * z_scale;                                              \
            int bin = min((int)(z * bins_per_z), NBINS - 1);                                   \
            unsigned int peers = __match_any_sync(__activemask(), bin);                        \
            if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hs[bin], __popc(peers));    \
            if (z >= store_z) {                                                                \
                unsigned int p = atomicAdd(count, 1u);                                         \
                if (p < cap) { out_idx[p] = (long long)j * n + i; out_r[p] = r; }              \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
    __syncthreads();                                                                           \
    for (int b = threadIdx.x; b < NBINS; b += blockDim.x)                                      \
        if (hs[b]) atomicAdd(&hist[b], hs[b]);                                                 \
}
CENSUS_PASS(census_pass_f32, float)
CENSUS_PASS(census_pass_i32, int)

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sums of two values; thread 0 gets the totals. blockDim.x must be a multiple of 32.
__device__ void block_sum2(float& a, float& b) {
    __shared__ float sa[32], sb[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    a = warp_sum(a); b = warp_sum(b);
    if (lane == 0) { sa[warp] = a; sb[warp] = b; }
    __syncthreads();
    if (warp == 0) {
        a = lane < warps ? sa[lane] : 0.f; b = lane < warps ? sb[lane] : 0.f;
        a = warp_sum(a); b = warp_sum(b);
    }
}

// INT8 panel of window sums from per-column prefix sums (one block per column).
// prefix: column-major, column j at prefix + j * prefix_ld, prefix[0] = 0, prefix[r + 1] = sum
// of rows 0..=r. Row q < m of the panel is the sum over rows (e - h, e] with
// e = g0 + qs * step + offset, qs = (q + shift) % m, times `scale`, rounded and clipped to
// +-127; rows m..kp are zero. Also writes the column's sum and sum of squares (as quantized).
// Sign for observation q of a sign-flip null: the same for every stock, so volatility bursts
// stay aligned and only the leader-target relationship is destroyed. Matches
// `c2_engine::sign_flip` on the host.
__device__ float flip_sign(unsigned int seed, unsigned int q) {
    unsigned int h = seed ^ (q * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return (h & 1u) ? -1.f : 1.f;
}

extern "C" __global__ void build_panel(const double* __restrict__ prefix, int prefix_ld, int n,
        int g0, int step, int offset, int h, int m, int shift, unsigned int sign_seed, float scale, int kp,
        signed char* __restrict__ out, float* __restrict__ col_sum, float* __restrict__ col_sumsq) {
    int j = blockIdx.x;
    if (j >= n) return;
    const double* p = prefix + (long long)j * prefix_ld;
    signed char* o = out + (long long)j * kp;
    float s = 0.f, ss = 0.f;
    for (int q = threadIdx.x; q < kp; q += blockDim.x) {
        float x = 0.f;
        if (q < m) {
            int qs = (q + shift) % m;
            int e = g0 + qs * step + offset;
            x = (float)(p[e + 1] - p[e + 1 - h]) * scale;
            if (sign_seed) x *= flip_sign(sign_seed, (unsigned int)q);
            x = fminf(fmaxf(rintf(x), -127.f), 127.f);
        }
        o[q] = (signed char)x;
        s += x; ss += x * x;
    }
    block_sum2(s, ss);
    if (threadIdx.x == 0) { col_sum[j] = s; col_sumsq[j] = ss; }
}

// Sum and sum of squares of the first `rows` rows of each column of an INT8 panel.
extern "C" __global__ void col_stats(const signed char* __restrict__ panel, int kp, int rows, int n,
        float* __restrict__ col_sum, float* __restrict__ col_sumsq) {
    int j = blockIdx.x;
    if (j >= n) return;
    const signed char* c = panel + (long long)j * kp;
    float s = 0.f, ss = 0.f;
    for (int q = threadIdx.x; q < rows; q += blockDim.x) { float x = c[q]; s += x; ss += x * x; }
    block_sum2(s, ss);
    if (threadIdx.x == 0) { col_sum[j] = s; col_sumsq[j] = ss; }
}

// ---- Census v2: |r| histograms, thresholds chosen on the device, compaction. No host round
// trip per matrix. RBINS bins of |r| (1/1024 wide); for one matrix (fixed m) |r| ordering is |z|
// ordering, so false-discovery and top-K cut-offs can be found from |r| bins alone.
#define RBINS 1024

#define CENSUS_R(i, j)                                                                         \
    float r = ((float)col[i] - sx[i] * myj) * ax[i] * byj;                                    \
    float a = fminf(fabsf(r), 0.999999f);

extern "C" __global__ void census_hist_i32(const int* __restrict__ c, int n,
        const float* __restrict__ sx, const float* __restrict__ my,
        const float* __restrict__ ax, const float* __restrict__ by, unsigned int* hist) {
    __shared__ unsigned int hs[RBINS];
    for (int b = threadIdx.x; b < RBINS; b += blockDim.x) hs[b] = 0;
    __syncthreads();
    for (int j = blockIdx.y; j < n; j += gridDim.y) {
        const int* col = c + (long long)j * n;
        float myj = my[j], byj = by[j];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            if (i == j) continue;
            CENSUS_R(i, j)
            int bin = (int)(a * RBINS);
            unsigned int peers = __match_any_sync(__activemask(), bin);
            if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hs[bin], __popc(peers));
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < RBINS; b += blockDim.x)
        if (hs[b]) atomicAdd(&hist[b], hs[b]);
}

// One thread. From the real histogram and `nulls` null histograms (contiguous after it):
// out[0] = |r| cut-off for real entries: the top `keep_real`, extended down to the false
// discovery cut-off (mean null count / real count <= q) when use_fdr, but never more than
// keep_real + fdr_cap entries; out[1] = cut-off for the top `keep_null` of null 0;
// out[2] = the false-discovery cut-off itself (2.0 if none).
// The expected chance count above a cut-off is the larger of the nulls' mean count and the
// analytic one (pairs x P(|Fisher z| >= z) for m observations): with only a few nulls the
// empirical far tail is usually empty, which would let any single extreme real entry pass.
extern "C" __global__ void choose_thresholds(const unsigned int* hist, int nulls, int keep_real,
        int keep_null, int fdr_cap, float q, int use_fdr, double pairs, int m, float* out) {
    if (blockIdx.x || threadIdx.x) return;
    const unsigned int* real = hist;
    double cr = 0.0, cn = 0.0;
    double zs = sqrt(fmax((double)m - 3.0, 1.0));
    int b_top = 0, b_fdr = -1, found = 0;
    for (int b = RBINS - 1; b >= 0; b--) {
        cr += real[b];
        double nm = 0.0;
        for (int t = 0; t < nulls; t++) nm += hist[(1 + t) * RBINS + b];
        cn += nm / (double)(nulls > 0 ? nulls : 1);
        double r = fmin(b / (double)RBINS, 0.999999);
        double analytic = pairs * erfc(atanh(r) * zs / sqrt(2.0));
        double expect = fmax(cn, analytic);
        if (!found && cr >= keep_real) { b_top = b; found = 1; }
        if (use_fdr && cr > 0.0 && expect / cr <= q) b_fdr = b;
    }
    int b_real = b_top;
    if (b_fdr >= 0 && b_fdr < b_top) {
        double cum = 0.0;
        int b_cap = 0;
        for (int b = RBINS - 1; b >= 0; b--) {
            cum += real[b];
            if (cum > (double)(keep_real + fdr_cap)) { b_cap = b + 1; break; }
        }
        b_real = min(max(b_fdr, b_cap), b_top);
    }
    double c0 = 0.0;
    int b_null = 0;
    for (int b = RBINS - 1; b >= 0; b--) {
        c0 += hist[RBINS + b];
        if (c0 >= keep_null) { b_null = b; break; }
    }
    out[0] = b_real / (float)RBINS;
    out[1] = b_null / (float)RBINS;
    out[2] = b_fdr >= 0 ? b_fdr / (float)RBINS : 2.0f;
}

// Entries with |r| >= thr[which] (up to cap; count may exceed cap).
extern "C" __global__ void census_compact_i32(const int* __restrict__ c, int n,
        const float* __restrict__ sx, const float* __restrict__ my,
        const float* __restrict__ ax, const float* __restrict__ by,
        const float* __restrict__ thr, int which, unsigned int* count, unsigned int cap,
        long long* out_idx, float* out_r) {
    float t = thr[which];
    for (int j = blockIdx.y; j < n; j += gridDim.y) {
        const int* col = c + (long long)j * n;
        float myj = my[j], byj = by[j];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            if (i == j) continue;
            CENSUS_R(i, j)
            if (fminf(a, 0.999999f) >= t) {
                unsigned int p = atomicAdd(count, 1u);
                if (p < cap) { out_idx[p] = (long long)j * n + i; out_r[p] = r; }
            }
        }
    }
}

// Per-stock census-pass factors from column sums over m rows.
extern "C" __global__ void pass_factors(const float* sx_sum, const float* sx_sq, const float* sy_sum,
        const float* sy_sq, int n, float m, float* sx, float* my, float* ax, float* by) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float vx = sx_sq[j] - sx_sum[j] * sx_sum[j] / m, vy = sy_sq[j] - sy_sum[j] * sy_sum[j] / m;
    sx[j] = sx_sum[j];
    my[j] = sy_sum[j] / m;
    ax[j] = vx > 1e-3f ? rsqrtf(vx) : 0.f;
    by[j] = vy > 1e-3f ? rsqrtf(vy) : 0.f;
}
"#;

pub const HIST_BINS: usize = 128;

/// Bins of |r| in the census histograms (each 1/1024 wide).
pub const R_BINS: usize = 1024;

/// The CUDA libraries this backend needs that can't be loaded right now.
/// (cudarc panics if a library is missing when first used, so check before touching the GPU.)
pub fn missing_libraries() -> Vec<&'static str> {
    let mut missing = Vec::new();
    unsafe {
        if !cudarc::driver::sys::is_culib_present() {
            missing.push("CUDA driver (libcuda)");
        }
        if !cudarc::cublas::sys::is_culib_present() {
            missing.push("cuBLAS");
        }
        if !cudarc::cublaslt::sys::is_culib_present() {
            missing.push("cuBLASLt");
        }
        if !cudarc::nvrtc::sys::is_culib_present() {
            missing.push("NVRTC");
        }
    }
    missing
}

/// Where to look for the CUDA libraries: `C2_CUDA_LIB_DIR`, then known installs on this machine.
fn cuda_library_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = std::env::var_os("C2_CUDA_LIB_DIR").into_iter().map(Into::into).collect();
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(std::path::Path::new(&home).join("model-lab/qwen3.8-flash-next/cuda-env/lib"));
    }
    dirs.extend(["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"].map(Into::into));
    dirs
}

/// If the CUDA libraries can't be loaded but one of `cuda_library_dirs` has them, re-run this
/// program with that directory on `LD_LIBRARY_PATH` (the loader reads it only at startup).
/// Returns when no re-run is needed or possible; the caller then falls back to the CPU.
pub fn reexec_with_cuda_libraries() {
    use std::os::unix::process::CommandExt;
    if std::env::var_os("C2_CUDA_REEXEC").is_some() || missing_libraries().is_empty() {
        return;
    }
    let Some(dir) = cuda_library_dirs().into_iter().find(|d| d.join("libcublas.so.13").exists() || d.join("libcublas.so").exists()) else {
        return;
    };
    let mut path = dir.into_os_string();
    if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
        path.push(":");
        path.push(old);
    }
    let Ok(exe) = std::env::current_exe() else { return };
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", path)
        .env("C2_CUDA_REEXEC", "1")
        .exec();
    eprintln!("could not re-run with the CUDA libraries: {err}");
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    lt: ltsys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    pass_f32: CudaFunction,
    pass_i32: CudaFunction,
    build_panel_fn: CudaFunction,
    col_stats_fn: CudaFunction,
    factors_fn: CudaFunction,
    hist_r_fn: CudaFunction,
    choose_fn: CudaFunction,
    compact_fn: CudaFunction,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe {
            let _ = lt::destroy_handle(self.lt);
        }
    }
}

impl Gpu {
    /// Opens device 0, creates cuBLAS/cuBLASLt handles and compiles the NVRTC kernels for this GPU.
    pub fn new() -> Result<Self> {
        let missing = missing_libraries();
        if !missing.is_empty() {
            return Err(anyhow!("CUDA libraries not found: {} (set C2_CUDA_LIB_DIR to their directory)", missing.join(", ")));
        }
        let ctx = CudaContext::new(0).context("no usable CUDA device")?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let lt = lt::create_handle()?;
        let workspace = stream.alloc_zeros::<u8>(32 << 20)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let ptx = compile_ptx_with_opts(KERNELS, CompileOptions { arch: Some(arch), ..Default::default() })
            .map_err(|e| anyhow!("NVRTC compile for {arch} failed: {e:?}"))?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            pass_f32: module.load_function("census_pass_f32")?,
            pass_i32: module.load_function("census_pass_i32")?,
            build_panel_fn: module.load_function("build_panel")?,
            col_stats_fn: module.load_function("col_stats")?,
            factors_fn: module.load_function("pass_factors")?,
            hist_r_fn: module.load_function("census_hist_i32")?,
            choose_fn: module.load_function("choose_thresholds")?,
            compact_fn: module.load_function("census_compact_i32")?,
            ctx,
            stream,
            blas,
            lt,
            workspace,
        })
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.ctx.name()?)
    }

    /// (free, total) device memory in bytes.
    pub fn memory(&self) -> Result<(usize, usize)> {
        Ok(cudarc::driver::result::mem_get_info()?)
    }

    /// Uploads a column-major panel, converting to the precision's storage type.
    pub fn upload(&self, panel: &[f32], p: Precision) -> Result<DevPanel> {
        let s = &self.stream;
        Ok(match p {
            Precision::Fp32 | Precision::Tf32 => DevPanel::F32(s.clone_htod(panel)?),
            Precision::Fp16 => DevPanel::F16(s.clone_htod(&panel.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>())?),
            Precision::Bf16 => DevPanel::Bf16(s.clone_htod(&panel.iter().map(|&v| half::bf16::from_f32(v)).collect::<Vec<_>>())?),
            Precision::Int8 => DevPanel::I8(s.clone_htod(&panel.iter().map(|&v| crate::quantize(v)).collect::<Vec<_>>())?),
        })
    }

    pub fn alloc_product(&self, p: Precision, n: usize) -> Result<DevProduct> {
        Ok(match p {
            Precision::Int8 => DevProduct::I32(self.stream.alloc_zeros::<i32>(n * n)?),
            _ => DevProduct::F32(self.stream.alloc_zeros::<f32>(n * n)?),
        })
    }

    /// Plans `C (n x n) = X^T Y` for `k x n` panels with leading dimension `k` (cuBLASLt only;
    /// FP32 uses classic SGEMM and needs no plan).
    pub fn plan(&self, p: Precision, n: usize, k: usize) -> Result<LtPlan> {
        let (ab, c_ty, compute, scale) = p.lt_types();
        let (n64, k64) = (n as u64, k as u64);
        let desc = lt::create_matmul_desc(compute, scale)?;
        let transpose: i32 = 1;
        unsafe {
            lt::set_matmul_desc_attribute(
                desc,
                ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &transpose as *const i32 as *const c_void,
                size_of::<i32>(),
            )?;
        }
        let a = lt::create_matrix_layout(ab, k64, n64, k as i64)?;
        let b = lt::create_matrix_layout(ab, k64, n64, k as i64)?;
        let c = lt::create_matrix_layout(c_ty, n64, n64, n as i64)?;
        let pref = lt::create_matmul_pref()?;
        let ws = self.workspace.len();
        let heuristic = unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                ltsys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const usize as *const c_void,
                size_of::<usize>(),
            )?;
            let h = lt::get_matmul_algo_heuristic(self.lt, desc, a, b, c, c, pref);
            let _ = lt::destroy_matmul_pref(pref);
            h.with_context(|| format!("cuBLASLt has no {} algorithm for n={n}, k={k}", p.name()))?
        };
        Ok(LtPlan { desc, a, b, c, algo: heuristic.algo, integer: p == Precision::Int8 })
    }

    /// Runs `C = X^T Y`. `plan` must come from `self.plan` for the same precision and shape
    /// (it is ignored for FP32, which uses SGEMM).
    pub fn xprod(&self, p: Precision, plan: Option<&LtPlan>, x: &DevPanel, y: &DevPanel, out: &mut DevProduct, n: usize, k: usize) -> Result<()> {
        if let (Precision::Fp32, DevPanel::F32(x), DevPanel::F32(y), DevProduct::F32(c)) = (p, x, y, &mut *out) {
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: n as i32,
                k: k as i32,
                alpha: 1.0f32,
                lda: k as i32,
                ldb: k as i32,
                beta: 0.0f32,
                ldc: n as i32,
            };
            unsafe { self.blas.gemm(cfg, x, y, c)? };
            return Ok(());
        }
        let plan = plan.ok_or_else(|| anyhow!("{} needs a cuBLASLt plan", p.name()))?;
        let s = &self.stream;
        let (xa, _gx) = match x {
            DevPanel::F32(v) => v.device_ptr(s),
            DevPanel::F16(v) => v.device_ptr(s),
            DevPanel::Bf16(v) => v.device_ptr(s),
            DevPanel::I8(v) => v.device_ptr(s),
        };
        let (ya, _gy) = match y {
            DevPanel::F32(v) => v.device_ptr(s),
            DevPanel::F16(v) => v.device_ptr(s),
            DevPanel::Bf16(v) => v.device_ptr(s),
            DevPanel::I8(v) => v.device_ptr(s),
        };
        let (ca, _gc) = match &*out {
            DevProduct::F32(v) => v.device_ptr(s),
            DevProduct::I32(v) => v.device_ptr(s),
        };
        let (wa, _gw) = self.workspace.device_ptr(s);
        let (fa, fb, ia, ib) = (1.0f32, 0.0f32, 1i32, 0i32);
        let (alpha, beta) = if plan.integer {
            (&ia as *const i32 as *const c_void, &ib as *const i32 as *const c_void)
        } else {
            (&fa as *const f32 as *const c_void, &fb as *const f32 as *const c_void)
        };
        unsafe {
            lt::matmul(
                self.lt, plan.desc, alpha, beta,
                xa as *const c_void, plan.a,
                ya as *const c_void, plan.b,
                ca as *const c_void, plan.c,
                ca as *mut c_void, plan.c,
                &plan.algo, wa as *mut c_void, self.workspace.len(),
                s.cu_stream() as *mut _,
            )?;
        }
        Ok(())
    }

    /// One census pass over a product: Pearson r for every pair (diagonal skipped), a |Fisher z|
    /// histogram (`HIST_BINS` bins of 1/`bins_per_z`) and the entries with |z| >= `store_z` (up to
    /// `cap`; `count` may exceed `cap`). `hist` and `count` are accumulated into, not cleared.
    #[allow(clippy::too_many_arguments)]
    pub fn census_pass(
        &self, product: &DevProduct, n: usize, rows: f64, factors: &PassFactors,
        bins_per_z: f32, store_z: f32, cap: usize,
        hist: &mut CudaSlice<u32>, count: &mut CudaSlice<u32>, out_idx: &mut CudaSlice<i64>, out_r: &mut CudaSlice<f32>,
    ) -> Result<()> {
        match product {
            DevProduct::F32(c) => self.pass_launch(&self.pass_f32, c, n, rows, factors, bins_per_z, store_z, cap, hist, count, out_idx, out_r),
            DevProduct::I32(c) => self.pass_launch(&self.pass_i32, c, n, rows, factors, bins_per_z, store_z, cap, hist, count, out_idx, out_r),
        }
    }

    /// `census_pass` for an INT32 product held directly.
    #[allow(clippy::too_many_arguments)]
    pub fn census_pass_i32(
        &self, c: &CudaSlice<i32>, n: usize, rows: f64, factors: &PassFactors,
        bins_per_z: f32, store_z: f32, cap: usize,
        hist: &mut CudaSlice<u32>, count: &mut CudaSlice<u32>, out_idx: &mut CudaSlice<i64>, out_r: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.pass_launch(&self.pass_i32, c, n, rows, factors, bins_per_z, store_z, cap, hist, count, out_idx, out_r)
    }

    /// `census_pass_i32` writing the histogram and count into sub-ranges of larger buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn census_pass_i32_view(
        &self, c: &CudaSlice<i32>, n: usize, rows: f64, factors: &PassFactors,
        bins_per_z: f32, store_z: f32, cap: usize,
        hist: &mut cudarc::driver::CudaViewMut<u32>, count: &mut cudarc::driver::CudaViewMut<u32>,
        out_idx: &mut CudaSlice<i64>, out_r: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let (n_i, z_scale, cap_u) = (n as i32, ((rows - 3.0).max(1.0)).sqrt() as f32, cap as u32);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(1024) as u32, n.min(1024) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.pass_i32);
        b.arg(c).arg(&n_i).arg(&factors.sx).arg(&factors.my).arg(&factors.ax).arg(&factors.by).arg(&z_scale)
            .arg(&bins_per_z).arg(&store_z).arg(hist).arg(count).arg(&cap_u).arg(out_idx).arg(out_r);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn pass_launch<T: cudarc::driver::DeviceRepr>(
        &self, func: &CudaFunction, c: &CudaSlice<T>, n: usize, rows: f64, factors: &PassFactors,
        bins_per_z: f32, store_z: f32, cap: usize,
        hist: &mut CudaSlice<u32>, count: &mut CudaSlice<u32>, out_idx: &mut CudaSlice<i64>, out_r: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let (n_i, z_scale, cap_u) = (n as i32, ((rows - 3.0).max(1.0)).sqrt() as f32, cap as u32);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(1024) as u32, n.min(1024) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(func);
        b.arg(c).arg(&n_i).arg(&factors.sx).arg(&factors.my).arg(&factors.ax).arg(&factors.by).arg(&z_scale)
            .arg(&bins_per_z).arg(&store_z).arg(hist).arg(count).arg(&cap_u).arg(out_idx).arg(out_r);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// Fills an INT8 panel of window sums from per-column prefix sums (see the `build_panel`
    /// kernel) and writes each column's sum and sum of squares.
    #[allow(clippy::too_many_arguments)]
    /// `sign_seed` 0 builds the real panel; non-zero builds a sign-flip null (see `sign_flip`).
    pub fn build_panel(
        &self, prefix: &CudaSlice<f64>, prefix_ld: usize, n: usize, g0: usize, step: usize, offset: usize,
        h: usize, m: usize, shift: usize, sign_seed: u32, kp: usize,
        out: &mut CudaSlice<i8>, sum: &mut CudaSlice<f32>, sumsq: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let scale = crate::INT8_SCALE / (h as f32).sqrt();
        let ints = [prefix_ld, n, g0, step, offset, h, m, shift].map(|v| v as i32);
        let kp_i = kp as i32;
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.build_panel_fn);
        b.arg(prefix);
        for v in &ints {
            b.arg(v);
        }
        b.arg(&sign_seed).arg(&scale).arg(&kp_i).arg(out).arg(sum).arg(sumsq);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// Sum and sum of squares of the first `rows` rows of each column of an INT8 panel.
    pub fn col_stats(&self, panel: &CudaSlice<i8>, kp: usize, rows: usize, n: usize, sum: &mut CudaSlice<f32>, sumsq: &mut CudaSlice<f32>) -> Result<()> {
        let (kp_i, rows_i, n_i) = (kp as i32, rows as i32, n as i32);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.col_stats_fn);
        b.arg(panel).arg(&kp_i).arg(&rows_i).arg(&n_i).arg(sum).arg(sumsq);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    /// Census-pass factors from leader and target column sums over `m` rows, on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn factors(&self, sx_sum: &CudaSlice<f32>, sx_sq: &CudaSlice<f32>, sy_sum: &CudaSlice<f32>, sy_sq: &CudaSlice<f32>, n: usize, m: usize, out: &mut PassFactors) -> Result<()> {
        let (n_i, m_f) = (n as i32, m as f32);
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.factors_fn);
        b.arg(sx_sum).arg(sx_sq).arg(sy_sum).arg(sy_sq).arg(&n_i).arg(&m_f).arg(&mut out.sx).arg(&mut out.my).arg(&mut out.ax).arg(&mut out.by);
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    fn grid2d(n: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (n.div_ceil(1024) as u32, n.min(1024) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// Adds the |r| histogram (`R_BINS` bins, diagonal skipped) of an INT32 product to `hist`.
    pub fn census_hist(&self, c: &CudaSlice<i32>, n: usize, f: &PassFactors, hist: &mut cudarc::driver::CudaViewMut<u32>) -> Result<()> {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.hist_r_fn);
        b.arg(c).arg(&n_i).arg(&f.sx).arg(&f.my).arg(&f.ax).arg(&f.by).arg(hist);
        unsafe { b.launch(Self::grid2d(n))? };
        Ok(())
    }

    /// Picks storage cut-offs from `hist` = real histogram followed by `nulls` null histograms
    /// (see the `choose_thresholds` kernel); writes 3 floats to `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn choose_thresholds(
        &self, hist: &cudarc::driver::CudaView<u32>, nulls: usize, keep_real: usize, keep_null: usize, fdr_cap: usize,
        q: f32, use_fdr: bool, n: usize, m: usize, out: &mut cudarc::driver::CudaViewMut<f32>,
    ) -> Result<()> {
        let args = [nulls as i32, keep_real as i32, keep_null as i32, fdr_cap as i32];
        let use_fdr = i32::from(use_fdr);
        let (pairs, m_i) = ((n * n.saturating_sub(1)) as f64, m as i32);
        let mut b = self.stream.launch_builder(&self.choose_fn);
        b.arg(hist).arg(&args[0]).arg(&args[1]).arg(&args[2]).arg(&args[3]).arg(&q).arg(&use_fdr).arg(&pairs).arg(&m_i).arg(out);
        unsafe { b.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 })? };
        Ok(())
    }

    /// Compacts entries with |r| >= `thr[which]` (up to `cap`) from an INT32 product.
    #[allow(clippy::too_many_arguments)]
    pub fn census_compact(
        &self, c: &CudaSlice<i32>, n: usize, f: &PassFactors, thr: &cudarc::driver::CudaView<f32>, which: usize,
        count: &mut cudarc::driver::CudaViewMut<u32>, cap: usize,
        idx: &mut cudarc::driver::CudaViewMut<i64>, r: &mut cudarc::driver::CudaViewMut<f32>,
    ) -> Result<()> {
        let (n_i, which_i, cap_u) = (n as i32, which as i32, cap as u32);
        let mut b = self.stream.launch_builder(&self.compact_fn);
        b.arg(c).arg(&n_i).arg(&f.sx).arg(&f.my).arg(&f.ax).arg(&f.by).arg(thr).arg(&which_i).arg(count).arg(&cap_u).arg(idx).arg(r);
        unsafe { b.launch(Self::grid2d(n))? };
        Ok(())
    }

    pub fn alloc_factors(&self, n: usize) -> Result<PassFactors> {
        let s = &self.stream;
        Ok(PassFactors { sx: s.alloc_zeros(n)?, my: s.alloc_zeros(n)?, ax: s.alloc_zeros(n)?, by: s.alloc_zeros(n)? })
    }

    /// INT8 `C = X^T Y` with a plan from `plan(Precision::Int8, n, kp)`.
    pub fn xprod_i8(&self, plan: &LtPlan, x: &CudaSlice<i8>, y: &CudaSlice<i8>, out: &mut CudaSlice<i32>) -> Result<()> {
        let s = &self.stream;
        let (xa, _gx) = x.device_ptr(s);
        let (ya, _gy) = y.device_ptr(s);
        let (ca, _gc) = out.device_ptr(s);
        let (wa, _gw) = self.workspace.device_ptr(s);
        let (alpha, beta) = (1i32, 0i32);
        unsafe {
            lt::matmul(
                self.lt, plan.desc,
                &alpha as *const i32 as *const c_void, &beta as *const i32 as *const c_void,
                xa as *const c_void, plan.a, ya as *const c_void, plan.b,
                ca as *const c_void, plan.c, ca as *mut c_void, plan.c,
                &plan.algo, wa as *mut c_void, self.workspace.len(), s.cu_stream() as *mut _,
            )?;
        }
        Ok(())
    }

    /// Per-stock factors for `census_pass`, from leader and target column sums over `rows` rows.
    pub fn pass_factors(&self, leader: &crate::ColumnStats, target: &crate::ColumnStats, rows: f64) -> Result<PassFactors> {
        let inv_sd = |st: &crate::ColumnStats| -> Vec<f32> {
            st.sum.iter().zip(&st.sum_sq).map(|(s, q)| (1.0 / (q - s * s / rows).max(1e-12).sqrt()) as f32).collect()
        };
        let s = &self.stream;
        Ok(PassFactors {
            sx: s.clone_htod(&leader.sum.iter().map(|&v| v as f32).collect::<Vec<_>>())?,
            my: s.clone_htod(&target.sum.iter().map(|&v| (v / rows) as f32).collect::<Vec<_>>())?,
            ax: s.clone_htod(&inv_sd(leader))?,
            by: s.clone_htod(&inv_sd(target))?,
        })
    }
}
