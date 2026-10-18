//! CUDA backend: cross-products in several precisions (cuBLAS / cuBLASLt) and NVRTC kernels.
//!
//! cuBLAS works column-major. Panels are `k x n` column-major with leading dimension `k`, and
//! the cross-product `C = X^T Y` is `n x n` column-major: `C[i + j * n]` pairs leader `i` with
//! target `j`.

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
            float r = ((float)col[i] - sx[i] * myj) * ax[i] * byj;                             \
            r = fminf(fmaxf(r, -0.999999f), 0.999999f);                                        \
            float z = fabsf(atanhf(r)) * z_scale;                                              \
            int bin = min((int)(z * bins_per_z), NBINS - 1);                                   \
            unsigned int peers = __match_any_sync(__activemask(), bin);                        \
            if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hs[bin], __popc(peers));    \
            if (z >= store_z && i != j) {                                                      \
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
"#;

pub const HIST_BINS: usize = 128;

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

    /// One census pass over a product: Pearson r for every pair, a |Fisher z| histogram
    /// (`HIST_BINS` bins of 1/`bins_per_z`), and the entries with |z| >= `store_z` (up to `cap`).
    #[allow(clippy::too_many_arguments)]
    pub fn census_pass(
        &self, product: &DevProduct, n: usize, rows: f64, factors: &PassFactors,
        bins_per_z: f32, store_z: f32, cap: usize,
        hist: &mut CudaSlice<u32>, count: &mut CudaSlice<u32>, out_idx: &mut CudaSlice<i64>, out_r: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let s = &self.stream;
        let (func, n_i, z_scale, cap_u) = (
            match product { DevProduct::F32(_) => &self.pass_f32, DevProduct::I32(_) => &self.pass_i32 },
            n as i32,
            ((rows - 3.0).max(1.0)).sqrt() as f32,
            cap as u32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(1024) as u32, n.min(1024) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(func);
        match product {
            DevProduct::F32(c) => b.arg(c),
            DevProduct::I32(c) => b.arg(c),
        };
        b.arg(&n_i).arg(&factors.sx).arg(&factors.my).arg(&factors.ax).arg(&factors.by).arg(&z_scale)
            .arg(&bins_per_z).arg(&store_z).arg(hist).arg(count).arg(&cap_u).arg(out_idx).arg(out_r);
        unsafe { b.launch(cfg)? };
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
