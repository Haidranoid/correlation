//! INT8 cross-products on the CPU with AVX-VNNI, using a hand-written `vpdpbusd` micro-kernel.
//!
//! `vpdpbusd acc, a, b` adds, in each of 8 32-bit lanes, the dot product of 4 unsigned bytes of
//! `a` with 4 signed bytes of `b`. Leaders are stored offset by +128 (as u8); the offset adds
//! 128 * sum(target column) to every result, which is subtracted afterwards.
//!
//! The micro-kernel computes a tile of `MR` leaders x `NR` targets, 4 rows (days) per step:
//! 12 accumulators + 3 target registers + 1 broadcast register = all 16 ymm registers.
//! Panels are packed so every load in the inner loop is sequential:
//!   leaders: `[kp/4][MR][4]` bytes per MR-block, targets: `[kp/4][NR][4]` bytes per NR-block.
//! Blocking: a target block (NR x kp bytes, 30 KB at kp = 1,280) stays in L1; a chunk of
//! `MC` leaders (320 KB) stays in L2.
//!
//! Built with `target-cpu=native` (see `.cargo/config.toml`), so AVX-VNNI is assumed present.

use rayon::prelude::*;

pub const MR: usize = 4;
pub const NR: usize = 24;
const MC: usize = 256;
const NC: usize = NR * 8;

/// Leaders `[block][kp/4][MR][4]` as u8 (+128), zero values (128) past `n`.
fn pack_leaders(x: &[i8], ld: usize, kp: usize, n: usize) -> Vec<u8> {
    let blocks = n.div_ceil(MR);
    let mut out = vec![128u8; blocks * kp * MR];
    out.par_chunks_mut(kp * MR).enumerate().for_each(|(b, dst)| {
        for ii in 0..MR {
            let i = b * MR + ii;
            if i >= n {
                continue;
            }
            let col = &x[i * ld..i * ld + kp];
            for s4 in 0..kp / 4 {
                for q in 0..4 {
                    dst[s4 * MR * 4 + ii * 4 + q] = (col[s4 * 4 + q] as i16 + 128) as u8;
                }
            }
        }
    });
    out
}

/// Targets `[block][kp/4][NR][4]` as i8, zeros past `n`.
fn pack_targets(y: &[i8], ld: usize, kp: usize, n: usize) -> Vec<i8> {
    let blocks = n.div_ceil(NR);
    let mut out = vec![0i8; blocks * kp * NR];
    out.par_chunks_mut(kp * NR).enumerate().for_each(|(b, dst)| {
        for jj in 0..NR {
            let j = b * NR + jj;
            if j >= n {
                continue;
            }
            let col = &y[j * ld..j * ld + kp];
            for s4 in 0..kp / 4 {
                dst[s4 * NR * 4 + jj * 4..s4 * NR * 4 + jj * 4 + 4].copy_from_slice(&col[s4 * 4..s4 * 4 + 4]);
            }
        }
    });
    out
}

/// `tile[ii][jj] = sum over steps of a[step][ii] . b[step][jj]` (4-byte dot products), row-major
/// `MR x NR` i32.
///
/// # Safety
/// `a` must hold `steps * MR * 4` bytes, `b` `steps * NR * 4` bytes, `tile` `MR * NR` i32s, and
/// `steps` must be at least 1. Requires AVX2 and AVX-VNNI.
#[inline]
unsafe fn micro_kernel(a: *const u8, b: *const i8, steps: usize, tile: *mut i32) {
    unsafe {
        core::arch::asm!(
            "vpxor ymm4, ymm4, ymm4",
            "vpxor ymm5, ymm5, ymm5",
            "vpxor ymm6, ymm6, ymm6",
            "vpxor ymm7, ymm7, ymm7",
            "vpxor ymm8, ymm8, ymm8",
            "vpxor ymm9, ymm9, ymm9",
            "vpxor ymm10, ymm10, ymm10",
            "vpxor ymm11, ymm11, ymm11",
            "vpxor ymm12, ymm12, ymm12",
            "vpxor ymm13, ymm13, ymm13",
            "vpxor ymm14, ymm14, ymm14",
            "vpxor ymm15, ymm15, ymm15",
            "2:",
            "vmovdqu ymm0, ymmword ptr [{b}]",
            "vmovdqu ymm1, ymmword ptr [{b} + 32]",
            "vmovdqu ymm2, ymmword ptr [{b} + 64]",
            "vpbroadcastd ymm3, dword ptr [{a}]",
            "{{vex}} vpdpbusd ymm4, ymm3, ymm0",
            "{{vex}} vpdpbusd ymm5, ymm3, ymm1",
            "{{vex}} vpdpbusd ymm6, ymm3, ymm2",
            "vpbroadcastd ymm3, dword ptr [{a} + 4]",
            "{{vex}} vpdpbusd ymm7, ymm3, ymm0",
            "{{vex}} vpdpbusd ymm8, ymm3, ymm1",
            "{{vex}} vpdpbusd ymm9, ymm3, ymm2",
            "vpbroadcastd ymm3, dword ptr [{a} + 8]",
            "{{vex}} vpdpbusd ymm10, ymm3, ymm0",
            "{{vex}} vpdpbusd ymm11, ymm3, ymm1",
            "{{vex}} vpdpbusd ymm12, ymm3, ymm2",
            "vpbroadcastd ymm3, dword ptr [{a} + 12]",
            "{{vex}} vpdpbusd ymm13, ymm3, ymm0",
            "{{vex}} vpdpbusd ymm14, ymm3, ymm1",
            "{{vex}} vpdpbusd ymm15, ymm3, ymm2",
            "add {a}, 16",
            "add {b}, 96",
            "dec {steps}",
            "jnz 2b",
            "vmovdqu ymmword ptr [{t}], ymm4",
            "vmovdqu ymmword ptr [{t} + 32], ymm5",
            "vmovdqu ymmword ptr [{t} + 64], ymm6",
            "vmovdqu ymmword ptr [{t} + 96], ymm7",
            "vmovdqu ymmword ptr [{t} + 128], ymm8",
            "vmovdqu ymmword ptr [{t} + 160], ymm9",
            "vmovdqu ymmword ptr [{t} + 192], ymm10",
            "vmovdqu ymmword ptr [{t} + 224], ymm11",
            "vmovdqu ymmword ptr [{t} + 256], ymm12",
            "vmovdqu ymmword ptr [{t} + 288], ymm13",
            "vmovdqu ymmword ptr [{t} + 320], ymm14",
            "vmovdqu ymmword ptr [{t} + 352], ymm15",
            a = inout(reg) a => _,
            b = inout(reg) b => _,
            steps = inout(reg) steps => _,
            t = in(reg) tile,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            options(nostack),
        );
    }
}

/// Packed INT8 panels, reusable across many products with the same inputs.
pub struct Packed {
    leaders: Vec<u8>,
    targets: Vec<i8>,
    /// 128 * column sum of each target, the correction for the leader offset.
    offset: Vec<i32>,
    kp: usize,
    n: usize,
}

impl Packed {
    /// Packs column-major `kp x n` INT8 panels (columns `ld` apart; `kp` a multiple of 4).
    pub fn new(x: &[i8], y: &[i8], kp: usize, n: usize, ld: usize) -> Self {
        assert!(kp % 4 == 0 && kp >= 4 && ld >= kp);
        let offset = (0..n).map(|j| 128 * y[j * ld..j * ld + kp].iter().map(|&v| v as i32).sum::<i32>()).collect();
        Self { leaders: pack_leaders(x, ld, kp, n), targets: pack_targets(y, ld, kp, n), offset, kp, n }
    }

    /// Calls `visit(i0, j0, tile)` for every `MR x NR` tile of exact INT32 cross-products
    /// (row-major `tile[ii * NR + jj]` for leader `i0 + ii`, target `j0 + jj`; entries past `n`
    /// are meaningless). Tiles are produced in parallel on all cores; `visit` must be `Sync`.
    pub fn for_each_tile(&self, visit: impl Fn(usize, usize, &[i32; MR * NR]) + Sync) {
        let (kp, n) = (self.kp, self.n);
        let (lb, tb) = (n.div_ceil(MR), n.div_ceil(NR));
        let chunks: Vec<(usize, usize)> = (0..lb.div_ceil(MC / MR))
            .flat_map(|ic| (0..tb.div_ceil(NC / NR)).map(move |jc| (ic, jc)))
            .collect();
        chunks.par_iter().for_each(|&(ic, jc)| {
            let mut tile = [0i32; MR * NR];
            for tbi in jc * (NC / NR)..((jc + 1) * (NC / NR)).min(tb) {
                let b = &self.targets[tbi * kp * NR..];
                for lbi in ic * (MC / MR)..((ic + 1) * (MC / MR)).min(lb) {
                    let a = &self.leaders[lbi * kp * MR..];
                    unsafe { micro_kernel(a.as_ptr(), b.as_ptr(), kp / 4, tile.as_mut_ptr()) };
                    for jj in 0..NR {
                        let j = tbi * NR + jj;
                        let corr = if j < n { self.offset[j] } else { 0 };
                        for ii in 0..MR {
                            tile[ii * NR + jj] -= corr;
                        }
                    }
                    visit(lbi * MR, tbi * NR, &tile);
                }
            }
        });
    }

    /// Full product, row-major `out[i * n + j]`.
    pub fn xprod(&self, out: &mut [i32]) {
        let n = self.n;
        assert!(out.len() >= n * n);
        let ptr = out.as_mut_ptr() as usize;
        self.for_each_tile(|i0, j0, tile| {
            for ii in 0..MR.min(n.saturating_sub(i0)) {
                for jj in 0..NR.min(n.saturating_sub(j0)) {
                    // Each (i, j) belongs to exactly one tile, so writes never overlap.
                    unsafe { *(ptr as *mut i32).add((i0 + ii) * n + j0 + jj) = tile[ii * NR + jj] };
                }
            }
        });
    }
}
