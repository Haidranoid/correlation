//! Synthetic panels with stock-market-like structure, for benchmarks and self-tests.

/// Small, fast, deterministic generator (xorshift64*).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Column-major `t x n` panel of standardized returns: a market factor, 20 sector factors, a
/// small one-day lead from each sector's first stock to the rest of its sector, idiosyncratic
/// noise, and occasional fat-tailed days. Clipped at +-5 sd like real census inputs.
pub fn returns_panel(t: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let market: Vec<f64> = (0..t).map(|_| rng.normal()).collect();
    let sectors: Vec<Vec<f64>> = (0..20).map(|_| (0..t).map(|_| rng.normal()).collect()).collect();
    let mut panel = vec![0f32; t * n];
    let mut leaders = vec![vec![0f64; t]; 20];
    for j in 0..n {
        let (s, beta, gamma) = (j % 20, 0.3 + 0.4 * rng.uniform(), 0.2 + 0.3 * rng.uniform());
        let mut col = Vec::with_capacity(t);
        for d in 0..t {
            let tail = if rng.uniform() < 0.01 { 4.0 } else { 1.0 };
            let lead = if j >= 20 && d > 0 { 0.05 * leaders[s][d - 1] } else { 0.0 };
            col.push(beta * market[d] + gamma * sectors[s][d] + lead + tail * rng.normal());
        }
        let mean = col.iter().sum::<f64>() / t as f64;
        let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / t as f64).sqrt();
        for (d, v) in col.iter().enumerate() {
            panel[j * t + d] = ((v - mean) / sd).clamp(-5.0, 5.0) as f32;
        }
        if j < 20 {
            leaders[s] = (0..t).map(|d| panel[j * t + d] as f64).collect();
        }
    }
    panel
}
