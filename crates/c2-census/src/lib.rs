//! The correlation census: every pair's lead-lag correlation for each rolling window, return
//! horizon and lag, screened in INT8 on the GPU and stored with exact recomputation.

pub mod analyze;
pub mod grid;
pub mod holdout;
pub mod run;
pub mod validate;
pub mod variable;
