
extern crate slp_shared;
extern crate slp_lang_hst;

mod ir;
mod hir_intrinsics;
pub use ir::*;
pub use hir_intrinsics::*;

pub mod globals_analysis;
