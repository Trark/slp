
extern crate slp_shared;
#[cfg(test)]
extern crate slp_lang_hst;
extern crate slp_lang_hir;
extern crate slp_lang_cil;
extern crate slp_lang_cst;
#[cfg(test)]
extern crate slp_transform_hst_to_hir;

mod transpiler;

pub use transpiler::TranspileError;
pub use transpiler::transpile;
