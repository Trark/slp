
//! slp library for parsing shader languages
//!
//! This is a meta-crate that links all the individual crates together. The
//! crates are re-exported into modules.

pub extern crate slp_shared;
pub extern crate slp_lang_htk;
pub extern crate slp_lang_hst;
pub extern crate slp_lang_hir;
pub extern crate slp_lang_cil;
pub extern crate slp_lang_cst;
pub extern crate slp_transform_preprocess;
pub extern crate slp_transform_lexer;
pub extern crate slp_transform_htk_to_hst;
pub extern crate slp_transform_hst_to_hir;
pub extern crate slp_transform_hir_to_cil;
pub extern crate slp_transform_cil_to_cst;
pub extern crate slp_transform_cst_printer;
pub extern crate slp_sequence_hlsl_to_cl;

pub use slp_shared as shared;

/// The lang crates are intermediate language trees
pub mod lang {
    pub use slp_lang_htk as htk;
    pub use slp_lang_hst as hst;
    pub use slp_lang_hir as hir;
    pub use slp_lang_cil as cil;
    pub use slp_lang_cst as cst;
}

/// The transform crates are functions to transform from one language to another
pub mod transform {
    pub use slp_transform_preprocess as preprocess;
    pub use slp_transform_lexer as lexer;
    pub use slp_transform_hst_to_hir as hst_to_hir;
    pub use slp_transform_hir_to_cil as hir_to_cil;
    pub use slp_transform_cil_to_cst as cil_to_cst;
    pub use slp_transform_cst_printer as cst_printer;
}

/// The sequence crates are for combining sequences of transforms
pub mod sequence {
    pub use slp_sequence_hlsl_to_cl as hlsl_to_cl;
}

#[cfg(test)]
pub mod tests;
