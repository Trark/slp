
extern crate slp_shared;
extern crate slp_lang_cil;
extern crate slp_lang_cst;

mod untyper;

pub use untyper::UntyperError;
pub use untyper::untype_module;
