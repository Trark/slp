
extern crate slp_shared;
extern crate slp_lang_hst;
extern crate slp_lang_hir;

mod intrinsics;
mod typer;
mod casting;
mod pel;
mod rel;

pub use typer::TyperError;
pub use typer::typeparse;
