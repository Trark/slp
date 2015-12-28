
extern crate slp_shared;
extern crate slp_lang_htk;
extern crate slp_lang_hst;
#[cfg(test)]
extern crate slp_transform_preprocess;
#[cfg(test)]
extern crate slp_transform_lexer;
#[macro_use]
extern crate nom;

mod parser;

pub use parser::ParseError;
pub use parser::parse;
