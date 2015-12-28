
extern crate slp_shared;
extern crate slp_lang_htk;
extern crate slp_transform_preprocess;
#[macro_use]
extern crate nom;

mod lexer;

pub use lexer::LexError;
pub use lexer::lex;
