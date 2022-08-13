#[macro_use]
extern crate nom;

mod lexer;

pub use lexer::lex;
pub use lexer::LexError;
