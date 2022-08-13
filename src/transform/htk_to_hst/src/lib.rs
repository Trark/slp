#[macro_use]
extern crate nom;

mod parser;

pub use parser::parse;
pub use parser::ParseError;
