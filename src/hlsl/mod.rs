
pub mod tokens;
pub mod lexer;

pub mod ast;
pub mod parser;

pub mod ir;
pub mod typer;
pub mod intrinsics;

#[cfg(test)]
mod tests;
