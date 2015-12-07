
pub mod tokens;
pub mod lexer;

pub mod ast;
pub mod parser;

pub mod ir;
pub mod typer;
pub mod intrinsics;
pub mod casting;
pub mod globals_analysis;

#[cfg(test)]
mod tests;
