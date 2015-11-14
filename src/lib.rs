#![feature(plugin)]

#[macro_use]
extern crate nom;

pub mod hlsl;
pub mod clc;

use hlsl::lexer::LexError;
use hlsl::parser::ParseError;
use hlsl::typer::TyperError;
use clc::transpiler::TranspileError;

#[derive(PartialEq, Debug, Clone)]
pub enum CompileError {
    LexError(LexError),
    ParseError(ParseError),
    TyperError(TyperError),
    TranspileError(TranspileError),
}

pub fn hlsl_to_cl(hlsl_source: &[u8], entry_point: &'static str) -> Result<clc::binary::Binary, CompileError> {

    let hlsl::tokens::TokenStream(tokens) = try!(hlsl::lexer::lex(hlsl_source));

    let ast = try!(hlsl::parser::parse(entry_point.to_string(), &tokens[..]));

    let ir = try!(hlsl::typer::typeparse(&ast));

    let cir = try!(clc::transpiler::transpile(&ir));

    let cl_binary = clc::binary::Binary::from_cir(&cir);

    Ok(cl_binary)
}

impl From<LexError> for CompileError {
    fn from(err: hlsl::lexer::LexError) -> CompileError {
        CompileError::LexError(err)
    }
}

impl From<ParseError> for CompileError {
    fn from(err: ParseError) -> CompileError {
        CompileError::ParseError(err)
    }
}

impl From<TyperError> for CompileError {
    fn from(err: TyperError) -> CompileError {
        CompileError::TyperError(err)
    }
}

impl From<TranspileError> for CompileError {
    fn from(err: TranspileError) -> CompileError {
        CompileError::TranspileError(err)
    }
}

#[cfg(test)]
pub mod tests;
