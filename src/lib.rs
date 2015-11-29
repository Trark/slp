#![feature(plugin)]

#[macro_use]
extern crate nom;

pub mod hlsl;
pub mod clc;

use std::error;
use std::fmt;
use std::collections::HashMap;
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

pub type KernelParamSlot = u32;

#[derive(PartialEq, Debug, Clone)]
pub struct BindMap {
    read_map: HashMap<u32, KernelParamSlot>,
    write_map: HashMap<u32, KernelParamSlot>,
    cbuffer_map: HashMap<u32, KernelParamSlot>,
    sampler_map: HashMap<u32, KernelParamSlot>,
}

impl BindMap {
    fn new() -> BindMap {
        BindMap {
            read_map: HashMap::new(),
            write_map: HashMap::new(),
            cbuffer_map: HashMap::new(),
            sampler_map: HashMap::new(),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Output {
    pub code: clc::binary::Binary,
    pub binds: BindMap,
}

pub fn hlsl_to_cl(hlsl_source: &[u8], entry_point: &'static str) -> Result<Output, CompileError> {

    let hlsl::tokens::TokenStream(tokens) = try!(hlsl::lexer::lex(hlsl_source));

    let ast = try!(hlsl::parser::parse(entry_point.to_string(), &tokens[..]));

    let ir = try!(hlsl::typer::typeparse(&ast));

    let cir = try!(clc::transpiler::transpile(&ir));

    let cl_binary = clc::binary::Binary::from_cir(&cir);

    Ok(Output { code: cl_binary, binds: cir.binds })
}

impl error::Error for CompileError {
    fn description(&self) -> &str {
        match *self {
            CompileError::LexError(ref lexer_error) => error::Error::description(lexer_error),
            CompileError::ParseError(ref parser_error) => error::Error::description(parser_error),
            CompileError::TyperError(ref typer_error) => error::Error::description(typer_error),
            CompileError::TranspileError(ref transpiler_error) => error::Error::description(transpiler_error),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
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
