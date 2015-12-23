
#[macro_use]
extern crate nom;

pub mod hlsl;
pub mod clc;

use std::error;
use std::fmt;
use std::collections::HashMap;
use hlsl::preprocess::PreprocessError;
use hlsl::lexer::LexError;
use hlsl::parser::ParseError;
use hlsl::typer::TyperError;
use clc::transpiler::TranspileError;
use clc::untyper::UntyperError;

/// A file used as an input
#[derive(PartialEq, Debug, Clone)]
pub enum File {
    Unknown,
    Name(String),
}

/// A line number in a file
#[derive(PartialEq, Debug, Clone)]
pub struct Line(pub u64);

/// The column index in a line
#[derive(PartialEq, Debug, Clone)]
pub struct Column(pub u64);

/// Fully qualified location
#[derive(PartialEq, Debug, Clone)]
pub struct FileLocation(pub File, pub Line, pub Column);

/// The raw number of bytes from the start of a stream
#[derive(PartialEq, Debug, Clone)]
pub struct StreamLocation(pub u64);

/// Wrapper to pair a node with a FileLocation
#[derive(PartialEq, Debug, Clone)]
pub struct Located<T> {
    node: T,
    location: FileLocation,
}

impl<T> Located<T> {
    fn new(node: T, loc: FileLocation) -> Located<T> {
        Located {
            node: node,
            location: loc,
        }
    }
    pub fn to_node(self) -> T { self.node }
    pub fn to_loc(self) -> FileLocation { self.location }
    #[cfg(test)]
    fn loc(line: u64, column: u64, node: T) -> Located<T> {
        Located {
            node: node,
            location: FileLocation(File::Unknown, Line(line), Column(column)),
        }
    }
    #[cfg(test)]
    fn none(node: T) -> Located<T> {
        Located {
            node: node,
            location: FileLocation(File::Unknown, Line(0), Column(0)),
        }
    }
}

impl<T> std::ops::Deref for Located<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.node
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum CompileError {
    PreprocessError(PreprocessError),
    LexError(LexError),
    ParseError(ParseError),
    TyperError(TyperError),
    TranspileError(TranspileError),
    UntyperError(UntyperError),
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

/// Trait for loading files from #include directives
pub trait FileLoader {
    fn load(&self, &str) -> Result<String, ()>;
}

/// A file loader that fails to load any files
pub struct NullFileLoader;

impl FileLoader for NullFileLoader {
    fn load(&self, _: &str) -> Result<String, ()> { Err(()) }
}

pub struct Input {
    pub entry_point: String,
    pub main_file: String,
    pub file_loader: Box<FileLoader>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Output {
    pub code: clc::binary::Binary,
    pub binds: BindMap,
}

pub fn hlsl_to_cl(input: Input) -> Result<Output, CompileError> {

    let preprocessed = try!(hlsl::preprocess::preprocess(&input.main_file, &*input.file_loader));

    let tokens = try!(hlsl::lexer::lex(&preprocessed));

    let ast = try!(hlsl::parser::parse(input.entry_point.to_string(), &tokens.stream));

    let ir = try!(hlsl::typer::typeparse(&ast));

    let cil = try!(clc::transpiler::transpile(&ir));

    let cst = try!(clc::untyper::untype_module(&cil));

    let cl_binary = clc::binary::Binary::from_cir(&cst);

    Ok(Output { code: cl_binary, binds: cst.binds })
}

impl error::Error for CompileError {
    fn description(&self) -> &str {
        match *self {
            CompileError::PreprocessError(ref preprocess_error) => error::Error::description(preprocess_error),
            CompileError::LexError(ref lexer_error) => error::Error::description(lexer_error),
            CompileError::ParseError(ref parser_error) => error::Error::description(parser_error),
            CompileError::TyperError(ref typer_error) => error::Error::description(typer_error),
            CompileError::TranspileError(ref transpiler_error) => error::Error::description(transpiler_error),
            CompileError::UntyperError(ref untyper_error) => error::Error::description(untyper_error),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

impl From<PreprocessError> for CompileError {
    fn from(err: hlsl::preprocess::PreprocessError) -> CompileError {
        CompileError::PreprocessError(err)
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

impl From<UntyperError> for CompileError {
    fn from(err: UntyperError) -> CompileError {
        CompileError::UntyperError(err)
    }
}

#[cfg(test)]
pub mod tests;
