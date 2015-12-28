extern crate slp_shared;
extern crate slp_lang_htk;
extern crate slp_lang_hst;
extern crate slp_lang_hir;
extern crate slp_lang_cil;
extern crate slp_lang_cst;
extern crate slp_transform_preprocess;
extern crate slp_transform_lexer;
extern crate slp_transform_htk_to_hst;
extern crate slp_transform_hst_to_hir;
extern crate slp_transform_hir_to_cil;
extern crate slp_transform_cil_to_cst;
extern crate slp_transform_cst_printer;

use std::error;
use std::fmt;
use slp_transform_preprocess::PreprocessError;
use slp_transform_lexer::LexError;
use slp_transform_htk_to_hst::ParseError;
use slp_transform_hst_to_hir::TyperError;
use slp_transform_hir_to_cil::TranspileError;
use slp_transform_cil_to_cst::UntyperError;
use slp_transform_cst_printer::Binary;

pub use slp_shared::IncludeHandler;
pub use slp_shared::BindMap;

#[derive(PartialEq, Debug, Clone)]
pub enum CompileError {
    PreprocessError(PreprocessError),
    LexError(LexError),
    ParseError(ParseError),
    TyperError(TyperError),
    TranspileError(TranspileError),
    UntyperError(UntyperError),
}

pub struct Input {
    pub entry_point: String,
    pub main_file: String,
    pub file_loader: Box<IncludeHandler>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Output {
    pub code: Binary,
    pub binds: BindMap,
}

pub fn hlsl_to_cl(input: Input) -> Result<Output, CompileError> {

    let preprocessed = try!(slp_transform_preprocess::preprocess(&input.main_file, &*input.file_loader));

    let tokens = try!(slp_transform_lexer::lex(&preprocessed));

    let ast = try!(slp_transform_htk_to_hst::parse(input.entry_point.to_string(), &tokens.stream));

    let ir = try!(slp_transform_hst_to_hir::typeparse(&ast));

    let cil = try!(slp_transform_hir_to_cil::transpile(&ir));

    let cst = try!(slp_transform_cil_to_cst::untype_module(&cil));

    let cl_binary = slp_transform_cst_printer::Binary::from_cir(&cst);

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
    fn from(err: PreprocessError) -> CompileError {
        CompileError::PreprocessError(err)
    }
}

impl From<LexError> for CompileError {
    fn from(err: LexError) -> CompileError {
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
