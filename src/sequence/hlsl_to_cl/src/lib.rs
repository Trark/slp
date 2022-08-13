//! Compiles source Hlsl to OpenCL C and exports bindmaps

use slp_transform_cil_to_cst::UntyperError;
use slp_transform_cst_printer::Binary;
use slp_transform_hir_to_cil::TranspileError;
use slp_transform_hst_to_hir::TyperError;
use slp_transform_htk_to_hst::ParseError;
use slp_transform_lexer::LexError;
use slp_transform_preprocess::PreprocessError;
use std::fmt;

pub use slp_shared::BindMap;
pub use slp_shared::IncludeHandler;

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
    pub file_loader: Box<dyn IncludeHandler>,
    pub kernel_name: String,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Output {
    pub code: Binary,
    pub binds: BindMap,
    pub kernel_name: String,
    pub dimensions: (u64, u64, u64),
}

pub fn hlsl_to_cl(mut input: Input) -> Result<Output, CompileError> {
    let preprocessed =
        slp_transform_preprocess::preprocess(&input.main_file, &mut *input.file_loader)?;

    let tokens = slp_transform_lexer::lex(&preprocessed)?;

    let ast = slp_transform_htk_to_hst::parse(input.entry_point.to_string(), &tokens.stream)?;

    let ir = slp_transform_hst_to_hir::typeparse(&ast)?;

    let cil = slp_transform_hir_to_cil::transpile(&ir)?;

    let cst = slp_transform_cil_to_cst::untype_module(&cil, &input.kernel_name)?;

    let cl_binary = slp_transform_cst_printer::Binary::from_cir(&cst);

    let dimensions = (
        cst.kernel_dimensions.0,
        cst.kernel_dimensions.1,
        cst.kernel_dimensions.2,
    );
    Ok(Output {
        code: cl_binary,
        binds: cst.binds,
        kernel_name: cst.kernel_name,
        dimensions: dimensions,
    })
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CompileError::PreprocessError(ref preprocess_error) => {
                write!(f, "{}", preprocess_error)
            }
            CompileError::LexError(ref lexer_error) => {
                write!(f, "{}", lexer_error)
            }
            CompileError::ParseError(ref parser_error) => {
                write!(f, "{}", parser_error)
            }
            CompileError::TyperError(ref typer_error) => {
                write!(f, "{}", typer_error)
            }
            CompileError::TranspileError(ref transpiler_error) => {
                write!(f, "{}", transpiler_error)
            }
            CompileError::UntyperError(ref untyper_error) => {
                write!(f, "{}", untyper_error)
            }
        }
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
