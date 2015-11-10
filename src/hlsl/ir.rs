
use std::collections::HashMap;
use super::ast;

pub use super::ast::ScalarType as ScalarType;
pub use super::ast::DataType as DataType;
pub use super::ast::StructuredType as StructuredType;
pub use super::ast::ObjectType as ObjectType;
pub use super::ast::Type as Type;
pub use super::ast::BinOp as BinOp;
pub use super::ast::UnaryOp as UnaryOp;

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {
    Float4(Box<Expression>, Box<Expression>, Box<Expression>, Box<Expression>),

    BufferLoad(Box<Expression>, Box<Expression>),
    StructuredBufferLoad(Box<Expression>, Box<Expression>),
}

pub use super::ast::Literal as Literal;

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(String),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(Box<Expression>, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Intrinsic),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub name: String,
    pub typename: Type,
    pub assignment: Option<Expression>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Condition {
    Expr(Expression),
    Assignment(VarDef),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Expression(Expression),
    Var(VarDef),
    Block(Vec<Statement>),
    If(Condition, Box<Statement>),
    For(Condition, Condition, Condition, Box<Statement>),
    While(Condition, Box<Statement>),
    Return(Expression),
}

pub use super::ast::StructMember as StructMember;
pub use super::ast::StructDefinition as StructDefinition;

pub use super::ast::PackSubOffset as PackSubOffset;
pub use super::ast::PackOffset as PackOffset;
pub use super::ast::ConstantVariable as ConstantVariable;

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantBuffer {
    pub name: String,
    pub members: Vec<ConstantVariable>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub name: String,
    pub typename: Type,
}

pub use super::ast::FunctionAttribute as FunctionAttribute;
pub use super::ast::FunctionParam as FunctionParam;

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub returntype: Type,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
    pub attributes: Vec<FunctionAttribute>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum KernelSemantic {
    DispatchThreadId,
    GroupId,
    GroupIndex,
    GroupThreadId,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Dimension(pub u64, pub u64, pub u64);

#[derive(PartialEq, Debug, Clone)]
pub struct KernelParam(pub String, pub KernelSemantic);

#[derive(PartialEq, Debug, Clone)]
pub struct Kernel {
    pub group_dimensions: Dimension,
    pub params: Vec<KernelParam>,
    pub body: Vec<Statement>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum RootDefinition {
    Struct(StructDefinition),
    SamplerState,
    ConstantBuffer(ConstantBuffer),
    GlobalVariable(GlobalVariable),
    Function(FunctionDefinition),
    Kernel(Kernel),
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalEntry {
    pub name: String,
    pub typename: ast::Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalTable {
    pub r_resources: HashMap<u32, GlobalEntry>,
    pub rw_resources: HashMap<u32, GlobalEntry>,
    pub samplers: HashMap<u32, String>,
    pub constants: HashMap<u32, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Module {
    pub entry_point: String,
    pub global_table: GlobalTable,
    pub root_definitions: Vec<RootDefinition>,
}

impl GlobalTable {
    pub fn new() -> GlobalTable {
        GlobalTable {
            r_resources: HashMap::new(),
            rw_resources: HashMap::new(),
            samplers: HashMap::new(),
            constants: HashMap::new(),
        }
    }
}

