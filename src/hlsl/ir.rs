
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

/// Id to function (in global scope)
pub type FunctionId = u32;
/// Id to variable in current scope
pub type VariableId = u32;
/// Number of scope levels to go up
pub type ScopeRef = u32;
/// Reference to a variable, combining both id and scope level
#[derive(PartialEq, Debug, Clone)]
pub struct VariableRef(pub VariableId, pub ScopeRef);

/// Map of declarations in the current scope
#[derive(PartialEq, Debug, Clone)]
pub struct ScopedDeclarations {
    pub variables: HashMap<VariableId, String>,
}

/// Map of declarations in the global scope
#[derive(PartialEq, Debug, Clone)]
pub struct GlobalDeclarations {
    pub functions: HashMap<FunctionId, String>,
    pub variables: HashMap<VariableId, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(VariableRef),
    Function(FunctionId),
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
    pub id: VariableId,
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
    Block(Vec<Statement>, ScopedDeclarations),
    If(Condition, Box<Statement>, ScopedDeclarations),
    For(Condition, Condition, Condition, Box<Statement>, ScopedDeclarations),
    While(Condition, Box<Statement>, ScopedDeclarations),
    Return(Expression),
}

pub use super::ast::StructMember as StructMember;
pub use super::ast::StructDefinition as StructDefinition;

pub use super::ast::PackSubOffset as PackSubOffset;
pub use super::ast::PackOffset as PackOffset;
pub use super::ast::ConstantVariable as ConstantVariable;

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantBuffer {
    pub id: VariableId,
    pub members: Vec<ConstantVariable>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub id: VariableId,
    pub typename: Type,
}

pub use super::ast::FunctionAttribute as FunctionAttribute;

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionParam {
    pub id: VariableId,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub id: FunctionId,
    pub returntype: Type,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
    pub scope: ScopedDeclarations,
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
pub struct KernelParam(pub VariableId, pub KernelSemantic);

#[derive(PartialEq, Debug, Clone)]
pub struct Kernel {
    pub group_dimensions: Dimension,
    pub params: Vec<KernelParam>,
    pub body: Vec<Statement>,
    pub scope: ScopedDeclarations,
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
    pub id: VariableId,
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
    pub global_declarations: GlobalDeclarations,
    pub root_definitions: Vec<RootDefinition>,
}

impl KernelSemantic {
    pub fn get_type(&self) -> Type {
        match self {
            &KernelSemantic::DispatchThreadId => Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 3))),
            &KernelSemantic::GroupId => Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 3))),
            &KernelSemantic::GroupIndex => Type::Structured(StructuredType::Data(DataType::Scalar(ScalarType::UInt))),
            &KernelSemantic::GroupThreadId => Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 3))),
        }
    }
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

