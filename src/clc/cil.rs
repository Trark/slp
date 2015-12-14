
use std::collections::HashMap;
use BindMap;
use super::fragments::Fragment;

pub type Identifier = String;

pub use super::cst::AccessModifier as AccessModifier;
pub use super::cst::Scalar as Scalar;
pub use super::cst::VectorDimension as VectorDimension;

#[derive(PartialEq, Debug, Clone)]
pub enum Type {
    Void,
    Bool,
    Scalar(Scalar),
    Vector(Scalar, VectorDimension),
    SizeT,
    PtrDiffT,
    IntPtrT,
    UIntPtrT,
    Struct(StructId),
    Pointer(AddressSpace, Box<Type>),
    Array(Box<Type>, u64),

    Image1D(AccessModifier),
    Image1DBuffer(AccessModifier),
    Image1DArray(AccessModifier),
    Image2D(AccessModifier),
    Image2DArray(AccessModifier),
    Image2DDepth(AccessModifier),
    Image2DArrayDepth(AccessModifier),
    Image3D(AccessModifier),
    Image3DArray(AccessModifier),
    Sampler,

    Queue,
    NDRange,
    ClkEvent,
    ReserveId,
    Event,
    MemFenceFlags,
}

pub use super::cst::AddressSpace as AddressSpace;
pub use super::cst::BinOp as BinOp;
pub use super::cst::UnaryOp as UnaryOp;
pub use super::cst::Literal as Literal;

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct FunctionId(pub u32);
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct StructId(pub u32);
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct GlobalId(pub u32);
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct LocalId(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub struct LocalDeclarations {
    pub locals: HashMap<LocalId, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalDeclarations {
    pub functions: HashMap<FunctionId, String>,
    pub globals: HashMap<GlobalId, String>,
    pub structs: HashMap<StructId, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Constructor {
    UInt3(Box<Expression>, Box<Expression>, Box<Expression>),
    Float4(Box<Expression>, Box<Expression>, Box<Expression>, Box<Expression>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {
    GetGlobalId(Box<Expression>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Constructor(Constructor),
    Local(LocalId),
    Global(GlobalId),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, Identifier),
    Deref(Box<Expression>),
    MemberDeref(Box<Expression>, Identifier),
    AddressOf(Box<Expression>),
    Call(FunctionId, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Intrinsic),
    UntypedIntrinsic(String, Vec<Expression>),
    UntypedLiteral(String),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub id: LocalId,
    pub typename: Type,
    pub assignment: Option<Expression>
}

#[derive(PartialEq, Debug, Clone)]
pub enum Condition {
    Expr(Expression),
    Assignment(VarDef),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Empty,
    Expression(Expression),
    Var(VarDef),
    Block(Vec<Statement>),
    If(Expression, Box<Statement>),
    For(Condition, Expression, Expression, Box<Statement>),
    While(Expression, Box<Statement>),
    Return(Expression),
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub id: GlobalId,
    pub ty: Type,
    pub address_space: AddressSpace, // Make part of Type?
    pub init: Option<Expression>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructMember {
    pub name: Identifier,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructDefinition {
    pub id: StructId,
    pub members: Vec<StructMember>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionParam {
    pub id: LocalId,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub id: FunctionId,
    pub returntype: Type,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
    pub local_declarations: LocalDeclarations,
}

pub use super::cst::Dimension as Dimension;

#[derive(PartialEq, Debug, Clone)]
pub struct KernelParam {
    pub id: LocalId,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Kernel {
    pub params: Vec<KernelParam>,
    pub body: Vec<Statement>,
    pub group_dimensions: Dimension,
    pub local_declarations: LocalDeclarations,
}

#[derive(PartialEq, Debug, Clone)]
pub enum RootDefinition {
    GlobalVariable(GlobalVariable),
    Struct(StructDefinition),
    Function(FunctionDefinition),
    Kernel(Kernel),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Module {
    pub root_definitions: Vec<RootDefinition>,
    pub global_declarations: GlobalDeclarations,
    pub fragments: HashMap<Fragment, FunctionId>,
    pub binds: BindMap,
}
