
use std::collections::HashMap;
use std::collections::HashSet;
use slp_shared::BindMap;
use slp_lang_cst::fragments::Fragment;

pub type Identifier = String;

pub use slp_lang_cst::AccessModifier;
pub use slp_lang_cst::Scalar;
pub use slp_lang_cst::VectorDimension;
pub use slp_lang_cst::NumericDimension;

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

pub use slp_lang_cst::AddressSpace;
pub use slp_lang_cst::BinOp;
pub use slp_lang_cst::UnaryOp;
pub use slp_lang_cst::Literal;

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
pub enum SwizzleSlot {
    X,
    Y,
    Z,
    W,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {
    GetGlobalId(Box<Expression>),
    GetLocalId(Box<Expression>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Local(LocalId),
    Global(GlobalId),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    Swizzle(Box<Expression>, Vec<SwizzleSlot>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, Identifier),
    Deref(Box<Expression>),
    MemberDeref(Box<Expression>, Identifier),
    AddressOf(Box<Expression>),
    Call(FunctionId, Vec<Expression>),
    NumericConstructor(Scalar, NumericDimension, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Intrinsic),
    UntypedIntrinsic(String, Vec<Expression>),
    UntypedLiteral(String),
}

#[derive(PartialEq, Debug, Clone)]
/// The node for representing the initial value of a variable
pub enum Initializer {
    /// Variable is initialized to the value of an expression
    Expression(Expression),
    /// Variable is initialized in parts (composite types and arrays)
    Aggregate(Vec<Initializer>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub id: LocalId,
    pub typename: Type,
    pub init: Option<Initializer>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum InitStatement {
    Empty,
    Expression(Expression),
    Declaration(VarDef),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Empty,
    Expression(Expression),
    Var(VarDef),
    Block(Vec<Statement>),
    If(Expression, Box<Statement>),
    IfElse(Expression, Box<Statement>, Box<Statement>),
    For(InitStatement, Expression, Expression, Box<Statement>),
    While(Expression, Box<Statement>),
    Return(Expression),
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub id: GlobalId,
    pub ty: Type,
    pub address_space: AddressSpace, // Make part of Type?
    pub init: Option<Initializer>,
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

pub use slp_lang_cst::Dimension;

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

pub use slp_shared::opencl::Extension;

#[derive(PartialEq, Debug, Clone)]
pub struct Module {
    pub root_definitions: Vec<RootDefinition>,
    pub global_declarations: GlobalDeclarations,
    pub fragments: HashMap<Fragment, FunctionId>,
    pub binds: BindMap,
    pub required_extensions: HashSet<Extension>,
}
