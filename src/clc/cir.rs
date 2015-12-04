
use BindMap;

pub type Identifier = String;

#[derive(PartialEq, Debug, Clone)]
pub enum AccessModifier {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Scalar {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    Half,
    Float,
    Double,
}

#[derive(PartialEq, Debug, Clone)]
pub enum VectorDimension {
    Two,
    Three,
    Four,
    Eight,
    Sixteen,
}

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
    Struct(Identifier),
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

#[derive(PartialEq, Debug, Clone)]
pub enum AddressSpace {
    Private,
    Local,
    Constant,
    Global,
}

#[derive(PartialEq, Debug, Clone)]
pub enum BinOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    LeftShift,
    RightShift,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LogicalAnd,
    LogicalOr,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Equality,
    Inequality,
    Assignment,
}

#[derive(PartialEq, Debug, Clone)]
pub enum UnaryOp {
    PrefixIncrement,
    PrefixDecrement,
    PostfixIncrement,
    PostfixDecrement,
    Plus,
    Minus,
    LogicalNot,
    BitwiseNot,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Literal {
    Bool(bool),
    Int(u64),
    UInt(u64),
    Long(u64),
    Half(f32),
    Float(f32),
    Double(f64),
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
    Variable(Identifier),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, Identifier),
    Deref(Box<Expression>),
    MemberDeref(Box<Expression>, Identifier),
    AddressOf(Box<Expression>),
    Call(Box<Expression>, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Intrinsic),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub name: Identifier,
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
    pub name: Identifier,
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
    pub name: Identifier,
    pub members: Vec<StructMember>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionParam {
    pub name: Identifier,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub name: Identifier,
    pub returntype: Type,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Dimension(pub u64, pub u64, pub u64);

#[derive(PartialEq, Debug, Clone)]
pub struct KernelParam {
    pub name: Identifier,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Kernel {
    pub params: Vec<KernelParam>,
    pub body: Vec<Statement>,
    pub group_dimensions: Dimension,
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
    pub binds: BindMap,
}
