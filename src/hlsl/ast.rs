
/// Basic scalar types
#[derive(PartialEq, Debug, Clone)]
pub enum ScalarType {
    Bool,
    UntypedInt,
    Int,
    UInt,
    Half,
    Float,
    Double,
}

/// A type that can be used in data buffers (Buffer / RWBuffer / etc)
/// These can interpret data in buffers bound with a format
/// FormatType might be a better name because they can bind resource
/// views with a format, but HLSL just called them Buffer and other
/// apis call them data buffers
#[derive(PartialEq, Debug, Clone)]
pub enum DataType {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
}

/// A type that can be used in structured buffers
/// These are the both struct defined types, the format data types
#[derive(PartialEq, Debug, Clone)]
pub enum StructuredType {
    Data(DataType),
    Custom(String), // Struct + User defined types
}

/// Hlsl Object Types
#[derive(PartialEq, Debug, Clone)]
pub enum ObjectType {

    // Data buffers
    Buffer(DataType),
    RWBuffer(DataType),

    // Raw buffers
    ByteAddressBuffer,
    RWByteAddressBuffer,

    // Structured buffers
    StructuredBuffer(StructuredType),
    RWStructuredBuffer(StructuredType),
    AppendStructuredBuffer(StructuredType),
    ConsumeStructuredBuffer(StructuredType),

    // Textures
    Texture1D(DataType),
    Texture1DArray(DataType),
    Texture2D(DataType),
    Texture2DArray(DataType),
    Texture2DMS(DataType),
    Texture2DMSArray(DataType),
    Texture3D(DataType),
    TextureCube(DataType),
    TextureCubeArray(DataType),
    RWTexture1D(DataType),
    RWTexture1DArray(DataType),
    RWTexture2D(DataType),
    RWTexture2DArray(DataType),
    RWTexture3D(DataType),

    // Tesselation patches
    InputPatch,
    OutputPatch,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Type {
    Void,
    Structured(StructuredType),
    SamplerState,
    Object(ObjectType),
    Array(Box<Type>),
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
    UntypedInt(u64),
    Int(u64),
    UInt(u64),
    Long(u64),
    Half(f32),
    Float(f32),
    Double(f64),
}

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
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub name: String,
    pub typename: Type,
    pub assignment: Option<Expression>,
}

impl VarDef {
    pub fn new(name: String, typename: Type, assignment: Option<Expression>) -> VarDef {
        VarDef { name: name, typename: typename, assignment: assignment }
    }
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
pub struct StructMember {
    pub name: String,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructDefinition {
    pub name: String,
    pub members: Vec<StructMember>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantSlot(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub enum PackSubOffset {
    None,
    X,
    Y,
    Z,
    W,
}

#[derive(PartialEq, Debug, Clone)]
pub struct PackOffset(pub u32, pub PackSubOffset);

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantVariable {
    pub name: String,
    pub typename: Type,
    pub offset: Option<PackOffset>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantBuffer {
    pub name: String,
    pub slot: Option<ConstantSlot>,
    pub members: Vec<ConstantVariable>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct SamplerSlot(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub enum GlobalSlot {
    ReadSlot(u32),
    ReadWriteSlot(u32),
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub name: String,
    pub typename: Type,
    pub slot: Option<GlobalSlot>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Semantic {
    DispatchThreadId,
    GroupId,
    GroupIndex,
    GroupThreadId,
}

#[derive(PartialEq, Debug, Clone)]
pub enum FunctionAttribute {
    NumThreads(u64, u64, u64),
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionParam {
    pub name: String,
    pub typename: Type,
    pub semantic: Option<Semantic>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub returntype: Type,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
    pub attributes: Vec<FunctionAttribute>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum RootDefinition {
    Struct(StructDefinition),
    SamplerState,
    ConstantBuffer(ConstantBuffer),
    GlobalVariable(GlobalVariable),
    Function(FunctionDefinition),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Module {
    pub entry_point: String,
    pub root_definitions: Vec<RootDefinition>,
}


impl Type {
    pub fn from_scalar(scalar: ScalarType) -> Type { Type::Structured(StructuredType::Data(DataType::Scalar(scalar))) }

    pub fn uint() -> Type { Type::from_scalar(ScalarType::UInt) }
    pub fn int() -> Type { Type::from_scalar(ScalarType::Int) }
    pub fn long() -> Type { Type::from_scalar(ScalarType::Int) }
    pub fn float() -> Type { Type::from_scalar(ScalarType::Float) }
    pub fn double() -> Type { Type::from_scalar(ScalarType::Double) }
    pub fn float4x4() -> Type { Type::Structured(StructuredType::Data(DataType::Matrix(ScalarType::Float, 4, 4))) }
    pub fn custom(name: &str) -> Type { Type::Structured(StructuredType::Custom(name.to_string())) }
}
