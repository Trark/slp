
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

#[derive(PartialEq, Debug, Clone)]
pub enum DataLayout {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
}

/// A type that can be used in data buffers (Buffer / RWBuffer / etc)
/// These can interpret data in buffers bound with a format
/// FormatType might be a better name because they can bind resource
/// views with a format, but HLSL just called them Buffer and other
/// apis call them data buffers
#[derive(PartialEq, Debug, Clone)]
pub struct DataType(pub DataLayout, pub TypeModifier);

#[derive(PartialEq, Debug, Clone)]
pub enum StructuredLayout {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
    Custom(String), // Struct + User defined types
}

/// A type that can be used in structured buffers
/// These are the both struct defined types, the format data types
#[derive(PartialEq, Debug, Clone)]
pub struct StructuredType(pub StructuredLayout, pub TypeModifier);

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
pub enum TypeLayout {
    Void,
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
    Custom(String),
    SamplerState,
    Object(ObjectType),
    Array(Box<Type>),
}

impl TypeLayout {
    pub fn from_scalar(scalar: ScalarType) -> TypeLayout { TypeLayout::Scalar(scalar) }
    pub fn from_object(object: ObjectType) -> TypeLayout { TypeLayout::Object(object) }

    pub fn uint() -> TypeLayout { TypeLayout::from_scalar(ScalarType::UInt) }
    pub fn int() -> TypeLayout { TypeLayout::from_scalar(ScalarType::Int) }
    pub fn long() -> TypeLayout { TypeLayout::from_scalar(ScalarType::Int) }
    pub fn float() -> TypeLayout { TypeLayout::from_scalar(ScalarType::Float) }
    pub fn double() -> TypeLayout { TypeLayout::from_scalar(ScalarType::Double) }
    pub fn float4x4() -> TypeLayout { TypeLayout::Matrix(ScalarType::Float, 4, 4) }
    pub fn custom(name: &str) -> TypeLayout { TypeLayout::Custom(name.to_string()) }
}

impl From<DataLayout> for TypeLayout {
    fn from(data: DataLayout) -> TypeLayout {
        match data {
            DataLayout::Scalar(scalar) => TypeLayout::Scalar(scalar),
            DataLayout::Vector(scalar, x) => TypeLayout::Vector(scalar, x),
            DataLayout::Matrix(scalar, x, y) => TypeLayout::Matrix(scalar, x, y),
        }
    }
}

impl From<StructuredLayout> for TypeLayout {
    fn from(structured: StructuredLayout) -> TypeLayout {
        match structured {
            StructuredLayout::Scalar(scalar) => TypeLayout::Scalar(scalar),
            StructuredLayout::Vector(scalar, x) => TypeLayout::Vector(scalar, x),
            StructuredLayout::Matrix(scalar, x, y) => TypeLayout::Matrix(scalar, x, y),
            StructuredLayout::Custom(name) => TypeLayout::Custom(name),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum RowOrder {
    Row,
    Column
}

#[derive(PartialEq, Debug, Clone)]
pub struct TypeModifier {
    pub is_const: bool,
    pub row_order: RowOrder,
    pub precise: bool,
    pub volatile: bool,
}

impl Default for TypeModifier {
    fn default() -> TypeModifier {
        TypeModifier { is_const: false, row_order: RowOrder::Column, precise: false, volatile: false }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum InterpolationModifier {
    NoInterpolation,
    Linear,
    Centroid,
    NoPerspective,
    Sample,
}

#[derive(PartialEq, Debug, Clone)]
pub enum GlobalStorage {
    Static,
    GroupShared,
}

#[derive(PartialEq, Debug, Clone)]
pub enum InputModifier {
    In,
    Out,
    InOut,
}

impl Default for InputModifier {
    fn default() -> InputModifier { InputModifier::In }
}

#[derive(PartialEq, Debug, Clone)]
pub enum LocalStorage {
    Local,
    Static,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Type(pub TypeLayout, pub TypeModifier);

impl Type {
    pub fn void() -> Type { Type::from_layout(TypeLayout::Void) }
    pub fn from_layout(layout: TypeLayout) -> Type { Type(layout, TypeModifier::default()) }
    pub fn from_scalar(scalar: ScalarType) -> Type { Type::from_layout(TypeLayout::from_scalar(scalar)) }
    pub fn from_object(object: ObjectType) -> Type { Type::from_layout(TypeLayout::from_object(object)) }

    pub fn uint() -> Type { Type::from_layout(TypeLayout::uint()) }
    pub fn int()  -> Type { Type::from_layout(TypeLayout::int()) }
    pub fn long()  -> Type { Type::from_layout(TypeLayout::long()) }
    pub fn float()  -> Type { Type::from_layout(TypeLayout::float()) }
    pub fn double()  -> Type { Type::from_layout(TypeLayout::double()) }
    pub fn float4x4() -> Type { Type::from_layout(TypeLayout::float4x4()) }
    pub fn custom(name: &str)  -> Type { Type::from_layout(TypeLayout::custom(name)) }
}

impl From<DataType> for Type {
    fn from(ty: DataType) -> Type {
        let DataType(layout, modifier) = ty;
        Type(layout.into(), modifier)
    }
}

impl From<StructuredType> for Type {
    fn from(ty: StructuredType) -> Type {
        let StructuredType(layout, modifier) = ty;
        Type(layout.into(), modifier)
    }
}

/// The type of any global declaration
#[derive(PartialEq, Debug, Clone)]
pub struct GlobalType(pub Type, pub GlobalStorage, pub Option<InterpolationModifier>);

/// The type of any parameter declaration
#[derive(PartialEq, Debug, Clone)]
pub struct ParamType(pub Type, pub InputModifier, pub Option<InterpolationModifier>);

impl From<Type> for ParamType {
    fn from(ty: Type) -> ParamType {
        ParamType(ty, InputModifier::default(), None)
    }
}

/// The type of any local variable declaration
#[derive(PartialEq, Debug, Clone)]
pub struct LocalType(pub Type, pub LocalStorage, pub Option<InterpolationModifier>);

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
    Bool(bool),
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
    pub param_type: ParamType,
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
