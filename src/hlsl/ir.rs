
use std::collections::HashMap;

pub use super::ast::ScalarType as ScalarType;
pub use super::ast::DataType as DataType;

#[derive(PartialEq, Debug, Clone)]
pub enum StructuredType {
    Data(DataType),
    Struct(StructId),
}

#[derive(PartialEq, Debug, Clone)]
pub enum ObjectType {
    Buffer(DataType),
    RWBuffer(DataType),

    ByteAddressBuffer,
    RWByteAddressBuffer,

    StructuredBuffer(StructuredType),
    RWStructuredBuffer(StructuredType),
    AppendStructuredBuffer(StructuredType),
    ConsumeStructuredBuffer(StructuredType),

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

impl Type {
    pub fn from_scalar(scalar: ScalarType) -> Type { Type::Structured(StructuredType::Data(DataType::Scalar(scalar))) }
    pub fn from_vector(scalar: ScalarType, dimension: u32) -> Type { Type::Structured(StructuredType::Data(DataType::Vector(scalar, dimension))) }

    pub fn bool() -> Type { Type::from_scalar(ScalarType::Bool) }
    pub fn booln(dim: u32) -> Type { Type::from_vector(ScalarType::Bool, dim) }
    pub fn uint() -> Type { Type::from_scalar(ScalarType::UInt) }
    pub fn uintn(dim: u32) -> Type { Type::from_vector(ScalarType::UInt, dim) }
    pub fn int() -> Type { Type::from_scalar(ScalarType::Int) }
    pub fn intn(dim: u32) -> Type { Type::from_vector(ScalarType::Int, dim) }
    pub fn float() -> Type { Type::from_scalar(ScalarType::Float) }
    pub fn floatn(dim: u32) -> Type { Type::from_vector(ScalarType::Float, dim) }
    pub fn double() -> Type { Type::from_scalar(ScalarType::Double) }
    pub fn doublen(dim: u32) -> Type { Type::from_vector(ScalarType::Double, dim) }

    pub fn long() -> Type { Type::from_scalar(ScalarType::Int) }
    pub fn float4x4() -> Type { Type::Structured(StructuredType::Data(DataType::Matrix(ScalarType::Float, 4, 4))) }
}

#[derive(PartialEq, Debug, Clone)]
pub enum RowOrder {
    Row,
    Column
}

/// Modifier for type
#[derive(PartialEq, Debug, Clone)]
pub struct TypeModifier {
    is_const: bool,
    row_order: RowOrder,
    precise: bool,
    volatile: bool,
}

impl TypeModifier {
    pub fn new() -> TypeModifier {
        TypeModifier { is_const: false, row_order: RowOrder::Column, precise: false, volatile: false }
    }
}

/// Interpolation modifier: exists on all variable definitions
#[derive(PartialEq, Debug, Clone)]
pub enum InterpolationModifier {
    NoInterpolation,
    Linear,
    Centroid,
    NoPerspective,
    Sample,
}

/// Storage type for global variables
#[derive(PartialEq, Debug, Clone)]
pub enum GlobalStorage {
    /// Statically allocated thread-local variable in global scope
    Static,

    /// Shared between every thread in the work group
    GroupShared,

    // extern not supported because constant buffers exist
    // uniform not supported because constant buffers exist
}

/// Binding type for parameters
#[derive(PartialEq, Debug, Clone)]
pub enum InputModifier {
    /// Function input
    In,
    /// Function output (must be written)
    Out,
    /// Function input and output
    InOut,

    // uniform not supported because constant buffers exist
}

// Storage type for local variables
#[derive(PartialEq, Debug, Clone)]
pub enum LocalStorage {
    /// Statically allocated thread-local variable
    Local,

    /// Statically allocated thread-local variable that persists between function calls
    /// Essentially the same as global static but name-scoped into a function
    Static,
}

/// The full type when paired with modifiers
#[derive(PartialEq, Debug, Clone)]
pub struct QualifiedType(pub Type, pub TypeModifier);

/// Value type for subexpressions. Doesn't appear in ir tree, but used for
/// reasoning about intermediates
#[derive(PartialEq, Debug, Clone)]
pub enum ValueType {
    Lvalue,
    Rvalue,
}

/// Type for value intermediates. Doesn't appear in ir tree, but used for
/// reasoning about intermediates. Doesn't include function intermediates.
#[derive(PartialEq, Debug, Clone)]
pub struct ExpressionType(pub Type, pub TypeModifier, pub ValueType);

/// The type of any global declaration
#[derive(PartialEq, Debug, Clone)]
pub struct GlobalType(pub QualifiedType, pub GlobalStorage, pub InterpolationModifier);

/// The type of any parameter declaration
#[derive(PartialEq, Debug, Clone)]
pub struct ParamType(pub QualifiedType, pub InputModifier, pub InterpolationModifier);

/// The type of any local variable declaration
#[derive(PartialEq, Debug, Clone)]
pub struct LocalType(pub QualifiedType, pub LocalStorage, pub InterpolationModifier);

pub use super::ast::BinOp as BinOp;
pub use super::ast::UnaryOp as UnaryOp;

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {

    AllMemoryBarrier,
    AllMemoryBarrierWithGroupSync,

    AsIntU(Expression),
    AsIntU1(Expression),
    AsIntU2(Expression),
    AsIntU3(Expression),
    AsIntU4(Expression),
    AsIntF(Expression),
    AsIntF1(Expression),
    AsIntF2(Expression),
    AsIntF3(Expression),
    AsIntF4(Expression),

    AsUIntI(Expression),
    AsUIntI1(Expression),
    AsUIntI2(Expression),
    AsUIntI3(Expression),
    AsUIntI4(Expression),
    AsUIntF(Expression),
    AsUIntF1(Expression),
    AsUIntF2(Expression),
    AsUIntF3(Expression),
    AsUIntF4(Expression),

    AsFloatI(Expression),
    AsFloatI1(Expression),
    AsFloatI2(Expression),
    AsFloatI3(Expression),
    AsFloatI4(Expression),
    AsFloatU(Expression),
    AsFloatU1(Expression),
    AsFloatU2(Expression),
    AsFloatU3(Expression),
    AsFloatU4(Expression),
    AsFloatF(Expression),
    AsFloatF1(Expression),
    AsFloatF2(Expression),
    AsFloatF3(Expression),
    AsFloatF4(Expression),

    AsDouble(Expression, Expression),

    ClampI(Expression, Expression, Expression),
    ClampI1(Expression, Expression, Expression),
    ClampI2(Expression, Expression, Expression),
    ClampI3(Expression, Expression, Expression),
    ClampI4(Expression, Expression, Expression),
    ClampF(Expression, Expression, Expression),
    ClampF1(Expression, Expression, Expression),
    ClampF2(Expression, Expression, Expression),
    ClampF3(Expression, Expression, Expression),
    ClampF4(Expression, Expression, Expression),

    Min(Expression, Expression),
    Max(Expression, Expression),

    // Constructors are here for the moment but should
    // probably be their own dedicated node
    // The varying number of parameters makes them awkward to parse
    Float4(Expression, Expression, Expression, Expression),

    BufferLoad(Expression, Expression),
    StructuredBufferLoad(Expression, Expression),
}

pub use super::ast::Literal as Literal;

/// Id to function (in global scope)
pub type FunctionId = u32;
/// Id to a user defined struct
pub type StructId = u32;
/// Id to constant buffer
pub type ConstantBufferId = u32;
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
    pub structs: HashMap<StructId, String>,
    pub constants: HashMap<ConstantBufferId, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(VariableRef),
    ConstantVariable(ConstantBufferId, String),
    Function(FunctionId),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(Box<Expression>, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Box<Intrinsic>),
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
pub struct ScopeBlock(pub Vec<Statement>, pub ScopedDeclarations);

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Expression(Expression),
    Var(VarDef),
    Block(ScopeBlock),
    If(Expression, ScopeBlock),
    For(Condition, Expression, Expression, ScopeBlock),
    While(Expression, ScopeBlock),
    Return(Expression),
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructMember {
    pub name: String,
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructDefinition {
    pub id: StructId,
    pub members: Vec<StructMember>,
}

pub use super::ast::PackSubOffset as PackSubOffset;
pub use super::ast::PackOffset as PackOffset;

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantVariable {
    pub name: String,
    pub typename: Type,
    pub offset: Option<PackOffset>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantBuffer {
    pub id: ConstantBufferId,
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
    pub typename: Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalTable {
    pub r_resources: HashMap<u32, GlobalEntry>,
    pub rw_resources: HashMap<u32, GlobalEntry>,
    pub samplers: HashMap<u32, String>,
    pub constants: HashMap<u32, ConstantBufferId>,
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

