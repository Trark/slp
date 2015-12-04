
use std::collections::HashMap;

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

/// Layout for DataType
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

/// Layout for StructuredType
#[derive(PartialEq, Debug, Clone)]
pub enum StructuredLayout {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
    Struct(StructId),
}

/// A type that can be used in structured buffers
/// These are the both all the data types and user defined structs
#[derive(PartialEq, Debug, Clone)]
pub struct StructuredType(pub StructuredLayout, pub TypeModifier);

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
pub enum TypeLayout {
    Void,
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
    Struct(StructId),
    SamplerState,
    Object(ObjectType),
    Array(Box<TypeLayout>, u64),
}

impl TypeLayout {
    pub fn void() -> TypeLayout { TypeLayout::Void }
    pub fn from_scalar(scalar: ScalarType) -> TypeLayout { TypeLayout::Scalar(scalar) }
    pub fn from_vector(scalar: ScalarType, x: u32) -> TypeLayout { TypeLayout::Vector(scalar, x) }
    pub fn from_matrix(scalar: ScalarType, x: u32, y: u32) -> TypeLayout { TypeLayout::Matrix(scalar, x, y) }
    pub fn from_data(ty: DataLayout) -> TypeLayout { TypeLayout::from(ty) }
    pub fn from_struct(id: StructId) -> TypeLayout { TypeLayout::Struct(id) }
    pub fn from_structured(ty: StructuredLayout) -> TypeLayout { TypeLayout::from(ty) }
    pub fn from_object(ty: ObjectType) -> TypeLayout { TypeLayout::Object(ty) }
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
            StructuredLayout::Struct(id) => TypeLayout::Struct(id),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum RowOrder {
    Row,
    Column
}

/// Modifier for type
#[derive(PartialEq, Debug, Clone)]
pub struct TypeModifier {
    pub is_const: bool,
    pub row_order: RowOrder,
    pub precise: bool,
    pub volatile: bool,
}

impl TypeModifier {
    pub fn new() -> TypeModifier { Default::default() }

    pub fn is_empty(&self) -> bool {
        self.is_const == false && self.row_order == RowOrder::Column && self.precise == false && self.volatile == false
    }
}

impl Default for TypeModifier {
    fn default() -> TypeModifier {
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

    // Input from outside the kernel (default)
    Extern,

    /// Statically allocated thread-local variable in global scope
    Static,

    /// Shared between every thread in the work group
    GroupShared,

    // extern not supported because constant buffers exist
    // uniform not supported because constant buffers exist
}

impl Default for GlobalStorage {
    fn default() -> GlobalStorage { GlobalStorage::Static }
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

impl Default for InputModifier {
    fn default() -> InputModifier { InputModifier::In }
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

impl Default for LocalStorage {
    fn default() -> LocalStorage { LocalStorage::Local }
}

/// The full type when paired with modifiers
#[derive(PartialEq, Debug, Clone)]
pub struct Type(pub TypeLayout, pub TypeModifier);

impl Type {
    pub fn void() -> Type { Type(TypeLayout::void(), TypeModifier::new()) }
    pub fn from_layout(layout_type: TypeLayout) -> Type { Type(layout_type, TypeModifier::new()) }
    pub fn from_scalar(scalar: ScalarType) -> Type { Type(TypeLayout::from_scalar(scalar), TypeModifier::new()) }
    pub fn from_vector(scalar: ScalarType, x: u32) -> Type { Type(TypeLayout::from_vector(scalar, x), TypeModifier::new()) }
    pub fn from_matrix(scalar: ScalarType, x: u32, y: u32) -> Type { Type(TypeLayout::from_matrix(scalar, x, y), TypeModifier::new()) }
    pub fn from_data(DataType(tyl, tym): DataType) -> Type { Type(TypeLayout::from_data(tyl), tym) }
    pub fn from_struct(id: StructId) -> Type { Type(TypeLayout::from_struct(id), TypeModifier::new()) }
    pub fn from_structured(StructuredType(tyl, tym): StructuredType) -> Type { Type(TypeLayout::from_structured(tyl), tym) }
    pub fn from_object(ty: ObjectType) -> Type { Type(TypeLayout::from_object(ty), TypeModifier::new()) }

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
    pub fn float4x4() -> Type { Type::from_matrix(ScalarType::Float, 4, 4) }
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
pub struct ExpressionType(pub Type, pub ValueType);

pub trait ToExpressionType {
    fn to_lvalue(self) -> ExpressionType;
    fn to_rvalue(self) -> ExpressionType;
}

impl ToExpressionType for Type {
    fn to_lvalue(self) -> ExpressionType { ExpressionType(self, ValueType::Lvalue) }
    fn to_rvalue(self) -> ExpressionType { ExpressionType(self, ValueType::Rvalue) }
}

impl<'a> ToExpressionType for &'a Type {
    fn to_lvalue(self) -> ExpressionType { self.clone().to_lvalue() }
    fn to_rvalue(self) -> ExpressionType { self.clone().to_rvalue() }
}

/// The type of any global declaration
#[derive(PartialEq, Debug, Clone)]
pub struct GlobalType(pub Type, pub GlobalStorage, pub Option<InterpolationModifier>);

impl From<Type> for GlobalType {
    fn from(ty: Type) -> GlobalType {
        GlobalType(ty, GlobalStorage::default(), None)
    }
}

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

impl From<Type> for LocalType {
    fn from(ty: Type) -> LocalType {
        LocalType(ty, LocalStorage::default(), None)
    }
}

pub use super::ast::BinOp as BinOp;
pub use super::ast::UnaryOp as UnaryOp;

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {

    AllMemoryBarrier,
    AllMemoryBarrierWithGroupSync,
    DeviceMemoryBarrier,
    DeviceMemoryBarrierWithGroupSync,
    GroupMemoryBarrier,
    GroupMemoryBarrierWithGroupSync,

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

    Cross(Expression, Expression),

    Distance1(Expression, Expression),
    Distance2(Expression, Expression),
    Distance3(Expression, Expression),
    Distance4(Expression, Expression),

    DotI1(Expression, Expression),
    DotI2(Expression, Expression),
    DotI3(Expression, Expression),
    DotI4(Expression, Expression),
    DotF1(Expression, Expression),
    DotF2(Expression, Expression),
    DotF3(Expression, Expression),
    DotF4(Expression, Expression),

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
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct FunctionId(pub u32);
/// Id to a user defined struct
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct StructId(pub u32);
/// Id to constant buffer
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct ConstantBufferId(pub u32);
/// Id to a global variable
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct GlobalId(pub u32);
/// Id to variable in current scope
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct VariableId(pub u32);
/// Number of scope levels to go up
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct ScopeRef(pub u32);
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
    pub globals: HashMap<GlobalId, String>,
    pub structs: HashMap<StructId, String>,
    pub constants: HashMap<ConstantBufferId, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(VariableRef),
    Global(GlobalId),
    ConstantVariable(ConstantBufferId, String),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(FunctionId, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Box<Intrinsic>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub id: VariableId,
    pub local_type: LocalType,
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
    pub id: GlobalId,
    pub global_type: GlobalType,
    pub assignment: Option<Expression>,
}

pub use super::ast::FunctionAttribute as FunctionAttribute;

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionParam {
    pub id: VariableId,
    pub param_type: ParamType,
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
    pub id: GlobalId,
    pub ty: GlobalType,
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
        match *self {
            KernelSemantic::DispatchThreadId => Type::from_vector(ScalarType::UInt, 3),
            KernelSemantic::GroupId => Type::from_vector(ScalarType::UInt, 3),
            KernelSemantic::GroupIndex => Type::from_scalar(ScalarType::UInt),
            KernelSemantic::GroupThreadId => Type::from_vector(ScalarType::UInt, 3),
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

