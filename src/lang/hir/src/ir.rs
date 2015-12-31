
use std::error;
use std::fmt;
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

impl DataLayout {
    pub fn to_scalar(&self) -> ScalarType {
        match *self {
            DataLayout::Scalar(ref scalar) |
            DataLayout::Vector(ref scalar, _) |
            DataLayout::Matrix(ref scalar, _, _) => scalar.clone(),
        }
    }
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
    pub fn void() -> TypeLayout {
        TypeLayout::Void
    }
    pub fn from_scalar(scalar: ScalarType) -> TypeLayout {
        TypeLayout::Scalar(scalar)
    }
    pub fn from_vector(scalar: ScalarType, x: u32) -> TypeLayout {
        TypeLayout::Vector(scalar, x)
    }
    pub fn from_matrix(scalar: ScalarType, x: u32, y: u32) -> TypeLayout {
        TypeLayout::Matrix(scalar, x, y)
    }
    pub fn from_data(ty: DataLayout) -> TypeLayout {
        TypeLayout::from(ty)
    }
    pub fn from_struct(id: StructId) -> TypeLayout {
        TypeLayout::Struct(id)
    }
    pub fn from_structured(ty: StructuredLayout) -> TypeLayout {
        TypeLayout::from(ty)
    }
    pub fn from_object(ty: ObjectType) -> TypeLayout {
        TypeLayout::Object(ty)
    }
    pub fn to_scalar(&self) -> Option<ScalarType> {
        match *self {
            TypeLayout::Scalar(ref scalar) |
            TypeLayout::Vector(ref scalar, _) |
            TypeLayout::Matrix(ref scalar, _, _) => Some(scalar.clone()),
            _ => None,
        }
    }
    pub fn to_x(&self) -> Option<u32> {
        match *self {
            TypeLayout::Vector(_, ref x) => Some(*x),
            TypeLayout::Matrix(_, ref x, _) => Some(*x),
            _ => None,
        }
    }
    pub fn to_y(&self) -> Option<u32> {
        match *self {
            TypeLayout::Matrix(_, _, ref y) => Some(*y),
            _ => None,
        }
    }
    pub fn max_dim(r1: Option<u32>, r2: Option<u32>) -> Option<u32> {
        use std::cmp::max;
        match (r1, r2) {
            (Some(x1), Some(x2)) => Some(max(x1, x2)),
            (Some(x1), None) => Some(x1),
            (None, Some(x2)) => Some(x2),
            (None, None) => None,
        }
    }
    pub fn get_num_elements(&self) -> u32 {
        match (self.to_x(), self.to_y()) {
            (Some(x1), Some(x2)) => x1 * x2,
            (Some(x1), None) => x1,
            (None, Some(x2)) => x2,
            (None, None) => 1,
        }
    }
    pub fn from_numeric(scalar: ScalarType, x_opt: Option<u32>, y_opt: Option<u32>) -> TypeLayout {
        match (x_opt, y_opt) {
            (Some(x), Some(y)) => TypeLayout::Matrix(scalar, x, y),
            (Some(x), None) => TypeLayout::Vector(scalar, x),
            (None, None) => TypeLayout::Scalar(scalar),
            _ => panic!("invalid numeric type"),
        }
    }
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
    Column,
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
    pub fn new() -> TypeModifier {
        Default::default()
    }

    pub fn is_empty(&self) -> bool {
        self.is_const == false && self.row_order == RowOrder::Column && self.precise == false &&
        self.volatile == false
    }

    pub fn const_only() -> TypeModifier {
        TypeModifier { is_const: true, ..TypeModifier::default() }
    }

    pub fn keep_precise(&self) -> TypeModifier {
        TypeModifier { precise: self.precise, ..TypeModifier::default() }
    }
}

impl Default for TypeModifier {
    fn default() -> TypeModifier {
        TypeModifier {
            is_const: false,
            row_order: RowOrder::Column,
            precise: false,
            volatile: false,
        }
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
    GroupShared, /* extern not supported because constant buffers exist
                  * uniform not supported because constant buffers exist */
}

impl Default for GlobalStorage {
    fn default() -> GlobalStorage {
        GlobalStorage::Static
    }
}

/// Binding type for parameters
#[derive(PartialEq, Debug, Clone)]
pub enum InputModifier {
    /// Function input
    In,
    /// Function output (must be written)
    Out,
    /// Function input and output
    InOut, // uniform not supported because constant buffers exist
}

impl Default for InputModifier {
    fn default() -> InputModifier {
        InputModifier::In
    }
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
    fn default() -> LocalStorage {
        LocalStorage::Local
    }
}

/// The full type when paired with modifiers
#[derive(PartialEq, Debug, Clone)]
pub struct Type(pub TypeLayout, pub TypeModifier);

impl Type {
    pub fn void() -> Type {
        Type(TypeLayout::void(), TypeModifier::new())
    }
    pub fn from_layout(layout_type: TypeLayout) -> Type {
        Type(layout_type, TypeModifier::new())
    }
    pub fn from_scalar(scalar: ScalarType) -> Type {
        Type(TypeLayout::from_scalar(scalar), TypeModifier::new())
    }
    pub fn from_vector(scalar: ScalarType, x: u32) -> Type {
        Type(TypeLayout::from_vector(scalar, x), TypeModifier::new())
    }
    pub fn from_matrix(scalar: ScalarType, x: u32, y: u32) -> Type {
        Type(TypeLayout::from_matrix(scalar, x, y), TypeModifier::new())
    }
    pub fn from_data(DataType(tyl, tym): DataType) -> Type {
        Type(TypeLayout::from_data(tyl), tym)
    }
    pub fn from_struct(id: StructId) -> Type {
        Type(TypeLayout::from_struct(id), TypeModifier::new())
    }
    pub fn from_structured(StructuredType(tyl, tym): StructuredType) -> Type {
        Type(TypeLayout::from_structured(tyl), tym)
    }
    pub fn from_object(ty: ObjectType) -> Type {
        Type(TypeLayout::from_object(ty), TypeModifier::new())
    }

    pub fn bool() -> Type {
        Type::from_scalar(ScalarType::Bool)
    }
    pub fn booln(dim: u32) -> Type {
        Type::from_vector(ScalarType::Bool, dim)
    }
    pub fn uint() -> Type {
        Type::from_scalar(ScalarType::UInt)
    }
    pub fn uintn(dim: u32) -> Type {
        Type::from_vector(ScalarType::UInt, dim)
    }
    pub fn int() -> Type {
        Type::from_scalar(ScalarType::Int)
    }
    pub fn intn(dim: u32) -> Type {
        Type::from_vector(ScalarType::Int, dim)
    }
    pub fn float() -> Type {
        Type::from_scalar(ScalarType::Float)
    }
    pub fn floatn(dim: u32) -> Type {
        Type::from_vector(ScalarType::Float, dim)
    }
    pub fn double() -> Type {
        Type::from_scalar(ScalarType::Double)
    }
    pub fn doublen(dim: u32) -> Type {
        Type::from_vector(ScalarType::Double, dim)
    }

    pub fn long() -> Type {
        Type::from_scalar(ScalarType::Int)
    }
    pub fn float4x4() -> Type {
        Type::from_matrix(ScalarType::Float, 4, 4)
    }
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
    fn to_lvalue(self) -> ExpressionType {
        ExpressionType(self, ValueType::Lvalue)
    }
    fn to_rvalue(self) -> ExpressionType {
        ExpressionType(self, ValueType::Rvalue)
    }
}

impl<'a> ToExpressionType for &'a Type {
    fn to_lvalue(self) -> ExpressionType {
        self.clone().to_lvalue()
    }
    fn to_rvalue(self) -> ExpressionType {
        self.clone().to_rvalue()
    }
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

pub use slp_lang_hst::BinOp;

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {
    // Unary operations
    PrefixIncrement(Type, Expression),
    PrefixDecrement(Type, Expression),
    PostfixIncrement(Type, Expression),
    PostfixDecrement(Type, Expression),
    Plus(Type, Expression),
    Minus(Type, Expression),
    LogicalNot(Type, Expression),
    BitwiseNot(Type, Expression),

    AllMemoryBarrier,
    AllMemoryBarrierWithGroupSync,
    DeviceMemoryBarrier,
    DeviceMemoryBarrierWithGroupSync,
    GroupMemoryBarrier,
    GroupMemoryBarrierWithGroupSync,

    AsIntU(Expression),
    AsIntU2(Expression),
    AsIntU3(Expression),
    AsIntU4(Expression),
    AsIntF(Expression),
    AsIntF2(Expression),
    AsIntF3(Expression),
    AsIntF4(Expression),

    AsUIntI(Expression),
    AsUIntI2(Expression),
    AsUIntI3(Expression),
    AsUIntI4(Expression),
    AsUIntF(Expression),
    AsUIntF2(Expression),
    AsUIntF3(Expression),
    AsUIntF4(Expression),

    AsFloatI(Expression),
    AsFloatI2(Expression),
    AsFloatI3(Expression),
    AsFloatI4(Expression),
    AsFloatU(Expression),
    AsFloatU2(Expression),
    AsFloatU3(Expression),
    AsFloatU4(Expression),
    AsFloatF(Expression),
    AsFloatF2(Expression),
    AsFloatF3(Expression),
    AsFloatF4(Expression),

    AsDouble(Expression, Expression),

    ClampI(Expression, Expression, Expression),
    ClampI2(Expression, Expression, Expression),
    ClampI3(Expression, Expression, Expression),
    ClampI4(Expression, Expression, Expression),
    ClampF(Expression, Expression, Expression),
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

    Floor(Expression),
    Floor2(Expression),
    Floor3(Expression),
    Floor4(Expression),

    Min(Expression, Expression),
    Max(Expression, Expression),

    Normalize1(Expression),
    Normalize2(Expression),
    Normalize3(Expression),
    Normalize4(Expression),

    SignI(Expression),
    SignI2(Expression),
    SignI3(Expression),
    SignI4(Expression),
    SignF(Expression),
    SignF2(Expression),
    SignF3(Expression),
    SignF4(Expression),

    Sqrt(Expression),
    Sqrt2(Expression),
    Sqrt3(Expression),
    Sqrt4(Expression),

    BufferLoad(Expression, Expression),
    RWBufferLoad(Expression, Expression),
    StructuredBufferLoad(Expression, Expression),
    RWStructuredBufferLoad(Expression, Expression),
    RWTexture2DLoad(Expression, Expression),

    // ByteAddressBuffer methods
    ByteAddressBufferLoad(Expression, Expression),
    ByteAddressBufferLoad2(Expression, Expression),
    ByteAddressBufferLoad3(Expression, Expression),
    ByteAddressBufferLoad4(Expression, Expression),
    RWByteAddressBufferLoad(Expression, Expression),
    RWByteAddressBufferLoad2(Expression, Expression),
    RWByteAddressBufferLoad3(Expression, Expression),
    RWByteAddressBufferLoad4(Expression, Expression),
    RWByteAddressBufferStore(Expression, Expression, Expression),
    RWByteAddressBufferStore2(Expression, Expression, Expression),
    RWByteAddressBufferStore3(Expression, Expression, Expression),
    RWByteAddressBufferStore4(Expression, Expression, Expression),
}

pub use slp_lang_hst::Literal;

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
    pub variables: HashMap<VariableId, (String, Type)>,
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
pub enum SwizzleSlot {
    X, // x or r
    Y, // y or g
    Z, // z or b
    W, // w or a
}

/// Element passed to a numeric constructor
/// Constructors can take variable numbers of arguments depending on dimensions
/// of the types of the input expressions
#[derive(PartialEq, Debug, Clone)]
pub struct ConstructorSlot {
    /// Vector dimension or Matrix total element count or 1 for scalars
    pub arity: u32,
    /// The expression argument for this slot
    /// The type of this expression must be the scalar type of the numeric
    /// constructor this is used in with the arity above
    pub expr: Expression,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(VariableRef),
    Global(GlobalId),
    ConstantVariable(ConstantBufferId, String),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    Swizzle(Box<Expression>, Vec<SwizzleSlot>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(FunctionId, Vec<Expression>),
    /// Constructors for builtin numeric types, such as `float2(1.0, 0.0)`
    NumericConstructor(DataLayout, Vec<ConstructorSlot>),
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
pub enum ForInit {
    Expression(Expression),
    Definitions(Vec<VarDef>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct ScopeBlock(pub Vec<Statement>, pub ScopedDeclarations);

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Expression(Expression),
    Var(VarDef),
    Block(ScopeBlock),
    If(Expression, ScopeBlock),
    IfElse(Expression, ScopeBlock, ScopeBlock),
    For(ForInit, Expression, Expression, ScopeBlock),
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

pub use slp_lang_hst::PackSubOffset;
pub use slp_lang_hst::PackOffset;

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

pub use slp_lang_hst::FunctionAttribute;

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
    pub scope_block: ScopeBlock,
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
    pub scope_block: ScopeBlock,
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

impl Default for GlobalTable {
    fn default() -> GlobalTable {
        GlobalTable {
            r_resources: HashMap::new(),
            rw_resources: HashMap::new(),
            samplers: HashMap::new(),
            constants: HashMap::new(),
        }
    }
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

/// Error in parsing the type of an expression
/// These will all be internal errors as they represent eeither incorrectly
/// generated ir trees or incorrectly tracking the ids in a tree
#[derive(PartialEq, Debug, Clone)]
pub enum TypeError {
    InvalidLocal,
    LocalScopeInvalid, // Same as LocalDoesNotExist but when ref is hard to calculate
    LocalDoesNotExist(VariableRef),
    GlobalDoesNotExist(GlobalId),
    ConstantBufferDoesNotExist(ConstantBufferId),
    ConstantDoesNotExist(ConstantBufferId, String),
    StructDoesNotExist(StructId),
    StructMemberDoesNotExist(StructId, String),
    FunctionDoesNotExist(FunctionId),

    InvalidTypeInEqualityOperation(TypeLayout),
    InvalidTypeForSwizzle(TypeLayout),
    MemberNodeMustBeUsedOnStruct(TypeLayout, String),
    ArrayIndexMustBeUsedOnArrayType(TypeLayout),

    InvalidType(Type),

    WrongObjectForBufferLoad(TypeLayout),
    WrongObjectForRWBufferLoad(TypeLayout),
    WrongObjectForStructuredBufferLoad(TypeLayout),
    WrongObjectForRWStructuredBufferLoad(TypeLayout),
    WrongObjectForRWTexture2DLoad(TypeLayout),
}

impl error::Error for TypeError {
    fn description(&self) -> &str {
        match *self {
            TypeError::InvalidLocal => "invalid local variable",
            TypeError::LocalScopeInvalid => "scope invalid",
            TypeError::LocalDoesNotExist(_) => "local variable does not exist",
            TypeError::GlobalDoesNotExist(_) => "global variable does not exist",
            TypeError::ConstantBufferDoesNotExist(_) => "constant buffer does not exist",
            TypeError::ConstantDoesNotExist(_, _) => "constant variable does not exist",
            TypeError::StructDoesNotExist(_) => "struct does not exist",
            TypeError::StructMemberDoesNotExist(_, _) => "struct member does not exist",
            TypeError::FunctionDoesNotExist(_) => "function does not exist",
            TypeError::InvalidTypeInEqualityOperation(_) => {
                "invalid numeric type in equality operation"
            }
            TypeError::InvalidTypeForSwizzle(_) => "swizzle nodes must be used on vectors",
            TypeError::MemberNodeMustBeUsedOnStruct(_, _) => "member used on non-struct type",
            TypeError::ArrayIndexMustBeUsedOnArrayType(_) => "array index used on non-array type",

            TypeError::InvalidType(_) => "invalid type in an intrinsic function",

            TypeError::WrongObjectForBufferLoad(_) => "Buffer::Load must be called on a buffer",
            TypeError::WrongObjectForRWBufferLoad(_) => "RWBuffer::Load must be called on a buffer",
            TypeError::WrongObjectForStructuredBufferLoad(_) => {
                "StructuredBuffer::Load must be called on a buffer"
            }
            TypeError::WrongObjectForRWStructuredBufferLoad(_) => {
                "RWStructuredBuffer::Load must be called on a buffer"
            }
            TypeError::WrongObjectForRWTexture2DLoad(_) => {
                "RWTexture2DLoad::Load must be called on a buffer"
            }
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

/// An object to hold all context of the type of definitions at a point in
/// the program
pub trait TypeContext {
    fn get_local(&self, var_ref: &VariableRef) -> Result<ExpressionType, TypeError>;
    fn get_global(&self, id: &GlobalId) -> Result<ExpressionType, TypeError>;
    fn get_constant(&self, id: &ConstantBufferId, name: &str) -> Result<ExpressionType, TypeError>;
    fn get_struct_member(&self, id: &StructId, name: &str) -> Result<ExpressionType, TypeError>;
    fn get_function_return(&self, id: &FunctionId) -> Result<ExpressionType, TypeError>;
}

/// A block of state for parsing complete ir trees
/// Implements TypeContext so types of defined nodes at a certain point in the
/// program can be queried
pub struct TypeState {
    globals: HashMap<GlobalId, Type>,
    constants: HashMap<ConstantBufferId, HashMap<String, Type>>,
    structs: HashMap<StructId, HashMap<String, Type>>,
    functions: HashMap<FunctionId, Type>, // Return type
    locals: Vec<HashMap<VariableId, Type>>,
}

impl TypeState {
    pub fn from_roots(root_definitions: &[RootDefinition]) -> Result<TypeState, ()> {
        let mut context = TypeState {
            globals: HashMap::new(),
            constants: HashMap::new(),
            structs: HashMap::new(),
            functions: HashMap::new(),
            locals: vec![],
        };
        for root_definition in root_definitions {
            match *root_definition {
                RootDefinition::Struct(ref sd) => {
                    let mut name_map = HashMap::new();
                    for constant in &sd.members {
                        match name_map.insert(constant.name.clone(), constant.typename.clone()) {
                            None => {}
                            Some(_) => return Err(()),
                        }
                    }
                    match context.structs.insert(sd.id, name_map) {
                        None => {}
                        Some(_) => return Err(()),
                    }
                }
                RootDefinition::SamplerState => unimplemented!(),
                RootDefinition::GlobalVariable(ref gv) => {
                    match context.globals.insert(gv.id.clone(), gv.global_type.0.clone()) {
                        None => {}
                        Some(_) => return Err(()),
                    }
                }
                RootDefinition::ConstantBuffer(ref cb) => {
                    let mut name_map = HashMap::new();
                    for constant in &cb.members {
                        match name_map.insert(constant.name.clone(), constant.typename.clone()) {
                            None => {}
                            Some(_) => return Err(()),
                        }
                    }
                    match context.constants.insert(cb.id, name_map) {
                        None => {}
                        Some(_) => return Err(()),
                    }
                }
                RootDefinition::Function(ref func) => {
                    match context.functions.insert(func.id.clone(), func.returntype.clone()) {
                        None => {}
                        Some(_) => return Err(()),
                    }
                }
                RootDefinition::Kernel(_) => {}
            }
        }
        Ok(context)
    }

    pub fn push_scope(&mut self, scope_block: &ScopeBlock) {
        let mut scope = HashMap::new();
        for (var_id, &(_, ref ty)) in &scope_block.1.variables {
            match scope.insert(var_id.clone(), ty.clone()) {
                None => {}
                Some(_) => panic!("Multiple locals in block with same id"),
            }
        }
        self.locals.push(scope);
    }

    pub fn pop_scope(&mut self) {
        assert!(self.locals.len() > 0);
        self.locals.pop();
    }
}

impl TypeContext for TypeState {
    fn get_local(&self, var_ref: &VariableRef) -> Result<ExpressionType, TypeError> {
        let up = (var_ref.1).0 as usize;
        if self.locals.len() <= up {
            return Err(TypeError::LocalScopeInvalid);
        } else {
            let level = self.locals.len() - up - 1;
            match self.locals[level].get(&var_ref.0) {
                Some(ref ty) => Ok(ty.to_lvalue()),
                None => Err(TypeError::LocalDoesNotExist(var_ref.clone())),
            }
        }
    }

    fn get_global(&self, id: &GlobalId) -> Result<ExpressionType, TypeError> {
        match self.globals.get(id) {
            Some(ref ty) => Ok(ty.to_lvalue()),
            None => Err(TypeError::GlobalDoesNotExist(id.clone())),
        }
    }

    fn get_constant(&self, id: &ConstantBufferId, name: &str) -> Result<ExpressionType, TypeError> {
        match self.constants.get(id) {
            Some(ref cm) => {
                match cm.get(name) {
                    Some(ref ty) => Ok(ty.to_lvalue()),
                    None => Err(TypeError::ConstantDoesNotExist(id.clone(), name.to_string())),
                }
            }
            None => Err(TypeError::ConstantBufferDoesNotExist(id.clone())),
        }
    }

    fn get_struct_member(&self, id: &StructId, name: &str) -> Result<ExpressionType, TypeError> {
        match self.structs.get(&id) {
            Some(ref cm) => {
                match cm.get(name) {
                    Some(ref ty) => Ok(ty.to_lvalue()),
                    None => Err(TypeError::StructMemberDoesNotExist(id.clone(), name.to_string())),
                }
            }
            None => Err(TypeError::StructDoesNotExist(id.clone())),
        }
    }

    fn get_function_return(&self, id: &FunctionId) -> Result<ExpressionType, TypeError> {
        match self.functions.get(id) {
            Some(ref ty) => Ok(ty.to_rvalue()),
            None => Err(TypeError::FunctionDoesNotExist(id.clone())),
        }
    }
}

pub struct TypeParser;

impl TypeParser {
    fn get_literal_type(literal: &Literal) -> ExpressionType {
        (match *literal {
            Literal::Bool(_) => Type::bool(),
            Literal::UntypedInt(_) => Type::from_scalar(ScalarType::UntypedInt),
            Literal::Int(_) => Type::int(),
            Literal::UInt(_) => Type::uint(),
            Literal::Long(_) => unimplemented!(),
            Literal::Half(_) => Type::from_scalar(ScalarType::Half),
            Literal::Float(_) => Type::float(),
            Literal::Double(_) => Type::double(),
        })
        .to_rvalue()
    }

    pub fn get_expression_type(expression: &Expression,
                               context: &TypeContext)
                               -> Result<ExpressionType, TypeError> {
        match *expression {
            Expression::Literal(ref lit) => Ok(TypeParser::get_literal_type(lit)),
            Expression::Variable(ref var_ref) => context.get_local(var_ref),
            Expression::Global(ref id) => context.get_global(id),
            Expression::ConstantVariable(ref id, ref name) => context.get_constant(id, name),
            Expression::BinaryOperation(ref op, ref expr, _) => {
                match *op {
                    BinOp::Add |
                    BinOp::Subtract |
                    BinOp::Multiply |
                    BinOp::Divide |
                    BinOp::Modulus |
                    BinOp::LeftShift |
                    BinOp::RightShift |
                    BinOp::BitwiseAnd |
                    BinOp::BitwiseOr |
                    BinOp::BitwiseXor => {
                        let base_type = try!(TypeParser::get_expression_type(expr, context)).0;
                        Ok(ExpressionType(base_type, ValueType::Rvalue))
                    }

                    BinOp::Assignment |
                    BinOp::SumAssignment |
                    BinOp::DifferenceAssignment => TypeParser::get_expression_type(expr, context),

                    BinOp::LessThan |
                    BinOp::LessEqual |
                    BinOp::GreaterThan |
                    BinOp::GreaterEqual |
                    BinOp::Equality |
                    BinOp::Inequality |
                    BinOp::BooleanAnd |
                    BinOp::BooleanOr => {
                        let ExpressionType(Type(ref tyl, _), _) =
                            try!(TypeParser::get_expression_type(expr, context));
                        Ok(ExpressionType(Type(match *tyl {
                                                   TypeLayout::Scalar(_) => {
                                                       TypeLayout::Scalar(ScalarType::Bool)
                                                   }
                                                   TypeLayout::Vector(_, ref x) => {
                                                       TypeLayout::Vector(ScalarType::Bool, *x)
                                                   }
                                                   _ => return Err(TypeError::InvalidTypeInEqualityOperation(tyl.clone())),
                                               },
                                               TypeModifier::default()),
                                          ValueType::Rvalue))
                    }
                }
            }
            Expression::TernaryConditional(_, ref expr_left, ref expr_right) => {
                // Ensure the layouts of each side are the same
                // Value types + modifiers can be different
                assert_eq!((try!(TypeParser::get_expression_type(expr_left, context)).0).0,
                           (try!(TypeParser::get_expression_type(expr_right, context)).0).0);
                TypeParser::get_expression_type(expr_left, context)
            }
            Expression::Swizzle(ref vec, ref swizzle) => {
                let ExpressionType(Type(vec_tyl, vec_mod), vec_vt) =
                    try!(TypeParser::get_expression_type(vec, context));
                let tyl = match vec_tyl {
                    TypeLayout::Vector(ref scalar, _) => {
                        if swizzle.len() == 1 {
                            TypeLayout::Scalar(scalar.clone())
                        } else {
                            TypeLayout::Vector(scalar.clone(), swizzle.len() as u32)
                        }
                    }
                    _ => return Err(TypeError::InvalidTypeForSwizzle(vec_tyl.clone())),
                };
                Ok(ExpressionType(Type(tyl, vec_mod), vec_vt))
            }
            Expression::ArraySubscript(ref array, _) => {
                let array_ty = try!(TypeParser::get_expression_type(&array, context));
                Ok(match (array_ty.0).0 {
                    TypeLayout::Array(ref element, _) => {
                        Type::from_layout(*element.clone()).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::Buffer(data_type)) => {
                        Type::from_data(data_type).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::RWBuffer(data_type)) => {
                        Type::from_data(data_type).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::StructuredBuffer(structured_type)) => {
                        Type::from_structured(structured_type).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::RWStructuredBuffer(structured_type)) => {
                        Type::from_structured(structured_type).to_lvalue()
                    }
                    tyl => return Err(TypeError::ArrayIndexMustBeUsedOnArrayType(tyl)),
                })
            }
            Expression::Member(ref expr, ref name) => {
                let expr_type = try!(TypeParser::get_expression_type(&expr, context));
                let id = match (expr_type.0).0 {
                    TypeLayout::Struct(id) => id,
                    tyl => {
                        return Err(TypeError::MemberNodeMustBeUsedOnStruct(tyl.clone(),
                                                                           name.clone()))
                    }
                };
                context.get_struct_member(&id, name)
            }
            Expression::Call(ref id, _) => context.get_function_return(id),
            Expression::NumericConstructor(ref dtyl, _) => {
                Ok(Type::from_layout(TypeLayout::from_data(dtyl.clone())).to_rvalue())
            }
            Expression::Cast(ref ty, _) => Ok(ty.to_rvalue()),
            Expression::Intrinsic(ref intrinsic) => {
                TypeParser::get_intrinsic_type(intrinsic, context)
            }
        }
    }

    fn get_intrinsic_type(intrinsic: &Intrinsic,
                          context: &TypeContext)
                          -> Result<ExpressionType, TypeError> {
        Ok(match *intrinsic {
            Intrinsic::PrefixIncrement(ref ty, _) => ty.to_lvalue(),
            Intrinsic::PrefixDecrement(ref ty, _) => ty.to_lvalue(),
            Intrinsic::PostfixIncrement(ref ty, _) => ty.to_lvalue(),
            Intrinsic::PostfixDecrement(ref ty, _) => ty.to_lvalue(),
            Intrinsic::Plus(ref ty, _) => ty.to_rvalue(),
            Intrinsic::Minus(ref ty, _) => ty.to_rvalue(),
            Intrinsic::LogicalNot(ref ty, _) => {
                match ty.0 {
                    TypeLayout::Scalar(_) => Type::bool().to_rvalue(),
                    TypeLayout::Vector(_, x) => Type::booln(x).to_rvalue(),
                    _ => return Err(TypeError::InvalidType(ty.clone())),
                }
            }
            Intrinsic::BitwiseNot(ref ty, _) => ty.to_rvalue(),
            Intrinsic::AllMemoryBarrier => Type::void().to_rvalue(),
            Intrinsic::AllMemoryBarrierWithGroupSync => Type::void().to_rvalue(),
            Intrinsic::DeviceMemoryBarrier => Type::void().to_rvalue(),
            Intrinsic::DeviceMemoryBarrierWithGroupSync => Type::void().to_rvalue(),
            Intrinsic::GroupMemoryBarrier => Type::void().to_rvalue(),
            Intrinsic::GroupMemoryBarrierWithGroupSync => Type::void().to_rvalue(),
            Intrinsic::AsIntU(_) => Type::int().to_rvalue(),
            Intrinsic::AsIntU2(_) => Type::intn(2).to_rvalue(),
            Intrinsic::AsIntU3(_) => Type::intn(3).to_rvalue(),
            Intrinsic::AsIntU4(_) => Type::intn(4).to_rvalue(),
            Intrinsic::AsIntF(_) => Type::int().to_rvalue(),
            Intrinsic::AsIntF2(_) => Type::intn(2).to_rvalue(),
            Intrinsic::AsIntF3(_) => Type::intn(3).to_rvalue(),
            Intrinsic::AsIntF4(_) => Type::intn(4).to_rvalue(),
            Intrinsic::AsUIntI(_) => Type::uint().to_rvalue(),
            Intrinsic::AsUIntI2(_) => Type::uintn(2).to_rvalue(),
            Intrinsic::AsUIntI3(_) => Type::uintn(3).to_rvalue(),
            Intrinsic::AsUIntI4(_) => Type::uintn(4).to_rvalue(),
            Intrinsic::AsUIntF(_) => Type::uint().to_rvalue(),
            Intrinsic::AsUIntF2(_) => Type::uintn(2).to_rvalue(),
            Intrinsic::AsUIntF3(_) => Type::uintn(3).to_rvalue(),
            Intrinsic::AsUIntF4(_) => Type::uintn(4).to_rvalue(),
            Intrinsic::AsFloatI(_) => Type::float().to_rvalue(),
            Intrinsic::AsFloatI2(_) => Type::floatn(2).to_rvalue(),
            Intrinsic::AsFloatI3(_) => Type::floatn(3).to_rvalue(),
            Intrinsic::AsFloatI4(_) => Type::floatn(4).to_rvalue(),
            Intrinsic::AsFloatU(_) => Type::float().to_rvalue(),
            Intrinsic::AsFloatU2(_) => Type::floatn(2).to_rvalue(),
            Intrinsic::AsFloatU3(_) => Type::floatn(3).to_rvalue(),
            Intrinsic::AsFloatU4(_) => Type::floatn(4).to_rvalue(),
            Intrinsic::AsFloatF(_) => Type::float().to_rvalue(),
            Intrinsic::AsFloatF2(_) => Type::floatn(2).to_rvalue(),
            Intrinsic::AsFloatF3(_) => Type::floatn(3).to_rvalue(),
            Intrinsic::AsFloatF4(_) => Type::floatn(4).to_rvalue(),
            Intrinsic::AsDouble(_, _) => Type::double().to_rvalue(),
            Intrinsic::ClampI(_, _, _) => Type::int().to_rvalue(),
            Intrinsic::ClampI2(_, _, _) => Type::intn(2).to_rvalue(),
            Intrinsic::ClampI3(_, _, _) => Type::intn(3).to_rvalue(),
            Intrinsic::ClampI4(_, _, _) => Type::intn(4).to_rvalue(),
            Intrinsic::ClampF(_, _, _) => Type::float().to_rvalue(),
            Intrinsic::ClampF2(_, _, _) => Type::floatn(2).to_rvalue(),
            Intrinsic::ClampF3(_, _, _) => Type::floatn(3).to_rvalue(),
            Intrinsic::ClampF4(_, _, _) => Type::floatn(4).to_rvalue(),
            Intrinsic::Cross(_, _) => Type::floatn(3).to_rvalue(),
            Intrinsic::Distance1(_, _) => Type::float().to_rvalue(),
            Intrinsic::Distance2(_, _) => Type::float().to_rvalue(),
            Intrinsic::Distance3(_, _) => Type::float().to_rvalue(),
            Intrinsic::Distance4(_, _) => Type::float().to_rvalue(),
            Intrinsic::DotI1(_, _) => Type::int().to_rvalue(),
            Intrinsic::DotI2(_, _) => Type::int().to_rvalue(),
            Intrinsic::DotI3(_, _) => Type::int().to_rvalue(),
            Intrinsic::DotI4(_, _) => Type::int().to_rvalue(),
            Intrinsic::DotF1(_, _) => Type::float().to_rvalue(),
            Intrinsic::DotF2(_, _) => Type::float().to_rvalue(),
            Intrinsic::DotF3(_, _) => Type::float().to_rvalue(),
            Intrinsic::DotF4(_, _) => Type::float().to_rvalue(),
            Intrinsic::Floor(_) => Type::float().to_rvalue(),
            Intrinsic::Floor2(_) => Type::floatn(2).to_rvalue(),
            Intrinsic::Floor3(_) => Type::floatn(3).to_rvalue(),
            Intrinsic::Floor4(_) => Type::floatn(4).to_rvalue(),
            Intrinsic::Min(_, _) => unimplemented!(),
            Intrinsic::Max(_, _) => unimplemented!(),
            Intrinsic::Normalize1(_) => Type::floatn(1).to_rvalue(),
            Intrinsic::Normalize2(_) => Type::floatn(2).to_rvalue(),
            Intrinsic::Normalize3(_) => Type::floatn(3).to_rvalue(),
            Intrinsic::Normalize4(_) => Type::floatn(4).to_rvalue(),
            Intrinsic::SignI(_) => Type::int().to_rvalue(),
            Intrinsic::SignI2(_) => Type::intn(1).to_rvalue(),
            Intrinsic::SignI3(_) => Type::intn(2).to_rvalue(),
            Intrinsic::SignI4(_) => Type::intn(3).to_rvalue(),
            Intrinsic::SignF(_) => Type::int().to_rvalue(),
            Intrinsic::SignF2(_) => Type::intn(1).to_rvalue(),
            Intrinsic::SignF3(_) => Type::intn(2).to_rvalue(),
            Intrinsic::SignF4(_) => Type::intn(3).to_rvalue(),
            Intrinsic::Sqrt(_) => Type::float().to_rvalue(),
            Intrinsic::Sqrt2(_) => Type::floatn(2).to_rvalue(),
            Intrinsic::Sqrt3(_) => Type::floatn(3).to_rvalue(),
            Intrinsic::Sqrt4(_) => Type::floatn(4).to_rvalue(),
            Intrinsic::BufferLoad(ref buffer, _) => {
                let buffer_ety = try!(TypeParser::get_expression_type(buffer, context));
                match (buffer_ety.0).0 {
                    TypeLayout::Object(ObjectType::Buffer(data_type)) => {
                        Type::from_data(data_type).to_rvalue()
                    }
                    tyl => return Err(TypeError::WrongObjectForBufferLoad(tyl)),
                }
            }
            Intrinsic::RWBufferLoad(ref buffer, _) => {
                let buffer_ety = try!(TypeParser::get_expression_type(buffer, context));
                match (buffer_ety.0).0 {
                    TypeLayout::Object(ObjectType::RWBuffer(data_type)) => {
                        Type::from_data(data_type).to_rvalue()
                    }
                    tyl => return Err(TypeError::WrongObjectForRWBufferLoad(tyl)),
                }
            }
            Intrinsic::StructuredBufferLoad(ref buffer, _) => {
                let buffer_ety = try!(TypeParser::get_expression_type(buffer, context));
                match (buffer_ety.0).0 {
                    TypeLayout::Object(ObjectType::StructuredBuffer(structured_type)) => {
                        Type::from_structured(structured_type).to_rvalue()
                    }
                    tyl => return Err(TypeError::WrongObjectForStructuredBufferLoad(tyl)),
                }
            }
            Intrinsic::RWStructuredBufferLoad(ref buffer, _) => {
                let buffer_ety = try!(TypeParser::get_expression_type(buffer, context));
                match (buffer_ety.0).0 {
                    TypeLayout::Object(ObjectType::RWStructuredBuffer(structured_type)) => {
                        Type::from_structured(structured_type).to_rvalue()
                    }
                    tyl => return Err(TypeError::WrongObjectForRWStructuredBufferLoad(tyl)),
                }
            }
            Intrinsic::RWTexture2DLoad(ref texture, _) => {
                let texture_ety = try!(TypeParser::get_expression_type(texture, context));
                match (texture_ety.0).0 {
                    TypeLayout::Object(ObjectType::RWTexture2D(data_type)) => {
                        Type::from_data(data_type).to_rvalue()
                    }
                    tyl => return Err(TypeError::WrongObjectForRWTexture2DLoad(tyl)),
                }
            }
            Intrinsic::ByteAddressBufferLoad(_, _) => Type::uint().to_rvalue(),
            Intrinsic::ByteAddressBufferLoad2(_, _) => Type::uintn(2).to_rvalue(),
            Intrinsic::ByteAddressBufferLoad3(_, _) => Type::uintn(3).to_rvalue(),
            Intrinsic::ByteAddressBufferLoad4(_, _) => Type::uintn(4).to_rvalue(),
            Intrinsic::RWByteAddressBufferLoad(_, _) => Type::uint().to_rvalue(),
            Intrinsic::RWByteAddressBufferLoad2(_, _) => Type::uintn(2).to_rvalue(),
            Intrinsic::RWByteAddressBufferLoad3(_, _) => Type::uintn(3).to_rvalue(),
            Intrinsic::RWByteAddressBufferLoad4(_, _) => Type::uintn(4).to_rvalue(),
            Intrinsic::RWByteAddressBufferStore(_, _, _) => Type::void().to_rvalue(),
            Intrinsic::RWByteAddressBufferStore2(_, _, _) => Type::void().to_rvalue(),
            Intrinsic::RWByteAddressBufferStore3(_, _, _) => Type::void().to_rvalue(),
            Intrinsic::RWByteAddressBufferStore4(_, _, _) => Type::void().to_rvalue(),
        })
    }
}
