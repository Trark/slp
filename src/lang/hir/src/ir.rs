use crate::hir_intrinsics::*;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;

/// Basic scalar types
#[derive(PartialEq, Clone)]
pub enum ScalarType {
    Bool,
    UntypedInt,
    Int,
    UInt,
    Half,
    Float,
    Double,
}

impl fmt::Debug for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match *self {
            ScalarType::Bool => "bool",
            ScalarType::UntypedInt => "UntypedInt",
            ScalarType::Int => "int",
            ScalarType::UInt => "uint",
            ScalarType::Half => "half",
            ScalarType::Float => "float",
            ScalarType::Double => "double",
        };
        write!(f, "{}", name)
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum NumericDimension {
    Scalar,
    Vector(u32),
    Matrix(u32, u32),
}

/// Layout for DataType
#[derive(PartialEq, Clone)]
pub enum DataLayout {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
}

impl fmt::Debug for DataLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DataLayout::Scalar(ref st) => write!(f, "{:?}", st),
            DataLayout::Vector(ref st, ref x) => write!(f, "{:?}{}", st, x),
            DataLayout::Matrix(ref st, ref x, ref y) => write!(f, "{:?}{}x{}", st, x, y),
        }
    }
}

impl DataLayout {
    pub fn new(scalar: ScalarType, dim: NumericDimension) -> DataLayout {
        match dim {
            NumericDimension::Scalar => DataLayout::Scalar(scalar),
            NumericDimension::Vector(x) => DataLayout::Vector(scalar, x),
            NumericDimension::Matrix(x, y) => DataLayout::Matrix(scalar, x, y),
        }
    }
    pub fn to_scalar(&self) -> ScalarType {
        match *self {
            DataLayout::Scalar(ref scalar)
            | DataLayout::Vector(ref scalar, _)
            | DataLayout::Matrix(ref scalar, _, _) => scalar.clone(),
        }
    }
    pub fn transform_scalar(self, to_scalar: ScalarType) -> DataLayout {
        match self {
            DataLayout::Scalar(_) => DataLayout::Scalar(to_scalar),
            DataLayout::Vector(_, x) => DataLayout::Vector(to_scalar, x),
            DataLayout::Matrix(_, x, y) => DataLayout::Matrix(to_scalar, x, y),
        }
    }
}

impl From<TypeLayout> for Option<DataLayout> {
    fn from(ty: TypeLayout) -> Option<DataLayout> {
        match ty {
            TypeLayout::Scalar(scalar) => Some(DataLayout::Scalar(scalar)),
            TypeLayout::Vector(scalar, x) => Some(DataLayout::Vector(scalar, x)),
            TypeLayout::Matrix(scalar, x, y) => Some(DataLayout::Matrix(scalar, x, y)),
            _ => None,
        }
    }
}

/// A type that can be used in data buffers (Buffer / RWBuffer / etc)
/// These can interpret data in buffers bound with a format
/// FormatType might be a better name because they can bind resource
/// views with a format, but HLSL just called them Buffer and other
/// apis call them data buffers
#[derive(PartialEq, Clone)]
pub struct DataType(pub DataLayout, pub TypeModifier);

impl fmt::Debug for DataType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}{:?}", self.1, self.0)
    }
}

impl From<Type> for Option<DataType> {
    fn from(ty: Type) -> Option<DataType> {
        let Type(tyl, ty_mod) = ty;
        match tyl.into() {
            Some(dtyl) => Some(DataType(dtyl, ty_mod)),
            None => None,
        }
    }
}

/// Layout for StructuredType
#[derive(PartialEq, Clone)]
pub enum StructuredLayout {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(ScalarType, u32, u32),
    Struct(StructId),
}

impl fmt::Debug for StructuredLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            StructuredLayout::Scalar(ref st) => write!(f, "{:?}", st),
            StructuredLayout::Vector(ref st, ref x) => write!(f, "{:?}{}", st, x),
            StructuredLayout::Matrix(ref st, ref x, ref y) => write!(f, "{:?}{}x{}", st, x, y),
            StructuredLayout::Struct(ref sid) => write!(f, "struct<{}>", sid.0),
        }
    }
}

/// A type that can be used in structured buffers
/// These are the both all the data types and user defined structs
#[derive(PartialEq, Clone)]
pub struct StructuredType(pub StructuredLayout, pub TypeModifier);

impl fmt::Debug for StructuredType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}{:?}", self.1, self.0)
    }
}

#[derive(PartialEq, Clone)]
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

impl fmt::Debug for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ObjectType::*;
        match *self {
            Buffer(ref dt) => write!(f, "Buffer<{:?}>", dt),
            RWBuffer(ref dt) => write!(f, "RWBuffer<{:?}>", dt),
            ByteAddressBuffer => write!(f, "ByteAddressBuffer"),
            RWByteAddressBuffer => write!(f, "RWByteAddressBuffer"),
            StructuredBuffer(ref st) => write!(f, "StructuredBuffer<{:?}>", st),
            RWStructuredBuffer(ref st) => write!(f, "RWStructuredBuffer<{:?}>", st),
            AppendStructuredBuffer(ref st) => write!(f, "AppendStructuredBuffer<{:?}>", st),
            ConsumeStructuredBuffer(ref st) => write!(f, "ConsumeStructuredBuffer<{:?}>", st),
            Texture1D(ref dt) => write!(f, "Texture1D<{:?}>", dt),
            Texture1DArray(ref dt) => write!(f, "Texture1DArray<{:?}>", dt),
            Texture2D(ref dt) => write!(f, "Texture2D<{:?}>", dt),
            Texture2DArray(ref dt) => write!(f, "Texture2DArray<{:?}>", dt),
            Texture2DMS(ref dt) => write!(f, "Texture2DMS<{:?}>", dt),
            Texture2DMSArray(ref dt) => write!(f, "Texture2DMSArray<{:?}>", dt),
            Texture3D(ref dt) => write!(f, "Texture3D<{:?}>", dt),
            TextureCube(ref dt) => write!(f, "TextureCube<{:?}>", dt),
            TextureCubeArray(ref dt) => write!(f, "TextureCubeArray<{:?}>", dt),
            RWTexture1D(ref dt) => write!(f, "RWTexture1D<{:?}>", dt),
            RWTexture1DArray(ref dt) => write!(f, "RWTexture1DArray<{:?}>", dt),
            RWTexture2D(ref dt) => write!(f, "RWTexture2D<{:?}>", dt),
            RWTexture2DArray(ref dt) => write!(f, "RWTexture2DArray<{:?}>", dt),
            RWTexture3D(ref dt) => write!(f, "RWTexture3D<{:?}>", dt),
            InputPatch => write!(f, "InputPatch"),
            OutputPatch => write!(f, "OutputPatch"),
        }
    }
}

#[derive(PartialEq, Clone)]
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
            TypeLayout::Scalar(ref scalar)
            | TypeLayout::Vector(ref scalar, _)
            | TypeLayout::Matrix(ref scalar, _, _) => Some(scalar.clone()),
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

    /// Replaces the scalar type inside a numeric type with the given scalar type
    pub fn transform_scalar(self, to_scalar: ScalarType) -> TypeLayout {
        match self {
            TypeLayout::Scalar(_) => TypeLayout::Scalar(to_scalar),
            TypeLayout::Vector(_, x) => TypeLayout::Vector(to_scalar, x),
            TypeLayout::Matrix(_, x, y) => TypeLayout::Matrix(to_scalar, x, y),
            _ => panic!("non-numeric type in TypeLayout::transform_scalar"),
        }
    }

    pub fn is_array(&self) -> bool {
        if let &TypeLayout::Array(_, _) = self {
            true
        } else {
            false
        }
    }
}

impl fmt::Debug for TypeLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TypeLayout::Void => write!(f, "void"),
            TypeLayout::Scalar(ref st) => write!(f, "{:?}", st),
            TypeLayout::Vector(ref st, ref x) => write!(f, "{:?}{}", st, x),
            TypeLayout::Matrix(ref st, ref x, ref y) => write!(f, "{:?}{}x{}", st, x, y),
            TypeLayout::Struct(ref sid) => write!(f, "struct<{}>", sid.0),
            TypeLayout::SamplerState => write!(f, "SamplerState"),
            TypeLayout::Object(ref ot) => write!(f, "{:?}", ot),
            TypeLayout::Array(ref ty, ref len) => write!(f, "{:?}[{}]", ty, len),
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
#[derive(PartialEq, Clone)]
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
        self.is_const == false
            && self.row_order == RowOrder::Column
            && self.precise == false
            && self.volatile == false
    }

    pub fn const_only() -> TypeModifier {
        TypeModifier {
            is_const: true,
            ..TypeModifier::default()
        }
    }

    pub fn keep_precise(&self) -> TypeModifier {
        TypeModifier {
            precise: self.precise,
            ..TypeModifier::default()
        }
    }
}

impl fmt::Debug for TypeModifier {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        let p1 = if self.is_const { "const " } else { "" };
        let p2 = if self.row_order == RowOrder::Row {
            "row_major "
        } else {
            ""
        };
        let p3 = if self.precise { "precise " } else { "" };
        let p4 = if self.volatile { "volatile " } else { "" };
        write!(f, "{}{}{}{}", p1, p2, p3, p4)
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
///
/// extern not supported because constant buffers exist
///
/// uniform not supported because constant buffers exist
#[derive(PartialEq, Debug, Clone)]
pub enum GlobalStorage {
    // Input from outside the kernel (default)
    Extern,

    /// Statically allocated thread-local variable in global scope
    Static,

    /// Shared between every thread in the work group
    GroupShared,
}

impl Default for GlobalStorage {
    fn default() -> GlobalStorage {
        GlobalStorage::Static
    }
}

/// Binding type for parameters
///
/// uniform not supported because constant buffers exist
#[derive(PartialEq, Debug, Clone)]
pub enum InputModifier {
    /// Function input
    In,
    /// Function output (must be written)
    Out,
    /// Function input and output
    InOut,
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
#[derive(PartialEq, Clone)]
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

    pub fn transform_scalar(self, to_scalar: ScalarType) -> Type {
        let Type(tyl, ty_mod) = self;
        Type(tyl.transform_scalar(to_scalar), ty_mod)
    }

    pub fn is_array(&self) -> bool {
        self.0.is_array()
    }

    pub fn is_void(&self) -> bool {
        self.0 == TypeLayout::Void
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}{:?}", self.1, self.0)
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
pub struct GlobalType(
    pub Type,
    pub GlobalStorage,
    pub Option<InterpolationModifier>,
);

impl From<Type> for GlobalType {
    fn from(ty: Type) -> GlobalType {
        GlobalType(ty, GlobalStorage::default(), None)
    }
}

/// The type of any parameter declaration
#[derive(PartialEq, Debug, Clone)]
pub struct ParamType(
    pub Type,
    pub InputModifier,
    pub Option<InterpolationModifier>,
);

impl From<Type> for ParamType {
    fn from(ty: Type) -> ParamType {
        ParamType(ty, InputModifier::default(), None)
    }
}

/// The type of any local variable declaration
#[derive(PartialEq, Debug, Clone)]
pub struct LocalType(
    pub Type,
    pub LocalStorage,
    pub Option<InterpolationModifier>,
);

impl From<Type> for LocalType {
    fn from(ty: Type) -> LocalType {
        LocalType(ty, LocalStorage::default(), None)
    }
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
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct VariableRef(pub VariableId, pub ScopeRef);

impl VariableRef {
    pub fn raw(id: u32, scope: u32) -> VariableRef {
        VariableRef(VariableId(id), ScopeRef(scope))
    }
}

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
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    Swizzle(Box<Expression>, Vec<SwizzleSlot>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(FunctionId, Vec<Expression>),
    /// Constructors for builtin numeric types, such as `float2(1.0, 0.0)`
    NumericConstructor(DataLayout, Vec<ConstructorSlot>),
    Cast(Type, Box<Expression>),
    Intrinsic0(Intrinsic0),
    Intrinsic1(Intrinsic1, Box<Expression>),
    Intrinsic2(Intrinsic2, Box<Expression>, Box<Expression>),
    Intrinsic3(
        Intrinsic3,
        Box<Expression>,
        Box<Expression>,
        Box<Expression>,
    ),
}

#[derive(PartialEq, Debug, Clone)]
/// The node for representing the initial value of a variable
pub enum Initializer {
    /// Variable is initialized to the value of an expression
    Expression(Expression),
    /// Variable is initialized in parts (composite types and arrays)
    /// Unlike HLSL or slp_lang_hst, this can not be used for scalars with
    /// a 1 element aggregate.
    Aggregate(Vec<Initializer>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub id: VariableId,
    pub local_type: LocalType,
    pub init: Option<Initializer>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum ForInit {
    Empty,
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
    Break,
    Continue,
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

pub use slp_lang_hst::PackOffset;
pub use slp_lang_hst::PackSubOffset;

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
    pub init: Option<Initializer>,
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
    pub samplers: HashMap<u32, GlobalEntry>,
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
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TypeError::InvalidLocal => write!(f, "invalid local variable"),
            TypeError::LocalScopeInvalid => write!(f, "scope invalid"),
            TypeError::LocalDoesNotExist(_) => write!(f, "local variable does not exist"),
            TypeError::GlobalDoesNotExist(_) => write!(f, "global variable does not exist"),
            TypeError::ConstantBufferDoesNotExist(_) => write!(f, "constant buffer does not exist"),
            TypeError::ConstantDoesNotExist(_, _) => write!(f, "constant variable does not exist"),
            TypeError::StructDoesNotExist(_) => write!(f, "struct does not exist"),
            TypeError::StructMemberDoesNotExist(_, _) => write!(f, "struct member does not exist"),
            TypeError::FunctionDoesNotExist(_) => write!(f, "function does not exist"),
            TypeError::InvalidTypeInEqualityOperation(_) => {
                write!(f, "invalid numeric type in equality operation")
            }
            TypeError::InvalidTypeForSwizzle(_) => {
                write!(f, "swizzle nodes must be used on vectors")
            }
            TypeError::MemberNodeMustBeUsedOnStruct(_, _) => {
                write!(f, "member used on non-struct type")
            }
            TypeError::ArrayIndexMustBeUsedOnArrayType(_) => {
                write!(f, "array index used on non-array type")
            }

            TypeError::InvalidType(_) => write!(f, "invalid type in an intrinsic function"),
        }
    }
}

/// An object to hold all context of the type of definitions at a point in
/// the program
pub trait TypeContext: AsTypeContext {
    fn get_local(&self, var_ref: &VariableRef) -> Result<ExpressionType, TypeError>;
    fn get_global(&self, id: &GlobalId) -> Result<ExpressionType, TypeError>;
    fn get_constant(&self, id: &ConstantBufferId, name: &str) -> Result<ExpressionType, TypeError>;
    fn get_struct_member(&self, id: &StructId, name: &str) -> Result<ExpressionType, TypeError>;
    fn get_function_return(&self, id: &FunctionId) -> Result<ExpressionType, TypeError>;
}

pub trait AsTypeContext {
    fn as_type_context(&self) -> &dyn TypeContext;
}

impl<T: TypeContext> AsTypeContext for T {
    fn as_type_context(&self) -> &dyn TypeContext {
        self
    }
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
                RootDefinition::GlobalVariable(ref gv) => {
                    match context
                        .globals
                        .insert(gv.id.clone(), gv.global_type.0.clone())
                    {
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
                    match context
                        .functions
                        .insert(func.id.clone(), func.returntype.clone())
                    {
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
            Some(ref cm) => match cm.get(name) {
                Some(ref ty) => Ok(ty.to_lvalue()),
                None => Err(TypeError::ConstantDoesNotExist(
                    id.clone(),
                    name.to_string(),
                )),
            },
            None => Err(TypeError::ConstantBufferDoesNotExist(id.clone())),
        }
    }

    fn get_struct_member(&self, id: &StructId, name: &str) -> Result<ExpressionType, TypeError> {
        match self.structs.get(&id) {
            Some(ref cm) => match cm.get(name) {
                Some(ref ty) => Ok(ty.to_lvalue()),
                None => Err(TypeError::StructMemberDoesNotExist(
                    id.clone(),
                    name.to_string(),
                )),
            },
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
    pub fn get_literal_type(literal: &Literal) -> ExpressionType {
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

    pub fn get_expression_type(
        expression: &Expression,
        context: &dyn TypeContext,
    ) -> Result<ExpressionType, TypeError> {
        match *expression {
            Expression::Literal(ref lit) => Ok(TypeParser::get_literal_type(lit)),
            Expression::Variable(ref var_ref) => context.get_local(var_ref),
            Expression::Global(ref id) => context.get_global(id),
            Expression::ConstantVariable(ref id, ref name) => context.get_constant(id, name),
            Expression::TernaryConditional(_, ref expr_left, ref expr_right) => {
                // Ensure the layouts of each side are the same
                // Value types + modifiers can be different
                assert_eq!(
                    (TypeParser::get_expression_type(expr_left, context)?.0).0,
                    (TypeParser::get_expression_type(expr_right, context)?.0).0
                );
                let ety = TypeParser::get_expression_type(expr_left, context)?;
                Ok(ety.0.to_rvalue())
            }
            Expression::Swizzle(ref vec, ref swizzle) => {
                let ExpressionType(Type(vec_tyl, vec_mod), vec_vt) =
                    TypeParser::get_expression_type(vec, context)?;
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
                let array_ty = TypeParser::get_expression_type(&array, context)?;
                // Todo: Modifiers on object type template parameters
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
                let expr_type = TypeParser::get_expression_type(&expr, context)?;
                let id = match (expr_type.0).0 {
                    TypeLayout::Struct(id) => id,
                    tyl => {
                        return Err(TypeError::MemberNodeMustBeUsedOnStruct(
                            tyl.clone(),
                            name.clone(),
                        ))
                    }
                };
                context.get_struct_member(&id, name)
            }
            Expression::Call(ref id, _) => context.get_function_return(id),
            Expression::NumericConstructor(ref dtyl, _) => {
                Ok(Type::from_layout(TypeLayout::from_data(dtyl.clone())).to_rvalue())
            }
            Expression::Cast(ref ty, _) => Ok(ty.to_rvalue()),
            Expression::Intrinsic0(ref intrinsic) => Ok(intrinsic.get_return_type()),
            Expression::Intrinsic1(ref intrinsic, _) => Ok(intrinsic.get_return_type()),
            Expression::Intrinsic2(ref intrinsic, _, _) => Ok(intrinsic.get_return_type()),
            Expression::Intrinsic3(ref intrinsic, _, _, _) => Ok(intrinsic.get_return_type()),
        }
    }
}
