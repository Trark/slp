
use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use slp_lang_hir as ir;
use slp_lang_hir::ExpressionType;
use slp_lang_hir::ToExpressionType;
use slp_lang_hst as ast;
use super::intrinsics;
use super::intrinsics::IntrinsicFactory;
use super::casting::ImplicitConversion;
use super::casting::ConversionPriority;

#[derive(PartialEq, Debug, Clone)]
pub enum TyperError {
    Unimplemented,

    ValueAlreadyDefined(String, ErrorType, ErrorType),
    StructAlreadyDefined(String),
    ConstantBufferAlreadyDefined(String),

    ConstantSlotAlreadyUsed(ir::ConstantBufferId, ir::ConstantBufferId),
    ReadResourceSlotAlreadyUsed(ir::GlobalId, ir::GlobalId),
    ReadWriteResourceSlotAlreadyUsed(ir::GlobalId, ir::GlobalId),

    UnknownIdentifier(String),
    UnknownType(ErrorType),

    TypeDoesNotHaveMembers(ErrorType),
    UnknownTypeMember(ErrorType, String),
    InvalidSwizzle(ErrorType, String),

    ArrayIndexingNonArrayType,
    ArraySubscriptIndexNotInteger,

    CallOnNonFunction,

    FunctionPassedToAnotherFunction(ErrorType, ErrorType),
    FunctionArgumentTypeMismatch(Vec<FunctionOverload>, Vec<ExpressionType>),
    NumericConstructorWrongArgumentCount,

    UnaryOperationWrongTypes(ast::UnaryOp, ErrorType),
    BinaryOperationWrongTypes(ir::BinOp, ErrorType, ErrorType),
    BinaryOperationNonNumericType,
    TernaryConditionRequiresBoolean(ErrorType),
    TernaryArmsMustHaveSameType(ErrorType, ErrorType),

    ExpectedValueExpression(ErrorType),

    InvalidCast(ErrorType, ErrorType),
    FunctionTypeInInitExpression,
    WrongTypeInInitExpression,
    WrongTypeInConstructor,
    WrongTypeInReturnStatement,
    FunctionNotCalled,

    KernelNotDefined,
    KernelDefinedMultipleTimes,
    KernelHasNoDispatchDimensions,
    KernelHasParamWithBadSemantic(ast::FunctionParam),
    KernelHasParamWithoutSemantic(ast::FunctionParam),

    LvalueRequired,
    ArrayDimensionsMustBeConstantExpression(ast::Expression),
}

pub type ReturnType = ir::Type;
pub type ClassType = ir::Type;

#[derive(PartialEq, Debug, Clone)]
pub enum FunctionName {
    Intrinsic(intrinsics::IntrinsicFactory),
    User(ir::FunctionId),
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionOverload(pub FunctionName, pub ReturnType, pub Vec<ir::ParamType>);

#[derive(PartialEq, Debug, Clone)]
pub struct ResolvedFunction(pub FunctionOverload);

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedFunction(pub String, pub Vec<FunctionOverload>);

#[derive(PartialEq, Debug, Clone)]
pub struct ResolvedMethod(pub ClassType, pub FunctionOverload, ir::Expression);

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedMethod(pub String, pub ClassType, pub Vec<FunctionOverload>, ir::Expression);

#[derive(PartialEq, Debug, Clone)]
pub enum ErrorType {
    Value(ast::Type),
    Function(UnresolvedFunction),
    Method(UnresolvedMethod),
    Unknown,
}

impl error::Error for TyperError {
    fn description(&self) -> &str {
        match *self {
            TyperError::Unimplemented => "unimplemented",

            TyperError::ValueAlreadyDefined(_, _, _) => "identifier already defined",
            TyperError::StructAlreadyDefined(_) => "struct aready defined",
            TyperError::ConstantBufferAlreadyDefined(_) => "cbuffer aready defined",

            TyperError::ConstantSlotAlreadyUsed(_, _) => "global constant slot already used",
            TyperError::ReadResourceSlotAlreadyUsed(_, _) => "global resource slot already used",
            TyperError::ReadWriteResourceSlotAlreadyUsed(_, _) => {
                "global writable resource slot already used"
            }

            TyperError::UnknownIdentifier(_) => "unknown identifier",
            TyperError::UnknownType(_) => "unknown type name",

            TyperError::TypeDoesNotHaveMembers(_) => "unknown member (type has no members)",
            TyperError::UnknownTypeMember(_, _) => "unknown member",
            TyperError::InvalidSwizzle(_, _) => "invalid swizzle",

            TyperError::ArrayIndexingNonArrayType => "array index applied to non-array type",
            TyperError::ArraySubscriptIndexNotInteger => "array subscripts must be integers",

            TyperError::CallOnNonFunction => "function call applied to non-function type",

            TyperError::FunctionPassedToAnotherFunction(_, _) => {
                "functions can not be passed to other functions"
            }
            TyperError::FunctionArgumentTypeMismatch(_, _) => "wrong parameters given to function",
            TyperError::NumericConstructorWrongArgumentCount => {
                "wrong number of arguments to constructor"
            }

            TyperError::UnaryOperationWrongTypes(_, _) => {
                "operation does not support the given types"
            }
            TyperError::BinaryOperationWrongTypes(_, _, _) => {
                "operation does not support the given types"
            }
            TyperError::BinaryOperationNonNumericType => "non-numeric type in binary operation",
            TyperError::TernaryConditionRequiresBoolean(_) => "ternary condition must be boolean",
            TyperError::TernaryArmsMustHaveSameType(_, _) => "ternary arms must have the same type",

            TyperError::ExpectedValueExpression(_) => "expected a value expression",

            TyperError::InvalidCast(_, _) => "invalid cast",
            TyperError::FunctionTypeInInitExpression => "function not called",
            TyperError::WrongTypeInInitExpression => "wrong type in variable initialization",
            TyperError::WrongTypeInConstructor => "wrong type in numeric constructor",
            TyperError::WrongTypeInReturnStatement => "wrong type in return statement",
            TyperError::FunctionNotCalled => "function not called",

            TyperError::KernelNotDefined => "entry point not found",
            TyperError::KernelDefinedMultipleTimes => "multiple entry points found",
            TyperError::KernelHasNoDispatchDimensions => {
                "compute kernels require a dispatch dimension"
            }
            TyperError::KernelHasParamWithBadSemantic(_) => {
                "kernel parameter did not have a valid kernel semantic"
            }
            TyperError::KernelHasParamWithoutSemantic(_) => {
                "kernel parameter did not have a kernel semantic"
            }

            TyperError::LvalueRequired => "lvalue is required in this context",
            TyperError::ArrayDimensionsMustBeConstantExpression(_) => {
                "array dimensions must be constant"
            }
        }
    }
}

impl fmt::Display for TyperError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

#[derive(PartialEq, Debug, Clone)]
enum TypedExpression {
    // Expression + Type
    Value(ir::Expression, ExpressionType),
    // Name of function + overloads
    Function(UnresolvedFunction),
    // Name of function + overloads + object
    Method(UnresolvedMethod),
}

#[derive(PartialEq, Debug, Clone)]
struct VariableBlock {
    variables: HashMap<String, (ir::Type, ir::VariableId)>,
    next_free_variable_id: ir::VariableId,
}

#[derive(PartialEq, Debug, Clone)]
struct TypeBlock {
    struct_ids: HashMap<String, ir::StructId>,
    struct_names: HashMap<ir::StructId, String>,
    struct_definitions: HashMap<ir::StructId, HashMap<String, ir::Type>>,
    next_free_struct_id: ir::StructId,

    cbuffer_ids: HashMap<String, ir::ConstantBufferId>,
    cbuffer_names: HashMap<ir::ConstantBufferId, String>,
    cbuffer_definitions: HashMap<ir::ConstantBufferId, HashMap<String, ir::Type>>,
    next_free_cbuffer_id: ir::ConstantBufferId,
}

#[derive(PartialEq, Debug, Clone)]
struct GlobalContext {
    functions: HashMap<String, UnresolvedFunction>,
    function_names: HashMap<ir::FunctionId, String>,
    next_free_function_id: ir::FunctionId,

    types: TypeBlock,
    globals: HashMap<String, (ir::Type, ir::GlobalId)>,
    global_names: HashMap<ir::GlobalId, String>,
    next_free_global_id: ir::GlobalId,

    current_return_type: Option<ir::Type>,

    global_slots_r: Vec<(u32, ir::GlobalEntry)>,
    global_slots_rw: Vec<(u32, ir::GlobalEntry)>,
    global_slots_constants: Vec<(u32, ir::ConstantBufferId)>,
}

#[derive(PartialEq, Debug, Clone)]
struct ScopeContext {
    parent: Box<Context>,

    variables: VariableBlock,
}

#[derive(PartialEq, Debug, Clone)]
enum Context {
    Global(GlobalContext),
    Scope(ScopeContext),
}

trait ExpressionContext : StructIdFinder + ErrorTypeContext + ir::TypeContext {
    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError>;
    fn find_struct_member(&self,
                          id: &ir::StructId,
                          member_name: &String)
                          -> Result<ir::Type, TyperError>;
    fn get_return_type(&self) -> ir::Type;

    fn as_struct_id_finder(&self) -> &StructIdFinder;
    fn as_type_context(&self) -> &ir::TypeContext;
}

trait StructIdFinder {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError>;
}

trait ErrorTypeContext {
    fn ir_type_to_error_type(&self, ty: &ir::Type) -> ErrorType;

    fn exp_type_to_error_type(&self, ty: &ExpressionType) -> ErrorType {
        self.ir_type_to_error_type(&ty.0)
    }

    fn typed_expression_to_error_type(&self, texp: &TypedExpression) -> ErrorType;
}

impl VariableBlock {
    fn new() -> VariableBlock {
        VariableBlock {
            variables: HashMap::new(),
            next_free_variable_id: ir::VariableId(0),
        }
    }

    fn insert_variable(&mut self,
                       name: String,
                       typename: ir::Type,
                       context: &ErrorTypeContext)
                       -> Result<ir::VariableId, TyperError> {
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name,
                                                       context.ir_type_to_error_type(ty),
                                                       context.ir_type_to_error_type(&typename)));
        };
        match self.variables.entry(name.clone()) {
            Entry::Occupied(occupied) => {
                Err(TyperError::ValueAlreadyDefined(name,
                                                    context.ir_type_to_error_type(&occupied.get()
                                                                                           .0),
                                                    context.ir_type_to_error_type(&typename)))
            }
            Entry::Vacant(vacant) => {
                let id = self.next_free_variable_id;
                self.next_free_variable_id = ir::VariableId(self.next_free_variable_id.0 + 1);
                vacant.insert((typename, id));
                Ok(id)
            }
        }
    }

    fn has_variable(&self, name: &String) -> Option<&(ir::Type, ir::VariableId)> {
        self.variables.get(name)
    }

    fn find_variable(&self, name: &String, scopes_up: u32) -> Option<TypedExpression> {
        match self.variables.get(name) {
            Some(&(ref ty, ref id)) => return Some(TypedExpression::Value(ir::Expression::Variable(ir::VariableRef(id.clone(), ir::ScopeRef(scopes_up))), ty.to_lvalue())),
            None => None,
        }
    }

    fn destruct(self) -> HashMap<ir::VariableId, (String, ir::Type)> {
        self.variables.iter().fold(HashMap::new(), |mut map, (name, &(ref ty, ref id))| {
            map.insert(id.clone(), (name.clone(), ty.clone()));
            map
        })
    }
}

impl TypeBlock {
    fn new() -> TypeBlock {
        TypeBlock {
            struct_ids: HashMap::new(),
            struct_names: HashMap::new(),
            struct_definitions: HashMap::new(),
            next_free_struct_id: ir::StructId(0),
            cbuffer_ids: HashMap::new(),
            cbuffer_names: HashMap::new(),
            cbuffer_definitions: HashMap::new(),
            next_free_cbuffer_id: ir::ConstantBufferId(0),
        }
    }

    fn insert_struct(&mut self,
                     name: &String,
                     members: HashMap<String, ir::Type>)
                     -> Option<ir::StructId> {
        let id = self.next_free_struct_id;
        self.next_free_struct_id = ir::StructId(self.next_free_struct_id.0 + 1);
        match (self.struct_ids.entry(name.clone()),
               self.struct_names.entry(id.clone()),
               self.struct_definitions.entry(id.clone())) {
            (Entry::Vacant(id_entry),
             Entry::Vacant(name_entry),
             Entry::Vacant(def_entry)) => {
                id_entry.insert(id.clone());
                name_entry.insert(name.clone());
                def_entry.insert(members);
                Some(id.clone())
            }
            _ => None,
        }
    }

    fn find_struct_member(&self,
                          id: &ir::StructId,
                          member_name: &String)
                          -> Result<ir::Type, TyperError> {
        match self.struct_definitions.get(id) {
            Some(def) => def.get(member_name).map(|ty| ty.clone()).ok_or(
                TyperError::UnknownTypeMember(
                    self.ir_type_to_error_type(&ir::Type::from_struct(id.clone())),
                    member_name.clone()
                )
            ),
            None => Err(TyperError::UnknownType(
                self.ir_type_to_error_type(&ir::Type::from_struct(id.clone()))
            )),
        }
    }

    fn insert_cbuffer(&mut self,
                      name: &String,
                      members: HashMap<String, ir::Type>)
                      -> Option<ir::ConstantBufferId> {
        let id = self.next_free_cbuffer_id;
        self.next_free_cbuffer_id = ir::ConstantBufferId(self.next_free_cbuffer_id.0 + 1);
        match (self.cbuffer_ids.entry(name.clone()),
               self.cbuffer_names.entry(id.clone()),
               self.cbuffer_definitions.entry(id.clone())) {
            (Entry::Vacant(id_entry),
             Entry::Vacant(name_entry),
             Entry::Vacant(def_entry)) => {
                id_entry.insert(id.clone());
                name_entry.insert(name.clone());
                def_entry.insert(members);
                Some(id.clone())
            }
            _ => None,
        }
    }

    fn find_variable(&self, name: &String) -> Option<TypedExpression> {
        for (id, members) in &self.cbuffer_definitions {
            for (member_name, ty) in members {
                if member_name == name {
                    return Some(TypedExpression::Value(
                        ir::Expression::ConstantVariable(id.clone(), name.clone()),
                        ty.to_lvalue()
                    ));
                }
            }
        }
        None
    }

    fn get_struct_name(&self, id: &ir::StructId) -> Option<String> {
        for (struct_name, struct_id) in &self.struct_ids {
            if id == struct_id {
                return Some(struct_name.clone());
            }
        }
        None
    }

    fn invert_scalartype(&self, ty: &ir::ScalarType) -> ast::ScalarType {
        match *ty {
            ir::ScalarType::Bool => ast::ScalarType::Bool,
            ir::ScalarType::UntypedInt => ast::ScalarType::UntypedInt,
            ir::ScalarType::Int => ast::ScalarType::Int,
            ir::ScalarType::UInt => ast::ScalarType::UInt,
            ir::ScalarType::Half => ast::ScalarType::Half,
            ir::ScalarType::Float => ast::ScalarType::Float,
            ir::ScalarType::Double => ast::ScalarType::Double,
        }
    }

    fn invert_row_order(&self, row_order: &ir::RowOrder) -> ast::RowOrder {
        match *row_order {
            ir::RowOrder::Row => ast::RowOrder::Row,
            ir::RowOrder::Column => ast::RowOrder::Column,
        }
    }

    fn invert_modifier(&self, modifier: &ir::TypeModifier) -> ast::TypeModifier {
        ast::TypeModifier {
            is_const: modifier.is_const,
            row_order: self.invert_row_order(&modifier.row_order),
            precise: modifier.precise,
            volatile: modifier.volatile,
        }
    }

    fn invert_datalayout(&self, ty: &ir::DataLayout) -> ast::DataLayout {
        match *ty {
            ir::DataLayout::Scalar(ref scalar) => {
                ast::DataLayout::Scalar(self.invert_scalartype(scalar))
            }
            ir::DataLayout::Vector(ref scalar, ref x) => {
                ast::DataLayout::Vector(self.invert_scalartype(scalar), *x)
            }
            ir::DataLayout::Matrix(ref scalar, ref x, ref y) => {
                ast::DataLayout::Matrix(self.invert_scalartype(scalar), *x, *y)
            }
        }
    }

    fn invert_datatype(&self, ty: &ir::DataType) -> ast::DataType {
        let &ir::DataType(ref tyl, ref modifier) = ty;
        ast::DataType(self.invert_datalayout(tyl), self.invert_modifier(modifier))
    }

    fn invert_structuredlayout(&self, ty: &ir::StructuredLayout) -> ast::StructuredLayout {
        match *ty {
            ir::StructuredLayout::Scalar(ref scalar) => {
                ast::StructuredLayout::Scalar(self.invert_scalartype(scalar))
            }
            ir::StructuredLayout::Vector(ref scalar, ref x) => {
                ast::StructuredLayout::Vector(self.invert_scalartype(scalar), *x)
            }
            ir::StructuredLayout::Matrix(ref scalar, ref x, ref y) => {
                ast::StructuredLayout::Matrix(self.invert_scalartype(scalar), *x, *y)
            }
            ir::StructuredLayout::Struct(ref id) => {
                ast::StructuredLayout::Custom(match self.get_struct_name(&id) {
                    Some(name) => name,
                    None => "<struct>".to_string(),
                })
            }
        }
    }

    fn invert_structuredtype(&self, ty: &ir::StructuredType) -> ast::StructuredType {
        let &ir::StructuredType(ref tyl, ref modifier) = ty;
        ast::StructuredType(self.invert_structuredlayout(tyl),
                            self.invert_modifier(modifier))
    }

    fn invert_objecttype(&self, ty: &ir::ObjectType) -> ast::ObjectType {
        match *ty {
            ir::ObjectType::Buffer(ref data_type) => {
                ast::ObjectType::Buffer(self.invert_datatype(data_type))
            }
            ir::ObjectType::RWBuffer(ref data_type) => {
                ast::ObjectType::RWBuffer(self.invert_datatype(data_type))
            }
            ir::ObjectType::ByteAddressBuffer => ast::ObjectType::ByteAddressBuffer,
            ir::ObjectType::RWByteAddressBuffer => ast::ObjectType::RWByteAddressBuffer,
            ir::ObjectType::StructuredBuffer(ref structured_type) => {
                ast::ObjectType::StructuredBuffer(self.invert_structuredtype(structured_type))
            }
            ir::ObjectType::RWStructuredBuffer(ref structured_type) => {
                ast::ObjectType::RWStructuredBuffer(self.invert_structuredtype(structured_type))
            }
            ir::ObjectType::AppendStructuredBuffer(ref structured_type) => {
                ast::ObjectType::AppendStructuredBuffer(self.invert_structuredtype(structured_type))
            }
            ir::ObjectType::ConsumeStructuredBuffer(ref structured_type) => ast::ObjectType::ConsumeStructuredBuffer(self.invert_structuredtype(structured_type)),
            ir::ObjectType::Texture1D(ref data_type) => {
                ast::ObjectType::Texture1D(self.invert_datatype(data_type))
            }
            ir::ObjectType::Texture1DArray(ref data_type) => {
                ast::ObjectType::Texture1DArray(self.invert_datatype(data_type))
            }
            ir::ObjectType::Texture2D(ref data_type) => {
                ast::ObjectType::Texture2D(self.invert_datatype(data_type))
            }
            ir::ObjectType::Texture2DArray(ref data_type) => {
                ast::ObjectType::Texture2DArray(self.invert_datatype(data_type))
            }
            ir::ObjectType::Texture2DMS(ref data_type) => {
                ast::ObjectType::Texture2DMS(self.invert_datatype(data_type))
            }
            ir::ObjectType::Texture2DMSArray(ref data_type) => {
                ast::ObjectType::Texture2DMSArray(self.invert_datatype(data_type))
            }
            ir::ObjectType::Texture3D(ref data_type) => {
                ast::ObjectType::Texture3D(self.invert_datatype(data_type))
            }
            ir::ObjectType::TextureCube(ref data_type) => {
                ast::ObjectType::TextureCube(self.invert_datatype(data_type))
            }
            ir::ObjectType::TextureCubeArray(ref data_type) => {
                ast::ObjectType::TextureCubeArray(self.invert_datatype(data_type))
            }
            ir::ObjectType::RWTexture1D(ref data_type) => {
                ast::ObjectType::RWTexture1D(self.invert_datatype(data_type))
            }
            ir::ObjectType::RWTexture1DArray(ref data_type) => {
                ast::ObjectType::RWTexture1DArray(self.invert_datatype(data_type))
            }
            ir::ObjectType::RWTexture2D(ref data_type) => {
                ast::ObjectType::RWTexture2D(self.invert_datatype(data_type))
            }
            ir::ObjectType::RWTexture2DArray(ref data_type) => {
                ast::ObjectType::RWTexture2DArray(self.invert_datatype(data_type))
            }
            ir::ObjectType::RWTexture3D(ref data_type) => {
                ast::ObjectType::RWTexture3D(self.invert_datatype(data_type))
            }
            ir::ObjectType::InputPatch => ast::ObjectType::InputPatch,
            ir::ObjectType::OutputPatch => ast::ObjectType::OutputPatch,
        }
    }

    fn invert_typelayout(&self, ty: &ir::TypeLayout) -> ast::TypeLayout {
        match *ty {
            ir::TypeLayout::Void => ast::TypeLayout::Void,
            ir::TypeLayout::Scalar(ref scalar) => {
                ast::TypeLayout::Scalar(self.invert_scalartype(scalar))
            }
            ir::TypeLayout::Vector(ref scalar, ref x) => {
                ast::TypeLayout::Vector(self.invert_scalartype(scalar), *x)
            }
            ir::TypeLayout::Matrix(ref scalar, ref x, ref y) => {
                ast::TypeLayout::Matrix(self.invert_scalartype(scalar), *x, *y)
            }
            ir::TypeLayout::Struct(ref id) => {
                ast::TypeLayout::Custom(match self.get_struct_name(&id) {
                    Some(name) => name,
                    None => "<struct>".to_string(),
                })
            }
            ir::TypeLayout::SamplerState => ast::TypeLayout::SamplerState,
            ir::TypeLayout::Object(ref object_type) => {
                ast::TypeLayout::Object(self.invert_objecttype(object_type))
            }
            ir::TypeLayout::Array(_, _) => {
                panic!("error with array type. can't invert into previous context")
            }
        }
    }

    fn invert_type(&self, ty: &ir::Type) -> ast::Type {
        let &ir::Type(ref tyl, ref modifier) = ty;
        ast::Type(self.invert_typelayout(tyl), self.invert_modifier(modifier))
    }
}

impl StructIdFinder for TypeBlock {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        self.struct_ids.get(name).map(|id| id.clone()).ok_or(
            TyperError::UnknownType(
                ErrorType::Value(ast::Type::from_layout(ast::TypeLayout::Custom(name.clone())))
            )
        )
    }
}

impl ErrorTypeContext for TypeBlock {
    fn ir_type_to_error_type(&self, ty: &ir::Type) -> ErrorType {
        ErrorType::Value(TypeBlock::invert_type(self, ty))
    }

    fn typed_expression_to_error_type(&self, texp: &TypedExpression) -> ErrorType {
        match *texp {
            TypedExpression::Value(_, ExpressionType(ref ty, _)) => {
                ErrorType::Value(TypeBlock::invert_type(self, ty))
            }
            TypedExpression::Function(ref unresolved) => ErrorType::Function(unresolved.clone()),
            TypedExpression::Method(ref unresolved) => ErrorType::Method(unresolved.clone()),
        }
    }
}

impl GlobalContext {
    fn new() -> GlobalContext {
        GlobalContext {
            functions: get_intrinsics(),
            function_names: HashMap::new(),
            next_free_function_id: ir::FunctionId(0),
            types: TypeBlock::new(),
            globals: HashMap::new(),
            global_names: HashMap::new(),
            next_free_global_id: ir::GlobalId(0),
            current_return_type: None,
            global_slots_r: vec![],
            global_slots_rw: vec![],
            global_slots_constants: vec![],
        }
    }

    pub fn insert_function(&mut self,
                           name: String,
                           function_type: FunctionOverload)
                           -> Result<(), TyperError> {
        // Error if a variable of the same name already exists
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name,
                                                       self.ir_type_to_error_type(ty),
                                                       ErrorType::Unknown));
        };

        fn insert_function_name(function_names: &mut HashMap<ir::FunctionId, String>,
                                function_type: FunctionOverload,
                                name: String) {
            match function_type.0 {
                FunctionName::User(id) => {
                    match function_names.entry(id) {
                        Entry::Occupied(_) => {
                            panic!("function id named twice");
                        }
                        Entry::Vacant(vacant) => {
                            vacant.insert(name);
                        }
                    }
                }
                FunctionName::Intrinsic(_) => {}
            }
        }

        // Try to add the function
        match self.functions.entry(name.clone()) {
            Entry::Occupied(mut occupied) => {
                // Fail if the overload already exists
                for &FunctionOverload(_, _, ref args) in &occupied.get().1 {
                    if *args == function_type.2 {
                        return Err(TyperError::ValueAlreadyDefined(name,
                                                                   ErrorType::Unknown,
                                                                   ErrorType::Unknown));
                    }
                }
                // Insert a new overload
                insert_function_name(&mut self.function_names, function_type.clone(), name);
                occupied.get_mut().1.push(function_type);
                Ok(())
            }
            Entry::Vacant(vacant) => {
                // Insert a new function with one overload
                insert_function_name(&mut self.function_names,
                                     function_type.clone(),
                                     name.clone());
                vacant.insert(UnresolvedFunction(name, vec![function_type]));
                Ok(())
            }
        }
    }

    fn find_variable_recur(&self,
                           name: &String,
                           scopes_up: u32)
                           -> Result<TypedExpression, TyperError> {
        assert!(scopes_up != 0);
        match self.functions.get(name) {
            Some(tys) => return Ok(TypedExpression::Function(tys.clone())),
            None => {}
        };
        match self.types.find_variable(name) {
            Some(tys) => return Ok(tys),
            None => {}
        };
        match self.find_global(name) {
            Some(tys) => return Ok(tys),
            None => {}
        };
        Err(TyperError::UnknownIdentifier(name.clone()))
    }

    fn make_function_id(&mut self) -> ir::FunctionId {
        let value = self.next_free_function_id;
        self.next_free_function_id = ir::FunctionId(self.next_free_function_id.0 + 1);
        value
    }

    fn has_variable(&self, name: &String) -> Option<&(ir::Type, ir::GlobalId)> {
        self.globals.get(name)
    }

    fn insert_global(&mut self,
                     name: String,
                     typename: ir::Type)
                     -> Result<ir::GlobalId, TyperError> {
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name,
                                                       self.ir_type_to_error_type(ty),
                                                       self.ir_type_to_error_type(&typename)));
        };
        match self.globals.entry(name.clone()) {
            Entry::Occupied(_) => unreachable!("global variable inserted multiple times"),
            Entry::Vacant(vacant) => {
                let id = self.next_free_global_id;
                self.next_free_global_id = ir::GlobalId(self.next_free_global_id.0 + 1);
                vacant.insert((typename, id));
                match self.global_names.insert(id, name) {
                    Some(_) => panic!("global variable named multiple times"),
                    None => {}
                };
                Ok(id)
            }
        }
    }

    fn find_global(&self, name: &String) -> Option<TypedExpression> {
        match self.globals.get(name) {
            Some(&(ref ty, ref id)) => {
                return Some(TypedExpression::Value(ir::Expression::Global(id.clone()),
                                                   ty.to_lvalue()))
            }
            None => None,
        }
    }

    fn insert_struct(&mut self,
                     name: &String,
                     members: HashMap<String, ir::Type>)
                     -> Option<ir::StructId> {
        self.types.insert_struct(name, members)
    }

    fn insert_cbuffer(&mut self,
                      name: &String,
                      members: HashMap<String, ir::Type>)
                      -> Option<ir::ConstantBufferId> {
        self.types.insert_cbuffer(name, members)
    }

    fn get_type_block(&self) -> &TypeBlock {
        &self.types
    }
}

impl ExpressionContext for GlobalContext {
    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct_member(&self,
                          id: &ir::StructId,
                          member_name: &String)
                          -> Result<ir::Type, TyperError> {
        self.types.find_struct_member(id, member_name)
    }

    fn get_return_type(&self) -> ir::Type {
        self.current_return_type.clone().expect("not inside function")
    }

    fn as_struct_id_finder(&self) -> &StructIdFinder {
        self
    }

    fn as_type_context(&self) -> &ir::TypeContext {
        self
    }
}

impl ir::TypeContext for GlobalContext {
    fn get_local(&self, _: &ir::VariableRef) -> Result<ExpressionType, ir::TypeError> {
        Err(ir::TypeError::InvalidLocal)
    }
    fn get_global(&self, id: &ir::GlobalId) -> Result<ExpressionType, ir::TypeError> {
        for &(ref global_ty, ref global_id) in self.globals.values() {
            if id == global_id {
                return Ok(global_ty.to_lvalue());
            }
        }
        Err(ir::TypeError::GlobalDoesNotExist(id.clone()))
    }
    fn get_constant(&self,
                    id: &ir::ConstantBufferId,
                    name: &str)
                    -> Result<ExpressionType, ir::TypeError> {
        match self.types.cbuffer_definitions.get(id) {
            Some(ref cm) => {
                match cm.get(name) {
                    Some(ref ty) => Ok(ty.to_lvalue()),
                    None => Err(ir::TypeError::ConstantDoesNotExist(id.clone(), name.to_string())),
                }
            }
            None => Err(ir::TypeError::ConstantBufferDoesNotExist(id.clone())),
        }
    }
    fn get_struct_member(&self,
                         id: &ir::StructId,
                         name: &str)
                         -> Result<ExpressionType, ir::TypeError> {
        match self.types.struct_definitions.get(&id) {
            Some(ref cm) => {
                match cm.get(name) {
                    Some(ref ty) => Ok(ty.to_lvalue()),
                    None => {
                        Err(ir::TypeError::StructMemberDoesNotExist(id.clone(), name.to_string()))
                    }
                }
            }
            None => Err(ir::TypeError::StructDoesNotExist(id.clone())),
        }
    }
    fn get_function_return(&self, id: &ir::FunctionId) -> Result<ExpressionType, ir::TypeError> {
        for unresolved in self.functions.values() {
            for overload in &unresolved.1 {
                match overload.0 {
                    FunctionName::Intrinsic(_) => {}
                    FunctionName::User(ref func_id) => {
                        if func_id == id {
                            return Ok(overload.1.clone().to_rvalue());
                        }
                    }
                }
            }
        }
        Err(ir::TypeError::FunctionDoesNotExist(id.clone()))
    }
}

impl StructIdFinder for GlobalContext {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        self.types.find_struct_id(name)
    }
}

impl ErrorTypeContext for GlobalContext {
    fn ir_type_to_error_type(&self, ty: &ir::Type) -> ErrorType {
        self.types.ir_type_to_error_type(ty)
    }

    fn typed_expression_to_error_type(&self, texp: &TypedExpression) -> ErrorType {
        self.types.typed_expression_to_error_type(texp)
    }
}

impl ScopeContext {
    fn from_scope(parent: &ScopeContext) -> ScopeContext {
        ScopeContext {
            parent: Box::new(Context::Scope(parent.clone())),
            variables: VariableBlock::new(),
        }
    }

    fn from_global(parent: &GlobalContext) -> ScopeContext {
        ScopeContext {
            parent: Box::new(Context::Global(parent.clone())),
            variables: VariableBlock::new(),
        }
    }

    fn find_variable_recur(&self,
                           name: &String,
                           scopes_up: u32)
                           -> Result<TypedExpression, TyperError> {
        match self.variables.find_variable(name, scopes_up) {
            Some(texp) => return Ok(texp),
            None => self.parent.find_variable_recur(name, scopes_up + 1),
        }
    }

    fn destruct(self) -> ir::ScopedDeclarations {
        ir::ScopedDeclarations { variables: self.variables.destruct() }
    }

    fn insert_variable(&mut self,
                       name: String,
                       typename: ir::Type)
                       -> Result<ir::VariableId, TyperError> {
        let type_block = self.parent.get_type_block();
        let variables = &mut self.variables;
        variables.insert_variable(name, typename, type_block)
    }

    fn get_type_block(&self) -> &TypeBlock {
        self.parent.get_type_block()
    }
}

impl ExpressionContext for ScopeContext {
    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct_member(&self,
                          id: &ir::StructId,
                          member_name: &String)
                          -> Result<ir::Type, TyperError> {
        self.parent.find_struct_member(id, member_name)
    }

    fn get_return_type(&self) -> ir::Type {
        self.parent.get_return_type()
    }

    fn as_struct_id_finder(&self) -> &StructIdFinder {
        self
    }

    fn as_type_context(&self) -> &ir::TypeContext {
        self
    }
}

impl ir::TypeContext for ScopeContext {
    fn get_local(&self, var_ref: &ir::VariableRef) -> Result<ExpressionType, ir::TypeError> {
        let &ir::VariableRef(ref id, ref scope) = var_ref;
        match scope.0 {
            0 => {
                for &(ref var_ty, ref var_id) in self.variables.variables.values() {
                    if id == var_id {
                        return Ok(var_ty.to_lvalue());
                    }
                }
                Err(ir::TypeError::InvalidLocal)
            }
            up => self.parent.get_local(&ir::VariableRef(id.clone(), ir::ScopeRef(up - 1))),
        }
    }
    fn get_global(&self, id: &ir::GlobalId) -> Result<ExpressionType, ir::TypeError> {
        self.parent.get_global(id)
    }
    fn get_constant(&self,
                    id: &ir::ConstantBufferId,
                    name: &str)
                    -> Result<ExpressionType, ir::TypeError> {
        self.parent.get_constant(id, name)
    }
    fn get_struct_member(&self,
                         id: &ir::StructId,
                         name: &str)
                         -> Result<ExpressionType, ir::TypeError> {
        self.parent.get_struct_member(id, name)
    }
    fn get_function_return(&self, id: &ir::FunctionId) -> Result<ExpressionType, ir::TypeError> {
        self.parent.get_function_return(id)
    }
}

impl StructIdFinder for ScopeContext {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        self.parent.find_struct_id(name)
    }
}

impl ErrorTypeContext for ScopeContext {
    fn ir_type_to_error_type(&self, ty: &ir::Type) -> ErrorType {
        self.parent.ir_type_to_error_type(ty)
    }
    fn typed_expression_to_error_type(&self, texp: &TypedExpression) -> ErrorType {
        self.parent.typed_expression_to_error_type(texp)
    }
}

impl Context {
    fn find_variable_recur(&self,
                           name: &String,
                           scopes_up: u32)
                           -> Result<TypedExpression, TyperError> {
        match *self {
            Context::Global(ref global) => global.find_variable_recur(name, scopes_up),
            Context::Scope(ref scope) => scope.find_variable_recur(name, scopes_up),
        }
    }

    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        match *self {
            Context::Global(ref global) => global.find_struct_id(name),
            Context::Scope(ref scope) => scope.find_struct_id(name),
        }
    }

    fn get_return_type(&self) -> ir::Type {
        match *self {
            Context::Global(ref global) => global.get_return_type(),
            Context::Scope(ref scope) => scope.get_return_type(),
        }
    }

    fn find_struct_member(&self,
                          id: &ir::StructId,
                          member_name: &String)
                          -> Result<ir::Type, TyperError> {
        match *self {
            Context::Global(ref global) => global.find_struct_member(id, member_name),
            Context::Scope(ref scope) => scope.find_struct_member(id, member_name),
        }
    }

    fn get_type_block(&self) -> &TypeBlock {
        match *self {
            Context::Global(ref global) => global.get_type_block(),
            Context::Scope(ref scope) => scope.get_type_block(),
        }
    }
}

impl StructIdFinder for Context {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        match *self {
            Context::Global(ref global) => global.find_struct_id(name),
            Context::Scope(ref scope) => scope.find_struct_id(name),
        }
    }
}

impl ErrorTypeContext for Context {
    fn ir_type_to_error_type(&self, ty: &ir::Type) -> ErrorType {
        match *self {
            Context::Global(ref global) => global.ir_type_to_error_type(ty),
            Context::Scope(ref scope) => scope.ir_type_to_error_type(ty),
        }
    }
    fn typed_expression_to_error_type(&self, texp: &TypedExpression) -> ErrorType {
        match *self {
            Context::Global(ref global) => global.typed_expression_to_error_type(texp),
            Context::Scope(ref scope) => scope.typed_expression_to_error_type(texp),
        }
    }
}

impl ir::TypeContext for Context {
    fn get_local(&self, var_ref: &ir::VariableRef) -> Result<ExpressionType, ir::TypeError> {
        match *self {
            Context::Global(ref global) => global.get_local(var_ref),
            Context::Scope(ref scope) => scope.get_local(var_ref),
        }
    }
    fn get_global(&self, id: &ir::GlobalId) -> Result<ExpressionType, ir::TypeError> {
        match *self {
            Context::Global(ref global) => global.get_global(id),
            Context::Scope(ref scope) => scope.get_global(id),
        }
    }
    fn get_constant(&self,
                    id: &ir::ConstantBufferId,
                    name: &str)
                    -> Result<ExpressionType, ir::TypeError> {
        match *self {
            Context::Global(ref global) => global.get_constant(id, name),
            Context::Scope(ref scope) => scope.get_constant(id, name),
        }
    }
    fn get_struct_member(&self,
                         id: &ir::StructId,
                         name: &str)
                         -> Result<ExpressionType, ir::TypeError> {
        match *self {
            Context::Global(ref global) => global.get_struct_member(id, name),
            Context::Scope(ref scope) => scope.get_struct_member(id, name),
        }
    }
    fn get_function_return(&self, id: &ir::FunctionId) -> Result<ExpressionType, ir::TypeError> {
        match *self {
            Context::Global(ref global) => global.get_function_return(id),
            Context::Scope(ref scope) => scope.get_function_return(id),
        }
    }
}

/// Create a map of all the intrinsic functions we need to parse
fn get_intrinsics() -> HashMap<String, UnresolvedFunction> {

    let funcs = intrinsics::get_intrinsics();

    let mut strmap: HashMap<String, UnresolvedFunction> = HashMap::new();
    for &(ref return_type, ref name, ref params, ref factory) in funcs {
        let overload = FunctionOverload(FunctionName::Intrinsic(factory.clone()),
                                        return_type.clone(),
                                        params.to_vec());
        match strmap.entry(name.to_string()) {
            Entry::Occupied(mut occupied) => {
                let &mut UnresolvedFunction(_, ref mut overloads) = occupied.get_mut();
                overloads.push(overload);
            }
            Entry::Vacant(vacant) => {
                vacant.insert(UnresolvedFunction(name.to_string(), vec![overload]));
            }
        }
    }
    strmap
}

fn parse_scalartype(ty: &ast::ScalarType) -> Result<ir::ScalarType, TyperError> {
    Ok(match *ty {
        ast::ScalarType::Bool => ir::ScalarType::Bool,
        ast::ScalarType::UntypedInt => ir::ScalarType::UntypedInt,
        ast::ScalarType::Int => ir::ScalarType::Int,
        ast::ScalarType::UInt => ir::ScalarType::UInt,
        ast::ScalarType::Half => ir::ScalarType::Half,
        ast::ScalarType::Float => ir::ScalarType::Float,
        ast::ScalarType::Double => ir::ScalarType::Double,
    })
}

fn parse_row_order(row_order: &ast::RowOrder) -> ir::RowOrder {
    match *row_order {
        ast::RowOrder::Row => ir::RowOrder::Row,
        ast::RowOrder::Column => ir::RowOrder::Column,
    }
}

fn parse_modifier(modifier: &ast::TypeModifier) -> ir::TypeModifier {
    ir::TypeModifier {
        is_const: modifier.is_const,
        row_order: parse_row_order(&modifier.row_order),
        precise: modifier.precise,
        volatile: modifier.volatile,
    }
}

fn parse_datalayout(ty: &ast::DataLayout) -> Result<ir::DataLayout, TyperError> {
    Ok(match *ty {
        ast::DataLayout::Scalar(ref scalar) => {
            ir::DataLayout::Scalar(try!(parse_scalartype(scalar)))
        }
        ast::DataLayout::Vector(ref scalar, ref x) => {
            ir::DataLayout::Vector(try!(parse_scalartype(scalar)), *x)
        }
        ast::DataLayout::Matrix(ref scalar, ref x, ref y) => {
            ir::DataLayout::Matrix(try!(parse_scalartype(scalar)), *x, *y)
        }
    })
}

fn parse_datatype(ty: &ast::DataType) -> Result<ir::DataType, TyperError> {
    let &ast::DataType(ref tyl, ref modifier) = ty;
    Ok(ir::DataType(try!(parse_datalayout(tyl)), parse_modifier(modifier)))
}

fn parse_structuredlayout(ty: &ast::StructuredLayout,
                          struct_finder: &StructIdFinder)
                          -> Result<ir::StructuredLayout, TyperError> {
    Ok(match *ty {
        ast::StructuredLayout::Scalar(ref scalar) => {
            ir::StructuredLayout::Scalar(try!(parse_scalartype(scalar)))
        }
        ast::StructuredLayout::Vector(ref scalar, ref x) => {
            ir::StructuredLayout::Vector(try!(parse_scalartype(scalar)), *x)
        }
        ast::StructuredLayout::Matrix(ref scalar, ref x, ref y) => {
            ir::StructuredLayout::Matrix(try!(parse_scalartype(scalar)), *x, *y)
        }
        ast::StructuredLayout::Custom(ref name) => {
            ir::StructuredLayout::Struct(try!(struct_finder.find_struct_id(name)))
        }
    })
}

fn parse_structuredtype(ty: &ast::StructuredType,
                        struct_finder: &StructIdFinder)
                        -> Result<ir::StructuredType, TyperError> {
    let &ast::StructuredType(ref tyl, ref modifier) = ty;
    Ok(ir::StructuredType(try!(parse_structuredlayout(tyl, struct_finder)),
                          parse_modifier(modifier)))
}

fn parse_objecttype(ty: &ast::ObjectType,
                    struct_finder: &StructIdFinder)
                    -> Result<ir::ObjectType, TyperError> {
    Ok(match *ty {
        ast::ObjectType::Buffer(ref data_type) => {
            ir::ObjectType::Buffer(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::RWBuffer(ref data_type) => {
            ir::ObjectType::RWBuffer(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::ByteAddressBuffer => ir::ObjectType::ByteAddressBuffer,
        ast::ObjectType::RWByteAddressBuffer => ir::ObjectType::RWByteAddressBuffer,
        ast::ObjectType::StructuredBuffer(ref structured_type) => {
            ir::ObjectType::StructuredBuffer(try!(parse_structuredtype(structured_type,
                                                                       struct_finder)))
        }
        ast::ObjectType::RWStructuredBuffer(ref structured_type) => {
            ir::ObjectType::RWStructuredBuffer(try!(parse_structuredtype(structured_type,
                                                                         struct_finder)))
        }
        ast::ObjectType::AppendStructuredBuffer(ref structured_type) => {
            ir::ObjectType::AppendStructuredBuffer(try!(parse_structuredtype(structured_type,
                                                                             struct_finder)))
        }
        ast::ObjectType::ConsumeStructuredBuffer(ref structured_type) => {
            ir::ObjectType::ConsumeStructuredBuffer(try!(parse_structuredtype(structured_type,
                                                                              struct_finder)))
        }
        ast::ObjectType::Texture1D(ref data_type) => {
            ir::ObjectType::Texture1D(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::Texture1DArray(ref data_type) => {
            ir::ObjectType::Texture1DArray(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::Texture2D(ref data_type) => {
            ir::ObjectType::Texture2D(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::Texture2DArray(ref data_type) => {
            ir::ObjectType::Texture2DArray(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::Texture2DMS(ref data_type) => {
            ir::ObjectType::Texture2DMS(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::Texture2DMSArray(ref data_type) => {
            ir::ObjectType::Texture2DMSArray(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::Texture3D(ref data_type) => {
            ir::ObjectType::Texture3D(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::TextureCube(ref data_type) => {
            ir::ObjectType::TextureCube(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::TextureCubeArray(ref data_type) => {
            ir::ObjectType::TextureCubeArray(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::RWTexture1D(ref data_type) => {
            ir::ObjectType::RWTexture1D(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::RWTexture1DArray(ref data_type) => {
            ir::ObjectType::RWTexture1DArray(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::RWTexture2D(ref data_type) => {
            ir::ObjectType::RWTexture2D(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::RWTexture2DArray(ref data_type) => {
            ir::ObjectType::RWTexture2DArray(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::RWTexture3D(ref data_type) => {
            ir::ObjectType::RWTexture3D(try!(parse_datatype(data_type)))
        }
        ast::ObjectType::InputPatch => ir::ObjectType::InputPatch,
        ast::ObjectType::OutputPatch => ir::ObjectType::OutputPatch,
    })
}

fn parse_typelayout(ty: &ast::TypeLayout,
                    struct_finder: &StructIdFinder)
                    -> Result<ir::TypeLayout, TyperError> {
    Ok(match *ty {
        ast::TypeLayout::Void => ir::TypeLayout::void(),
        ast::TypeLayout::Scalar(ref scalar) => {
            ir::TypeLayout::Scalar(try!(parse_scalartype(scalar)))
        }
        ast::TypeLayout::Vector(ref scalar, ref x) => {
            ir::TypeLayout::Vector(try!(parse_scalartype(scalar)), *x)
        }
        ast::TypeLayout::Matrix(ref scalar, ref x, ref y) => {
            ir::TypeLayout::Matrix(try!(parse_scalartype(scalar)), *x, *y)
        }
        ast::TypeLayout::Custom(ref name) => {
            ir::TypeLayout::Struct(try!(struct_finder.find_struct_id(name)))
        }
        ast::TypeLayout::SamplerState => ir::TypeLayout::SamplerState,
        ast::TypeLayout::Object(ref object_type) => {
            ir::TypeLayout::Object(try!(parse_objecttype(object_type, struct_finder)))
        }
    })
}

fn parse_type(ty: &ast::Type, struct_finder: &StructIdFinder) -> Result<ir::Type, TyperError> {
    let &ast::Type(ref tyl, ref modifier) = ty;
    Ok(ir::Type(try!(parse_typelayout(tyl, struct_finder)),
                parse_modifier(modifier)))
}

fn parse_interpolationmodifier(im: &ast::InterpolationModifier)
                               -> Result<ir::InterpolationModifier, TyperError> {
    Ok(match *im {
        ast::InterpolationModifier::NoInterpolation => ir::InterpolationModifier::NoInterpolation,
        ast::InterpolationModifier::Linear => ir::InterpolationModifier::Linear,
        ast::InterpolationModifier::Centroid => ir::InterpolationModifier::Centroid,
        ast::InterpolationModifier::NoPerspective => ir::InterpolationModifier::NoPerspective,
        ast::InterpolationModifier::Sample => ir::InterpolationModifier::Sample,
    })
}

fn parse_globalstorage(local_storage: &ast::GlobalStorage) -> Result<ir::GlobalStorage, TyperError> {
    Ok(match *local_storage {
        ast::GlobalStorage::Extern => ir::GlobalStorage::Extern,
        ast::GlobalStorage::Static => ir::GlobalStorage::Static,
        ast::GlobalStorage::GroupShared => ir::GlobalStorage::GroupShared,
    })
}

fn parse_globaltype(global_type: &ast::GlobalType,
                    struct_finder: &StructIdFinder)
                    -> Result<ir::GlobalType, TyperError> {
    let ty = try!(parse_type(&global_type.0, struct_finder));
    let interp = match global_type.2 {
        Some(ref im) => Some(try!(parse_interpolationmodifier(im))),
        None => None,
    };
    Ok(ir::GlobalType(ty, try!(parse_globalstorage(&global_type.1)), interp))
}

fn parse_inputmodifier(it: &ast::InputModifier) -> Result<ir::InputModifier, TyperError> {
    Ok(match *it {
        ast::InputModifier::In => ir::InputModifier::In,
        ast::InputModifier::Out => ir::InputModifier::Out,
        ast::InputModifier::InOut => ir::InputModifier::InOut,
    })
}

fn parse_paramtype(param_type: &ast::ParamType,
                   struct_finder: &StructIdFinder)
                   -> Result<ir::ParamType, TyperError> {
    let ty = try!(parse_type(&param_type.0, struct_finder));
    let interp = match param_type.2 {
        Some(ref im) => Some(try!(parse_interpolationmodifier(im))),
        None => None,
    };
    Ok(ir::ParamType(ty, try!(parse_inputmodifier(&param_type.1)), interp))
}

fn parse_localstorage(local_storage: &ast::LocalStorage) -> Result<ir::LocalStorage, TyperError> {
    Ok(match *local_storage {
        ast::LocalStorage::Local => ir::LocalStorage::Local,
        ast::LocalStorage::Static => ir::LocalStorage::Static,
    })
}

fn parse_localtype(local_type: &ast::LocalType,
                   struct_finder: &StructIdFinder)
                   -> Result<ir::LocalType, TyperError> {
    let ty = try!(parse_type(&local_type.0, struct_finder));
    let interp = match local_type.2 {
        Some(ref im) => Some(try!(parse_interpolationmodifier(im))),
        None => None,
    };
    Ok(ir::LocalType(ty, try!(parse_localstorage(&local_type.1)), interp))
}

fn find_function_type(overloads: &Vec<FunctionOverload>,
                      param_types: &[ExpressionType])
                      -> Result<(FunctionOverload, Vec<ImplicitConversion>), TyperError> {

    fn find_overload_casts(overload: &FunctionOverload,
                           param_types: &[ExpressionType])
                           -> Result<Vec<ImplicitConversion>, ()> {
        let mut overload_casts = Vec::with_capacity(param_types.len());
        for (required_type, source_type) in overload.2.iter().zip(param_types.iter()) {
            let &ir::ParamType(ref ty, ref it, ref interp) = required_type;

            let ety = match *it {
                ir::InputModifier::In => ty.to_rvalue(),
                ir::InputModifier::Out | ir::InputModifier::InOut => ty.to_lvalue(),
            };
            match *interp {
                Some(_) => return Err(()),
                None => {}
            };

            if let Ok(cast) = ImplicitConversion::find(source_type, &ety) {
                overload_casts.push(cast)
            } else {
                return Err(());
            }
        }
        Ok(overload_casts)
    }

    let mut casts = Vec::with_capacity(overloads.len());
    for overload in overloads {
        if param_types.len() == overload.2.len() {
            if let Ok(param_casts) = find_overload_casts(overload, param_types) {
                casts.push((overload.clone(), param_casts))
            }
        }
    }
    for &(ref candidate, ref candidate_casts) in &casts {
        let mut winning = true;
        for &(ref against, ref against_casts) in &casts {
            if candidate == against {
                continue;
            }
            assert_eq!(candidate_casts.len(), against_casts.len());
            let mut at_least_one_better_than = false;
            let mut not_worse_than = true;
            for (candidate_cast, against_cast) in candidate_casts.iter().zip(against_casts) {
                let candidate_rank = candidate_cast.get_rank();
                let against_rank = against_cast.get_rank();
                match candidate_rank.compare(&against_rank) {
                    ConversionPriority::Better => at_least_one_better_than = true,
                    ConversionPriority::Equal => {}
                    ConversionPriority::Worse => not_worse_than = false,
                };
            }
            if !at_least_one_better_than || !not_worse_than {
                winning = false;
                break;
            }
        }
        if winning {
            return Ok((candidate.clone(), candidate_casts.clone()));
        }
    }
    Err(TyperError::FunctionArgumentTypeMismatch(overloads.clone(), param_types.to_vec()))
}

fn apply_casts(casts: Vec<ImplicitConversion>, values: Vec<ir::Expression>) -> Vec<ir::Expression> {
    assert_eq!(casts.len(), values.len());
    values.into_iter()
          .enumerate()
          .map(|(index, value)| casts[index].apply(value))
          .collect::<Vec<_>>()
}

fn write_function(unresolved: UnresolvedFunction,
                  param_types: &[ExpressionType],
                  param_values: Vec<ir::Expression>)
                  -> Result<TypedExpression, TyperError> {
    // Find the matching function overload
    let (FunctionOverload(name, return_type_ty, _), casts) =
        try!(find_function_type(&unresolved.1, param_types));
    // Apply implicit casts
    let param_values = apply_casts(casts, param_values);
    let return_type = return_type_ty.to_rvalue();

    match name {
        FunctionName::Intrinsic(factory) => {
            Ok(TypedExpression::Value(
                ir::Expression::Intrinsic(Box::new(factory.create_intrinsic(&param_values))),
                return_type
            ))
        }
        FunctionName::User(id) => {
            Ok(TypedExpression::Value(ir::Expression::Call(id, param_values), return_type))
        }
    }
}

fn write_method(unresolved: UnresolvedMethod,
                param_types: &[ExpressionType],
                param_values: Vec<ir::Expression>)
                -> Result<TypedExpression, TyperError> {
    // Find the matching method overload
    let (FunctionOverload(name, return_type_ty, _), casts) =
        try!(find_function_type(&unresolved.2, param_types));
    // Apply implicit casts
    let mut param_values = apply_casts(casts, param_values);
    // Add struct as implied first argument
    param_values.insert(0, unresolved.3);
    let return_type = return_type_ty.to_rvalue();

    match name {
        FunctionName::Intrinsic(factory) => {
            Ok(TypedExpression::Value(
                ir::Expression::Intrinsic(Box::new(factory.create_intrinsic(&param_values))),
                return_type
            ))
        }
        FunctionName::User(_) => panic!("User defined methods should not exist"),
    }
}

fn parse_literal(ast: &ast::Literal) -> Result<TypedExpression, TyperError> {
    match ast {
        &ast::Literal::Bool(b) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Bool(b)),
                                      ir::Type::bool().to_rvalue()))
        }
        &ast::Literal::UntypedInt(i) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::UntypedInt(i)),
                                      ir::Type::from_scalar(ir::ScalarType::UntypedInt)
                                          .to_rvalue()))
        }
        &ast::Literal::Int(i) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Int(i)),
                                      ir::Type::int().to_rvalue()))
        }
        &ast::Literal::UInt(i) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::UInt(i)),
                                      ir::Type::uint().to_rvalue()))
        }
        &ast::Literal::Long(i) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Long(i)),
                                      ir::Type::from_scalar(ir::ScalarType::UntypedInt)
                                          .to_rvalue()))
        }
        &ast::Literal::Half(f) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Half(f)),
                                      ir::Type::float().to_rvalue()))
        }
        &ast::Literal::Float(f) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Float(f)),
                                      ir::Type::float().to_rvalue()))
        }
        &ast::Literal::Double(f) => {
            Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Double(f)),
                                      ir::Type::double().to_rvalue()))
        }
    }
}

fn parse_expr_unaryop(op: &ast::UnaryOp,
                      expr: &ast::Expression,
                      context: &ExpressionContext)
                      -> Result<TypedExpression, TyperError> {
    match try!(parse_expr(expr, context)) {
        TypedExpression::Value(expr_ir, expr_ty) => {
            fn enforce_increment_type(ety: &ExpressionType,
                                      op: &ast::UnaryOp)
                                      -> Result<(), TyperError> {
                match *ety {
                    ir::ExpressionType(_, ir::ValueType::Rvalue) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    ir::ExpressionType(ir::Type(ir::TypeLayout::Scalar(ir::ScalarType::Bool), _),
                                       _) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    ir::ExpressionType(ir::Type(ir::TypeLayout::Vector(ir::ScalarType::Bool, _),
                                                _),
                                       _) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    ir::ExpressionType(ir::Type(ir::TypeLayout::Matrix(ir::ScalarType::Bool,
                                                                       _,
                                                                       _),
                                                _),
                                       _) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    _ => Ok(()),
                }
            }
            let (intrinsic, ety) = match *op {
                ast::UnaryOp::PrefixIncrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (ir::Intrinsic::PrefixIncrement(expr_ty.0.clone(), expr_ir),
                     expr_ty)
                }
                ast::UnaryOp::PrefixDecrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (ir::Intrinsic::PrefixDecrement(expr_ty.0.clone(), expr_ir),
                     expr_ty)
                }
                ast::UnaryOp::PostfixIncrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (ir::Intrinsic::PostfixIncrement(expr_ty.0.clone(), expr_ir),
                     expr_ty)
                }
                ast::UnaryOp::PostfixDecrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (ir::Intrinsic::PostfixDecrement(expr_ty.0.clone(), expr_ir),
                     expr_ty)
                }
                ast::UnaryOp::Plus => {
                    (ir::Intrinsic::Plus(expr_ty.0.clone(), expr_ir),
                     expr_ty.0.to_rvalue())
                }
                ast::UnaryOp::Minus => {
                    (ir::Intrinsic::Minus(expr_ty.0.clone(), expr_ir),
                     expr_ty.0.to_rvalue())
                }
                ast::UnaryOp::LogicalNot => {
                    let ty = match expr_ty.0 {
                        ir::Type(ir::TypeLayout::Scalar(_), _) => {
                            ir::Type::from_layout(ir::TypeLayout::Scalar(ir::ScalarType::Bool))
                        }
                        ir::Type(ir::TypeLayout::Vector(_, x), _) => {
                            ir::Type::from_layout(ir::TypeLayout::Vector(ir::ScalarType::Bool, x))
                        }
                        ir::Type(ir::TypeLayout::Matrix(_, x, y), _) => {
                            ir::Type::from_layout(ir::TypeLayout::Matrix(ir::ScalarType::Bool,
                                                                         x,
                                                                         y))
                        }
                        _ => {
                            return Err(TyperError::UnaryOperationWrongTypes(op.clone(),
                                                                            ErrorType::Unknown))
                        }
                    };
                    let ety = ty.clone().to_rvalue();
                    (ir::Intrinsic::LogicalNot(ty, expr_ir), ety)
                }
                ast::UnaryOp::BitwiseNot => {
                    match (expr_ty.0).0 {
                        ir::TypeLayout::Scalar(ir::ScalarType::Int) |
                        ir::TypeLayout::Scalar(ir::ScalarType::UInt) => {
                            (ir::Intrinsic::BitwiseNot(expr_ty.0.clone(), expr_ir),
                             expr_ty.0.to_rvalue())
                        }
                        _ => {
                            return Err(TyperError::UnaryOperationWrongTypes(op.clone(),
                                                                            ErrorType::Unknown))
                        }
                    }
                }
            };
            Ok(TypedExpression::Value(ir::Expression::Intrinsic(Box::new(intrinsic)), ety))
        }
        _ => Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown)),
    }
}

fn resolve_arithmetic_types
    (binop: &ir::BinOp,
     left: &ExpressionType,
     right: &ExpressionType,
     context: &ExpressionContext)
     -> Result<(ImplicitConversion, ImplicitConversion, ExpressionType), TyperError> {
    use slp_lang_hir::Type;
    use slp_lang_hir::ScalarType;

    fn common_real_type(left: &ScalarType, right: &ScalarType) -> Result<ir::ScalarType, ()> {

        // The limited number of hlsl types means these happen to always have one type being the common type
        fn get_order(ty: &ScalarType) -> Result<u32, ()> {
            match *ty {
                ScalarType::Int => Ok(0),
                ScalarType::UInt => Ok(1),
                ScalarType::Half => Ok(2),
                ScalarType::Float => Ok(3),
                ScalarType::Double => Ok(4),
                _ => Err(()),
            }
        }

        let left = match *left {
            ScalarType::UntypedInt => ScalarType::Int,
            ScalarType::Bool => return Err(()),
            ref scalar => scalar.clone(),
        };
        let right = match *right {
            ScalarType::UntypedInt => ScalarType::Int,
            ScalarType::Bool => return Err(()),
            ref scalar => scalar.clone(),
        };

        let left_order = try!(get_order(&left));
        let right_order = try!(get_order(&right));

        if left_order > right_order {
            Ok(left)
        } else {
            Ok(right)
        }
    }

    // Calculate the output type from the input type and operation
    fn output_type(left: Type, right: Type, op: &ir::BinOp) -> ExpressionType {
        // Actually implement this by using the hir's type parser
        ir::TypeParser::get_binary_operation_output_type(left, right, op)
    }

    fn do_noerror(op: &ir::BinOp,
                  left: &ExpressionType,
                  right: &ExpressionType)
                  -> Result<(ImplicitConversion, ImplicitConversion, ExpressionType), ()> {
        let &ExpressionType(ir::Type(ref left_l, ref modl), _) = left;
        let &ExpressionType(ir::Type(ref right_l, ref modr), _) = right;
        let (ltl, rtl) = match (left_l, right_l) {
            (&ir::TypeLayout::Scalar(ref ls),
             &ir::TypeLayout::Scalar(ref rs)) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = ir::TypeLayout::from_scalar(common_scalar);
                let common_right = common_left.clone();
                (common_left, common_right)
            }
            (&ir::TypeLayout::Scalar(ref ls),
             &ir::TypeLayout::Vector(ref rs, ref x2)) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = ir::TypeLayout::from_scalar(common_scalar.clone());
                let common_right = ir::TypeLayout::from_vector(common_scalar, *x2);
                (common_left, common_right)
            }
            (&ir::TypeLayout::Vector(ref ls, ref x1),
             &ir::TypeLayout::Scalar(ref rs)) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = ir::TypeLayout::from_vector(common_scalar.clone(), *x1);
                let common_right = ir::TypeLayout::from_scalar(common_scalar);
                (common_left, common_right)
            }
            (&ir::TypeLayout::Vector(ref ls, ref x1),
             &ir::TypeLayout::Vector(ref rs, ref x2))
                if x1 == x2 => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = ir::TypeLayout::from_vector(common_scalar, *x2);
                let common_right = common_left.clone();
                (common_left, common_right)
            }
            (&ir::TypeLayout::Matrix(ref ls, ref x1, ref y1),
             &ir::TypeLayout::Matrix(ref rs, ref x2, ref y2))
                if x1 == x2 && y1 == y2 => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = ir::TypeLayout::from_matrix(common_scalar, *x2, *y2);
                let common_right = common_left.clone();
                (common_left, common_right)
            }
            _ => return Err(()),
        };
        let out_mod = ir::TypeModifier {
            is_const: false,
            row_order: ir::RowOrder::Column,
            precise: modl.precise || modr.precise,
            volatile: false,
        };
        let candidate_left = Type(ltl.clone(), out_mod.clone());
        let candidate_right = Type(rtl.clone(), out_mod.clone());
        let output_type = output_type(candidate_left, candidate_right, op);
        let elt = ExpressionType(ir::Type(ltl, out_mod.clone()), ir::ValueType::Rvalue);
        let lc = try!(ImplicitConversion::find(left, &elt));
        let ert = ExpressionType(ir::Type(rtl, out_mod), ir::ValueType::Rvalue);
        let rc = try!(ImplicitConversion::find(right, &ert));
        Ok((lc, rc, output_type))
    }

    match do_noerror(binop, left, right) {
        Ok(res) => Ok(res),
        Err(_) => {
            Err(TyperError::BinaryOperationWrongTypes(binop.clone(),
                                                      context.exp_type_to_error_type(left),
                                                      context.exp_type_to_error_type(right)))
        }
    }
}

fn parse_expr_binop(op: &ast::BinOp,
                    lhs: &ast::Expression,
                    rhs: &ast::Expression,
                    context: &ExpressionContext)
                    -> Result<TypedExpression, TyperError> {
    let lhs_texp = try!(parse_expr(lhs, context));
    let rhs_texp = try!(parse_expr(rhs, context));
    let lhs_pt = context.typed_expression_to_error_type(&lhs_texp);
    let rhs_pt = context.typed_expression_to_error_type(&rhs_texp);
    let (lhs_ir, lhs_type) = match lhs_texp {
        TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
        _ => return Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
    };
    let (rhs_ir, rhs_type) = match rhs_texp {
        TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
        _ => return Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
    };
    match *op {
        ast::BinOp::Add |
        ast::BinOp::Subtract |
        ast::BinOp::Multiply |
        ast::BinOp::Divide |
        ast::BinOp::Modulus |
        ast::BinOp::LessThan |
        ast::BinOp::LessEqual |
        ast::BinOp::GreaterThan |
        ast::BinOp::GreaterEqual |
        ast::BinOp::Equality |
        ast::BinOp::Inequality => {
            let types = try!(resolve_arithmetic_types(op, &lhs_type, &rhs_type, context));
            let (lhs_cast, rhs_cast, output_type) = types;
            let lhs_final = Box::new(lhs_cast.apply(lhs_ir));
            let rhs_final = Box::new(rhs_cast.apply(rhs_ir));
            let node = ir::Expression::BinaryOperation(op.clone(), lhs_final, rhs_final);
            Ok(TypedExpression::Value(node, output_type))
        }
        ast::BinOp::LeftShift |
        ast::BinOp::RightShift => Err(TyperError::Unimplemented),
        ast::BinOp::BitwiseAnd |
        ast::BinOp::BitwiseOr |
        ast::BinOp::BitwiseXor |
        ast::BinOp::BooleanAnd |
        ast::BinOp::BooleanOr => {
            let lhs_tyl = &(lhs_type.0).0;
            let rhs_tyl = &(rhs_type.0).0;
            let lhs_mod = &(lhs_type.0).1;
            let rhs_mod = &(rhs_type.0).1;
            let scalar = if *op == ast::BinOp::BooleanAnd || *op == ast::BinOp::BooleanOr {
                ir::ScalarType::Bool
            } else {
                let lhs_scalar = try!(lhs_tyl.to_scalar()
                                             .ok_or(TyperError::BinaryOperationNonNumericType));
                let rhs_scalar = try!(rhs_tyl.to_scalar()
                                             .ok_or(TyperError::BinaryOperationNonNumericType));
                match (lhs_scalar, rhs_scalar) {
                    (ir::ScalarType::Int, ir::ScalarType::Int) => ir::ScalarType::Int,
                    (ir::ScalarType::Int, ir::ScalarType::UInt) => ir::ScalarType::UInt,
                    (ir::ScalarType::UInt, ir::ScalarType::Int) => ir::ScalarType::UInt,
                    (ir::ScalarType::UInt, ir::ScalarType::UInt) => ir::ScalarType::UInt,
                    _ => {
                        return Err(TyperError::BinaryOperationWrongTypes(op.clone(),
                                                                         lhs_pt,
                                                                         rhs_pt))
                    }
                }
            };
            let x = ir::TypeLayout::max_dim(lhs_tyl.to_x(), rhs_tyl.to_x());
            let y = ir::TypeLayout::max_dim(lhs_tyl.to_y(), rhs_tyl.to_y());
            let tyl = ir::TypeLayout::from_numeric(scalar, x, y);
            let out_mod = ir::TypeModifier {
                is_const: false,
                row_order: ir::RowOrder::Column,
                precise: lhs_mod.precise || rhs_mod.precise,
                volatile: false,
            };
            let ty = ir::Type(tyl, out_mod).to_rvalue();
            let lhs_cast = match ImplicitConversion::find(&lhs_type, &ty) {
                Ok(cast) => cast,
                Err(()) => {
                    return Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt))
                }
            };
            let rhs_cast = match ImplicitConversion::find(&rhs_type, &ty) {
                Ok(cast) => cast,
                Err(()) => {
                    return Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt))
                }
            };
            assert_eq!(lhs_cast.get_target_type(), rhs_cast.get_target_type());
            let lhs_final = lhs_cast.apply(lhs_ir);
            let rhs_final = rhs_cast.apply(rhs_ir);
            Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(),
                                                                      Box::new(lhs_final),
                                                                      Box::new(rhs_final)),
                                      rhs_cast.get_target_type()))
        }
        ast::BinOp::Assignment |
        ast::BinOp::SumAssignment |
        ast::BinOp::DifferenceAssignment => {
            let required_rtype = match lhs_type.1 {
                ir::ValueType::Lvalue => ExpressionType(lhs_type.0.clone(), ir::ValueType::Rvalue),
                _ => return Err(TyperError::LvalueRequired),
            };
            match ImplicitConversion::find(&rhs_type, &required_rtype) {
                Ok(rhs_cast) => {
                    let rhs_final = rhs_cast.apply(rhs_ir);
                    Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(),
                                                                              Box::new(lhs_ir),
                                                                              Box::new(rhs_final)),
                                              lhs_type))
                }
                Err(()) => Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
            }
        }
    }
}

fn unwrap_value_expr(texp: TypedExpression,
                     context: &ExpressionContext)
                     -> Result<(ir::Expression, ExpressionType), TyperError> {
    Ok(match texp {
        TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
        texp => return Err(TyperError::ExpectedValueExpression(context.typed_expression_to_error_type(&texp))),
    })
}

fn parse_expr_ternary(cond: &ast::Expression,
                      lhs: &ast::Expression,
                      rhs: &ast::Expression,
                      context: &ExpressionContext)
                      -> Result<TypedExpression, TyperError> {
    let cond_texp = try!(parse_expr(cond, context));
    let lhs_texp = try!(parse_expr(lhs, context));
    let rhs_texp = try!(parse_expr(rhs, context));
    let (cond, cond_ety) = try!(unwrap_value_expr(cond_texp, context));
    let (lhs, lhs_ety) = try!(unwrap_value_expr(lhs_texp, context));
    let (rhs, rhs_ety) = try!(unwrap_value_expr(rhs_texp, context));
    let cond_cast = match ImplicitConversion::find(&cond_ety, &ir::Type::bool().to_rvalue()) {
        Ok(cast) => cast,
        Err(()) => return Err(TyperError::TernaryConditionRequiresBoolean(context.ir_type_to_error_type(&cond_ety.0))),
    };
    let cond = cond_cast.apply(cond);
    let ExpressionType(lhs_ty, _) = lhs_ety;
    let ExpressionType(rhs_ty, _) = rhs_ety;
    let final_type = if lhs_ty == rhs_ty {
        lhs_ty.to_rvalue()
    } else {
        assert!(false, "{:?} {:?}", lhs_ty, rhs_ty);
        return Err(TyperError::TernaryArmsMustHaveSameType(context.ir_type_to_error_type(&lhs_ty),
                                                           context.ir_type_to_error_type(&rhs_ty)));
    };
    Ok(TypedExpression::Value(ir::Expression::TernaryConditional(Box::new(cond),
                                                                 Box::new(lhs),
                                                                 Box::new(rhs)),
                              final_type))
}

fn parse_expr_unchecked(ast: &ast::Expression,
                        context: &ExpressionContext)
                        -> Result<TypedExpression, TyperError> {
    match ast {
        &ast::Expression::Literal(ref lit) => parse_literal(lit),
        &ast::Expression::Variable(ref s) => Ok(try!(context.find_variable(s))),
        &ast::Expression::UnaryOperation(ref op, ref expr) => parse_expr_unaryop(op, expr, context),
        &ast::Expression::BinaryOperation(ref op, ref lhs, ref rhs) => {
            parse_expr_binop(op, lhs, rhs, context)
        }
        &ast::Expression::TernaryConditional(ref cond, ref lhs, ref rhs) => {
            parse_expr_ternary(cond, lhs, rhs, context)
        }
        &ast::Expression::ArraySubscript(ref array, ref subscript) => {
            let array_texp = try!(parse_expr(array, context));
            let subscript_texp = try!(parse_expr(subscript, context));
            let (array_ir, array_ty) = match array_texp {
                TypedExpression::Value(array_ir, array_ty) => (array_ir, array_ty),
                _ => return Err(TyperError::ArrayIndexingNonArrayType),
            };
            let (subscript_ir, subscript_ty) = match subscript_texp {
                TypedExpression::Value(subscript_ir, subscript_ty) => (subscript_ir, subscript_ty),
                _ => return Err(TyperError::ArrayIndexingNonArrayType),
            };
            let ExpressionType(ir::Type(array_tyl, _), _) = array_ty;
            let indexed_type = match array_tyl {
                ir::TypeLayout::Array(ref element, _) => {
                    ir::Type::from_layout(*element.clone()).to_lvalue()
                }
                ir::TypeLayout::Object(ir::ObjectType::Buffer(data_type)) => {
                    // Todo: const
                    ir::Type::from_data(data_type).to_lvalue()
                }
                ir::TypeLayout::Object(ir::ObjectType::RWBuffer(data_type)) => {
                    ir::Type::from_data(data_type).to_lvalue()
                }
                ir::TypeLayout::Object(ir::ObjectType::StructuredBuffer(structured_type)) => {
                    // Todo: const
                    ir::Type::from_structured(structured_type).to_lvalue()
                }
                ir::TypeLayout::Object(ir::ObjectType::RWStructuredBuffer(structured_type)) => {
                    ir::Type::from_structured(structured_type).to_lvalue()
                }
                _ => return Err(TyperError::ArrayIndexingNonArrayType),
            };
            let cast_to_int_result = ImplicitConversion::find(&subscript_ty,
                                                              &ir::Type::int().to_rvalue());
            let subscript_final = match cast_to_int_result {
                Err(_) => return Err(TyperError::ArraySubscriptIndexNotInteger),
                Ok(cast) => cast.apply(subscript_ir),
            };
            Ok(TypedExpression::Value(ir::Expression::ArraySubscript(Box::new(array_ir),
                                                                     Box::new(subscript_final)),
                                      indexed_type))
        }
        &ast::Expression::Member(ref composite, ref member) => {
            let composite_texp = try!(parse_expr(composite, context));
            let composite_pt = context.typed_expression_to_error_type(&composite_texp);
            let (composite_ir, composite_ty) = match composite_texp {
                TypedExpression::Value(composite_ir, composite_type) => {
                    (composite_ir, composite_type)
                }
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_pt)),
            };
            let ExpressionType(ir::Type(composite_tyl, _), vt) = composite_ty;
            match &composite_tyl {
                &ir::TypeLayout::Struct(ref id) => {
                    match context.find_struct_member(id, member) {
                        Ok(ty) => Ok(TypedExpression::Value(ir::Expression::Member(Box::new(composite_ir), member.clone()), ty.to_lvalue())),
                        Err(err) => Err(err),
                    }
                }
                &ir::TypeLayout::Vector(ref scalar, ref x) => {
                    let mut swizzle_slots = Vec::with_capacity(member.len());
                    for c in member.chars() {
                        swizzle_slots.push(match c {
                            'x' | 'r' if *x >= 1 => ir::SwizzleSlot::X,
                            'y' | 'g' if *x >= 2 => ir::SwizzleSlot::Y,
                            'z' | 'b' if *x >= 3 => ir::SwizzleSlot::Z,
                            'w' | 'a' if *x >= 4 => ir::SwizzleSlot::W,
                            _ => {
                                return Err(TyperError::InvalidSwizzle(composite_pt, member.clone()))
                            }
                        });
                    }
                    let ety = ir::ExpressionType(ir::Type::from_layout(// Lets say single element swizzles go to scalars
                                                                       // Technically they might be going to 1 element vectors
                                                                       // that then get downcasted
                                                                       // But it's hard to tell as scalars + single element vectors
                                                                       // have the same overload precedence
                                                                       if swizzle_slots.len() ==
                                                                          1 {
                                                     ir::TypeLayout::Scalar(scalar.clone())
                                                 } else {
                                                     ir::TypeLayout::Vector(scalar.clone(), swizzle_slots.len() as u32)
                                                 }),
                                                 vt);
                    Ok(TypedExpression::Value(ir::Expression::Swizzle(Box::new(composite_ir),
                                                                      swizzle_slots),
                                              ety))
                }
                &ir::TypeLayout::Object(ref object_type) => {
                    match intrinsics::get_method(object_type, &member) {
                        Ok(intrinsics::MethodDefinition(object_type, name, method_overloads)) => {
                            let overloads = method_overloads.iter().map(|&(ref return_type, ref param_types, ref factory)| {
                                FunctionOverload(FunctionName::Intrinsic(factory.clone()), return_type.clone(), param_types.clone())
                            }).collect::<Vec<_>>();
                            Ok(
                                TypedExpression::Method(UnresolvedMethod(
                                    name,
                                    ir::Type::from_object(object_type),
                                    overloads,
                                    composite_ir
                                ))
                            )
                        }
                        Err(()) => Err(TyperError::UnknownTypeMember(composite_pt, member.clone())),
                    }
                }
                // Todo: Matrix components + Object members
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_pt)),
            }
        }
        &ast::Expression::Call(ref func, ref params) => {
            let func_texp = try!(parse_expr(func, context));
            let mut params_ir: Vec<ir::Expression> = vec![];
            let mut params_types: Vec<ExpressionType> = vec![];
            for param in params {
                let expr_texp = try!(parse_expr(param, context));
                let (expr_ir, expr_ty) = match expr_texp {
                    TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                    texp => return Err(TyperError::FunctionPassedToAnotherFunction(
                        context.typed_expression_to_error_type(&func_texp),
                        context.typed_expression_to_error_type(&texp)
                    )),
                };
                params_ir.push(expr_ir);
                params_types.push(expr_ty);
            }
            match func_texp {
                TypedExpression::Function(unresolved) => {
                    write_function(unresolved, &params_types, params_ir)
                }
                TypedExpression::Method(unresolved) => {
                    write_method(unresolved, &params_types, params_ir)
                }
                _ => return Err(TyperError::CallOnNonFunction),
            }
        }
        &ast::Expression::NumericConstructor(ref dtyl, ref params) => {
            let datalayout = try!(parse_datalayout(dtyl));
            let target_scalar = datalayout.to_scalar();
            let mut slots: Vec<ir::ConstructorSlot> = vec![];
            let mut total_arity = 0;
            for param in params {
                let expr_texp = try!(parse_expr(param, context));
                let (expr_base, ety) = match expr_texp {
                    TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                    _ => return Err(TyperError::FunctionNotCalled),
                };
                let &ir::ExpressionType(ir::Type(ref expr_tyl, _), _) = &ety;
                let arity = expr_tyl.get_num_elements();
                total_arity = total_arity + arity;
                let s = target_scalar.clone();
                let target_tyl = match *expr_tyl {
                    ir::TypeLayout::Scalar(_) => ir::TypeLayout::Scalar(s),
                    ir::TypeLayout::Vector(_, ref x) => ir::TypeLayout::Vector(s, *x),
                    ir::TypeLayout::Matrix(_, ref x, ref y) => ir::TypeLayout::Matrix(s, *x, *y),
                    _ => return Err(TyperError::WrongTypeInConstructor),
                };
                let target_type = ir::Type::from_layout(target_tyl).to_rvalue();
                let cast = match ImplicitConversion::find(&ety, &target_type) {
                    Ok(cast) => cast,
                    Err(()) => return Err(TyperError::WrongTypeInConstructor),
                };
                let expr = cast.apply(expr_base);
                slots.push(ir::ConstructorSlot {
                    arity: arity,
                    expr: expr,
                });
            }
            let type_layout = ir::TypeLayout::from_data(datalayout.clone());
            let expected_layout = type_layout.get_num_elements();
            let ty = ir::Type::from_layout(type_layout).to_rvalue();
            if total_arity == expected_layout {
                let cons = ir::Expression::NumericConstructor(datalayout, slots);
                Ok(TypedExpression::Value(cons, ty))
            } else {
                Err(TyperError::NumericConstructorWrongArgumentCount)
            }
        }
        &ast::Expression::Cast(ref ty, ref expr) => {
            let expr_texp = try!(parse_expr(expr, context));
            let expr_pt = context.typed_expression_to_error_type(&expr_texp);
            match expr_texp {
                TypedExpression::Value(expr_ir, _) => {
                    let ir_type = try!(parse_type(ty, context.as_struct_id_finder()));
                    Ok(TypedExpression::Value(ir::Expression::Cast(ir_type.clone(),
                                                                   Box::new(expr_ir)),
                                              ir_type.to_rvalue()))
                }
                _ => Err(TyperError::InvalidCast(expr_pt, ErrorType::Value(ty.clone()))),
            }
        }
    }
}

fn parse_expr(expr: &ast::Expression,
              context: &ExpressionContext)
              -> Result<TypedExpression, TyperError> {
    let texp = try!(parse_expr_unchecked(expr, context));
    match texp {
        #[cfg(debug_assertions)]
        TypedExpression::Value(ref expr, ref ty_expected) => {
            let ty_res = ir::TypeParser::get_expression_type(&expr, context.as_type_context());
            let ty = ty_res.expect("type unknown");
            assert!(ty == *ty_expected,
                    "{:?} == {:?}: {:?}",
                    ty,
                    *ty_expected,
                    expr);
        }
        _ => {}
    };
    Ok(texp)
}

fn parse_expr_value_only(expr: &ast::Expression,
                         context: &ExpressionContext)
                         -> Result<ir::Expression, TyperError> {
    let expr_ir = try!(parse_expr(expr, context));
    match expr_ir {
        TypedExpression::Value(expr, _) => Ok(expr),
        TypedExpression::Function(_) => Err(TyperError::FunctionNotCalled),
        TypedExpression::Method(_) => Err(TyperError::FunctionNotCalled),
    }
}

fn apply_variable_bind(ty: ir::Type, bind: &ast::VariableBind) -> Result<ir::Type, TyperError> {
    match *bind {
        ast::VariableBind::Array(ref dim) => {
            let ir::Type(layout, modifiers) = ty;

            // Todo: constant expressions
            let constant_dim = match **dim {
                ast::Expression::Literal(ast::Literal::UntypedInt(i)) => i,
                ast::Expression::Literal(ast::Literal::Int(i)) => i,
                ast::Expression::Literal(ast::Literal::UInt(i)) => i,
                _ => {
                    return Err(TyperError::ArrayDimensionsMustBeConstantExpression((**dim).clone()))
                }
            };

            Ok(ir::Type(ir::TypeLayout::Array(Box::new(layout), constant_dim),
                        modifiers))
        }
        ast::VariableBind::Normal => Ok(ty),
    }
}

fn parse_vardef(ast: &ast::VarDef,
                context: ScopeContext)
                -> Result<(Vec<ir::VarDef>, ScopeContext), TyperError> {
    let var_type = try!(parse_localtype(&ast.local_type, &context));
    let rvalue = var_type.0.clone().to_rvalue();

    let mut context = context;
    let mut vardefs = vec![];
    for local_variable in &ast.defs {
        let name = &local_variable.name;
        let assignment = &local_variable.assignment;

        let assign_ir = match *assignment {
            Some(ref expr) => {
                match try!(parse_expr(expr, &context)) {
                    TypedExpression::Value(expr_ir, expt_ty) => {
                        match ImplicitConversion::find(&expt_ty, &rvalue) {
                            Ok(rhs_cast) => Some(rhs_cast.apply(expr_ir)),
                            Err(()) => return Err(TyperError::WrongTypeInInitExpression),
                        }
                    }
                    _ => return Err(TyperError::FunctionTypeInInitExpression),
                }
            }
            None => None,
        };
        let var_name = name.clone();

        let ir::LocalType(lty, ls, interp) = var_type.clone();
        let bind = &local_variable.bind;
        let lv_type = ir::LocalType(try!(apply_variable_bind(lty, bind)), ls, interp);

        let var_id = try!(context.insert_variable(var_name.clone(), lv_type.0.clone()));
        vardefs.push(ir::VarDef {
            id: var_id,
            local_type: lv_type,
            assignment: assign_ir,
        });
    }

    Ok((vardefs, context))
}

fn parse_for_init(ast: &ast::InitStatement,
                  context: ScopeContext)
                  -> Result<(ir::ForInit, ScopeContext), TyperError> {
    match ast {
        &ast::InitStatement::Expression(ref expr) => {
            let expr_ir = match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => expr_ir,
                _ => return Err(TyperError::FunctionNotCalled),
            };
            Ok((ir::ForInit::Expression(expr_ir), context))
        }
        &ast::InitStatement::Declaration(ref vd) => {
            let (vd_ir, context) = try!(parse_vardef(vd, context));
            Ok((ir::ForInit::Definitions(vd_ir), context))
        }
    }
}

/// Parse the statement inside a block caused by an statement (if, for, while, etc),
/// with the intention of merging the scope created by the outer statement and the
/// scope of it's inner statement. This is to force a scope on single statements
/// inside the outer statement and cause a name conflict on using the same variable
/// in the inner statement as in the init expression of a for loop
///
/// A block statement will be parsed as if it doesn't create a new scope
///
/// Any other statement will be parsed normally (with the provided ScopeContext),
/// meaning any declarations it makes will end up in the outer statements scoped
/// declarations.
///
/// The given context is the newly created scoped context for the outer statement.
/// block_context is consumed and turned into a declarations list because all uses
/// of this execute it on the inner statement as the last operation in parsing a
/// loop
fn parse_scopeblock(ast: &ast::Statement,
                    block_context: ScopeContext)
                    -> Result<ir::ScopeBlock, TyperError> {
    match *ast {
        ast::Statement::Block(ref statement_vec) => {
            let (statements, block_context) = try!(parse_statement_vec(statement_vec,
                                                                       block_context));
            Ok(ir::ScopeBlock(statements, block_context.destruct()))
        }
        _ => {
            let (ir_statements, block_context) = try!(parse_statement(ast, block_context));
            Ok(ir::ScopeBlock(ir_statements, block_context.destruct()))
        }
    }
}

fn parse_statement(ast: &ast::Statement,
                   context: ScopeContext)
                   -> Result<(Vec<ir::Statement>, ScopeContext), TyperError> {
    match ast {
        &ast::Statement::Empty => Ok((vec![], context)),
        &ast::Statement::Expression(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => {
                    Ok((vec![ir::Statement::Expression(expr_ir)], context))
                }
                _ => return Err(TyperError::FunctionNotCalled),
            }
        }
        &ast::Statement::Var(ref vd) => {
            let (vd_ir, context) = try!(parse_vardef(vd, context));
            let vars = vd_ir.into_iter().map(|v| ir::Statement::Var(v)).collect::<Vec<_>>();
            Ok((vars, context))
        }
        &ast::Statement::Block(ref statement_vec) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (statements, scoped_context) = try!(parse_statement_vec(statement_vec,
                                                                        scoped_context));
            let decls = scoped_context.destruct();
            Ok((vec![ir::Statement::Block(ir::ScopeBlock(statements, decls))],
                context))
        }
        &ast::Statement::If(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let cond_ir = try!(parse_expr_value_only(cond, &scoped_context));
            let scope_block = try!(parse_scopeblock(statement, scoped_context));
            Ok((vec![ir::Statement::If(cond_ir, scope_block)], context))
        }
        &ast::Statement::IfElse(ref cond, ref true_statement, ref false_statement) => {
            let cond_ir = try!(parse_expr_value_only(cond, &context));
            let scoped_context = ScopeContext::from_scope(&context);
            let scope_block = try!(parse_scopeblock(true_statement, scoped_context));
            let scoped_context = ScopeContext::from_scope(&context);
            let else_block = try!(parse_scopeblock(false_statement, scoped_context));
            Ok((vec![ir::Statement::IfElse(cond_ir, scope_block, else_block)],
                context))
        }
        &ast::Statement::For(ref init, ref cond, ref iter, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (init_ir, scoped_context) = try!(parse_for_init(init, scoped_context));
            let cond_ir = try!(parse_expr_value_only(cond, &scoped_context));
            let iter_ir = try!(parse_expr_value_only(iter, &scoped_context));
            let scope_block = try!(parse_scopeblock(statement, scoped_context));
            Ok((vec![ir::Statement::For(init_ir, cond_ir, iter_ir, scope_block)],
                context))
        }
        &ast::Statement::While(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let cond_ir = try!(parse_expr_value_only(cond, &scoped_context));
            let scope_block = try!(parse_scopeblock(statement, scoped_context));
            Ok((vec![ir::Statement::While(cond_ir, scope_block)], context))
        }
        &ast::Statement::Return(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, expr_ty) => {
                    match ImplicitConversion::find(&expr_ty,
                                                   &context.get_return_type().to_rvalue()) {
                        Ok(rhs_cast) => {
                            Ok((vec![ir::Statement::Return(rhs_cast.apply(expr_ir))],
                                context))
                        }
                        Err(()) => return Err(TyperError::WrongTypeInReturnStatement),
                    }
                }
                _ => return Err(TyperError::FunctionNotCalled),
            }
        }
    }
}

fn parse_statement_vec(ast: &[ast::Statement],
                       context: ScopeContext)
                       -> Result<(Vec<ir::Statement>, ScopeContext), TyperError> {
    let mut context = context;
    let mut body_ir = vec![];
    for statement_ast in ast {
        let (mut statement_ir_vec, next_context) = try!(parse_statement(&statement_ast, context));
        body_ir.append(&mut statement_ir_vec);
        context = next_context;
    }
    Ok((body_ir, context))
}

fn parse_rootdefinition_struct(sd: &ast::StructDefinition,
                               mut context: GlobalContext)
                               -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let mut members = vec![];
    let mut member_map = HashMap::new();
    for ast_member in &sd.members {
        let name = ast_member.name.clone();
        let typename = try!(parse_type(&ast_member.typename, &context));
        member_map.insert(name.clone(), typename.clone());
        members.push(ir::StructMember {
            name: name,
            typename: typename,
        });
    }
    let name = &sd.name;
    match context.insert_struct(name, member_map) {
        Some(id) => {
            let struct_def = ir::StructDefinition {
                id: id,
                members: members,
            };
            Ok((ir::RootDefinition::Struct(struct_def), context))
        }
        None => Err(TyperError::StructAlreadyDefined(name.clone())),
    }
}

fn parse_rootdefinition_constantbuffer
    (cb: &ast::ConstantBuffer,
     mut context: GlobalContext)
     -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let cb_name = cb.name.clone();
    let mut members = vec![];
    let mut members_map = HashMap::new();
    for member in &cb.members {
        let var_name = member.name.clone();
        let var_type = try!(parse_type(&member.typename, &context));
        let var_offset = member.offset.clone();
        members_map.insert(var_name.clone(), var_type.clone());
        members.push(ir::ConstantVariable {
            name: var_name,
            typename: var_type,
            offset: var_offset,
        });
    }
    let id = match context.insert_cbuffer(&cb_name, members_map) {
        Some(id) => id,
        None => return Err(TyperError::ConstantBufferAlreadyDefined(cb_name.clone())),
    };
    let cb_ir = ir::ConstantBuffer {
        id: id,
        members: members,
    };
    match cb.slot {
        Some(ast::ConstantSlot(slot)) => {
            context.global_slots_constants.push((slot, cb_ir.id.clone()));;
        }
        None => {}
    }
    Ok((ir::RootDefinition::ConstantBuffer(cb_ir), context))
}

fn parse_rootdefinition_globalvariable
    (gv: &ast::GlobalVariable,
     mut context: GlobalContext)
     -> Result<(Vec<ir::RootDefinition>, GlobalContext), TyperError> {

    let var_type = try!(parse_globaltype(&gv.global_type, &context));

    let mut defs = vec![];

    for def in &gv.defs {

        // Resolve type
        let ir::GlobalType(lty, gs, interp) = var_type.clone();
        let bind = &def.bind;
        let gv_type = ir::GlobalType(try!(apply_variable_bind(lty, bind)), gs, interp);

        // Insert variable
        let var_name = def.name.clone();
        let input_type = gv_type.0.clone();
        let var_id = try!(context.insert_global(var_name.clone(), input_type.clone()));

        let var_assign = match &def.assignment {
            &Some(ref assign) => {
                let (uncasted, ty) = match try!(parse_expr(assign, &context)) {
                    TypedExpression::Value(expr, ty) => (expr, ty),
                    TypedExpression::Function(_) => return Err(TyperError::FunctionNotCalled),
                    TypedExpression::Method(_) => return Err(TyperError::FunctionNotCalled),
                };
                // Todo: review if we want to purge modifiers for rhs of assignment
                let cast_to = ir::Type(input_type.0.clone(), ir::TypeModifier::default())
                                  .to_rvalue();
                let cast = match ImplicitConversion::find(&ty, &cast_to) {
                    Ok(cast) => cast,
                    Err(()) => return Err(TyperError::WrongTypeInInitExpression),
                };
                let casted = cast.apply(uncasted);
                Some(casted)
            }
            &None => None,
        };
        let gv_ir = ir::GlobalVariable {
            id: var_id,
            global_type: gv_type,
            assignment: var_assign,
        };
        let entry = ir::GlobalEntry {
            id: var_id,
            ty: gv_ir.global_type.clone(),
        };
        match def.slot {
            Some(ast::GlobalSlot::ReadSlot(slot)) => {
                context.global_slots_r.push((slot, entry));
            }
            Some(ast::GlobalSlot::ReadWriteSlot(slot)) => {
                context.global_slots_rw.push((slot, entry));
            }
            None => {}
        }

        defs.push(ir::RootDefinition::GlobalVariable(gv_ir));
    }

    Ok((defs, context))
}

fn parse_rootdefinition_function(fd: &ast::FunctionDefinition,
                                 mut context: GlobalContext)
                                 -> Result<(ir::RootDefinition, GlobalContext), TyperError> {

    let return_type = try!(parse_type(&fd.returntype, &context));
    // Set the return type of the current function (for return statement parsing)
    assert_eq!(context.current_return_type, None);
    context.current_return_type = Some(return_type.clone());

    let mut scoped_context = ScopeContext::from_global(&context);
    let func_params = {
        let mut vec = vec![];
        for param in &fd.params {
            let var_type = try!(parse_paramtype(&param.param_type, &context));
            let var_id = try!(scoped_context.insert_variable(param.name.clone(),
                                                             var_type.0.clone()));
            vec.push(ir::FunctionParam {
                id: var_id,
                param_type: var_type,
            });
        }
        vec
    };
    let (body_ir, scoped_context) = try!(parse_statement_vec(&fd.body, scoped_context));
    let decls = scoped_context.destruct();

    // Unset the return type for the current function
    assert!(context.current_return_type != None);
    context.current_return_type = None;

    let fd_ir = ir::FunctionDefinition {
        id: context.make_function_id(),
        returntype: return_type,
        params: func_params,
        scope_block: ir::ScopeBlock(body_ir, decls),
        attributes: fd.attributes.clone(),
    };
    let func_type = FunctionOverload(FunctionName::User(fd_ir.id),
                                     fd_ir.returntype.clone(),
                                     fd_ir.params.iter().map(|p| p.param_type.clone()).collect());
    try!(context.insert_function(fd.name.clone(), func_type));
    Ok((ir::RootDefinition::Function(fd_ir), context))
}

fn parse_rootdefinition_kernel(fd: &ast::FunctionDefinition,
                               context: GlobalContext)
                               -> Result<(ir::RootDefinition, GlobalContext), TyperError> {

    let mut scoped_context = ScopeContext::from_global(&context);
    let kernel_params = {
        let mut vec = vec![];
        for param in &fd.params {
            let var_type = try!(parse_paramtype(&param.param_type, &context));
            let var_id = try!(scoped_context.insert_variable(param.name.clone(),
                                                             var_type.0.clone()));
            vec.push(ir::KernelParam(var_id,
                match &param.semantic {
                    &Some(ast::Semantic::DispatchThreadId) => ir::KernelSemantic::DispatchThreadId,
                    &Some(_) => return Err(TyperError::KernelHasParamWithBadSemantic(param.clone())),
                    &None => return Err(TyperError::KernelHasParamWithoutSemantic(param.clone())),
                }
            ));
        }
        vec
    };
    let (body_ir, scoped_context) = try!(parse_statement_vec(&fd.body, scoped_context));
    let decls = scoped_context.destruct();

    fn find_dispatch_dimensions(attributes: &[ast::FunctionAttribute])
                                -> Result<ir::Dimension, TyperError> {
        for attribute in attributes {
            match attribute {
                &ast::FunctionAttribute::NumThreads(x, y, z) => return Ok(ir::Dimension(x, y, z)),
            };
        }
        Err(TyperError::KernelHasNoDispatchDimensions)
    }
    let kernel = ir::Kernel {
        group_dimensions: try!(find_dispatch_dimensions(&fd.attributes[..])),
        params: kernel_params,
        scope_block: ir::ScopeBlock(body_ir, decls),
    };
    Ok((ir::RootDefinition::Kernel(kernel), context))
}

fn parse_rootdefinition(ast: &ast::RootDefinition,
                        context: GlobalContext,
                        entry_point: &str)
                        -> Result<(Vec<ir::RootDefinition>, GlobalContext), TyperError> {
    match ast {
        &ast::RootDefinition::Struct(ref sd) => {
            let (def, context) = try!(parse_rootdefinition_struct(sd, context));
            Ok((vec![def], context))
        }
        &ast::RootDefinition::SamplerState => unimplemented!(),
        &ast::RootDefinition::ConstantBuffer(ref cb) => {
            let (def, context) = try!(parse_rootdefinition_constantbuffer(cb, context));
            Ok((vec![def], context))
        }
        &ast::RootDefinition::GlobalVariable(ref gv) => {
            parse_rootdefinition_globalvariable(gv, context)
        }
        &ast::RootDefinition::Function(ref fd) if fd.name == entry_point => {
            let (def, context) = try!(parse_rootdefinition_kernel(fd, context));
            Ok((vec![def], context))
        }
        &ast::RootDefinition::Function(ref fd) => {
            let (def, context) = try!(parse_rootdefinition_function(fd, context));
            Ok((vec![def], context))
        }
    }
}

pub fn typeparse(ast: &ast::Module) -> Result<ir::Module, TyperError> {
    use slp_lang_hir::globals_analysis::GlobalUsage;

    let mut context = GlobalContext::new();

    let mut root_definitions = vec![];

    for def in &ast.root_definitions {
        let (mut def_ir, next_context) = try!(parse_rootdefinition(&def,
                                                                   context,
                                                                   &ast.entry_point.clone()));
        root_definitions.append(&mut def_ir);
        context = next_context;
    }

    let analysis = GlobalUsage::analyse(&root_definitions);

    let root_definitions = root_definitions.into_iter()
                                           .filter(|def| {
                                               match *def {
                                                   ir::RootDefinition::GlobalVariable(ref gv) => {
                                                       analysis.kernel.globals.contains(&gv.id)
                                                   }
                                                   ir::RootDefinition::ConstantBuffer(ref cb) => {
                                                       analysis.kernel.cbuffers.contains(&cb.id)
                                                   }
                                                   ir::RootDefinition::Function(ref func) => {
                                                       analysis.kernel.functions.contains(&func.id)
                                                   }
                                                   _ => true,
                                               }
                                           })
                                           .collect::<Vec<_>>();

    // Gather remaining global declaration names
    let global_declarations =
        root_definitions.iter().fold(ir::GlobalDeclarations {
                                         functions: HashMap::new(),
                                         globals: HashMap::new(),
                                         structs: HashMap::new(),
                                         constants: HashMap::new(),
                                     },
                                     |mut map, def| {
                                         match *def {
                                             ir::RootDefinition::Struct(ref sd) => {
                                                 match context.types.struct_names.get(&sd.id) {
                                                     Some(name) => {
                                                         map.structs.insert(sd.id, name.clone());
                                                     }
                                                     None => {
                                                         panic!("struct name does not exist");
                                                     }
                                                 }
                                             }
                                             ir::RootDefinition::ConstantBuffer(ref cb) => {
                                                 match context.types.cbuffer_names.get(&cb.id) {
                                                     Some(name) => {
                                                         map.constants.insert(cb.id, name.clone());
                                                     }
                                                     None => {
                                                         panic!("constant buffer name does not \
                                                                 exist");
                                                     }
                                                 }
                                             }
                                             ir::RootDefinition::GlobalVariable(ref gv) => {
                                                 match context.global_names.get(&gv.id) {
                                                     Some(name) => {
                                                         map.globals.insert(gv.id, name.clone());
                                                     }
                                                     None => {
                                                         panic!("global variable name does not \
                                                                 exist");
                                                     }
                                                 }
                                             }
                                             ir::RootDefinition::Function(ref func) => {
                                                 match context.function_names.get(&func.id) {
                                                     Some(name) => {
                                                         map.functions
                                                            .insert(func.id, name.clone());
                                                     }
                                                     None => {
                                                         panic!("function name does not exist");
                                                     }
                                                 }
                                             }
                                             _ => {}
                                         }
                                         map
                                     });

    // Resolve used globals into SRV list
    let mut global_table_r = HashMap::new();
    for (slot, entry) in context.global_slots_r {
        if global_declarations.globals.contains_key(&entry.id) {
            let error_id = entry.id.clone();
            match global_table_r.insert(slot, entry) {
                Some(currently_used_by) => {
                    return Err(TyperError::ReadResourceSlotAlreadyUsed(currently_used_by.id
                                                                                        .clone(),
                                                                       error_id))
                }
                None => {}
            }
        }
    }

    // Resolve used globals into UAV list
    let mut global_table_rw = HashMap::new();
    for (slot, entry) in context.global_slots_rw {
        if global_declarations.globals.contains_key(&entry.id) {
            let error_id = entry.id.clone();
            match global_table_rw.insert(slot, entry) {
                Some(currently_used_by) => return Err(TyperError::ReadWriteResourceSlotAlreadyUsed(currently_used_by.id.clone(), error_id)),
                None => {}
            }
        }
    }

    // Resolve used constant buffers into constabt buffer list
    let mut global_table_constants = HashMap::new();
    for (slot, cb_id) in context.global_slots_constants {
        if global_declarations.constants.contains_key(&cb_id) {
            let error_id = cb_id.clone();
            match global_table_constants.insert(slot, cb_id) {
                Some(currently_used_by) => {
                    return Err(TyperError::ConstantSlotAlreadyUsed(currently_used_by.clone(),
                                                                   error_id))
                }
                None => {}
            }
        }
    }

    // Make the table describing all global bindings
    let global_table = ir::GlobalTable {
        r_resources: global_table_r,
        rw_resources: global_table_rw,
        samplers: HashMap::new(),
        constants: global_table_constants,
    };

    let ir = ir::Module {
        entry_point: ast.entry_point.clone(),
        global_table: global_table,
        root_definitions: root_definitions,
        global_declarations: global_declarations,
    };

    // Ensure we have one kernel function
    let mut has_kernel = false;
    for root_def in &ir.root_definitions {
        match root_def {
            &ir::RootDefinition::Kernel(_) => {
                if has_kernel {
                    return Err(TyperError::KernelDefinedMultipleTimes);
                } else {
                    has_kernel = true;
                }
            }
            _ => {}
        }
    }
    if !has_kernel {
        return Err(TyperError::KernelNotDefined);
    }

    Ok(ir)
}

#[test]
fn test_typeparse() {

    use slp_shared::Located;

    let module = ast::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            ast::RootDefinition::GlobalVariable(ast::GlobalVariable {
                global_type: ast::Type::from_object(ast::ObjectType::Buffer(ast::DataType(ast::DataLayout::Scalar(ast::ScalarType::Int), ast::TypeModifier::default()))).into(),
                defs: vec![ast::GlobalVariableName {
                    name: "g_myInBuffer".to_string(),
                    bind: ast::VariableBind::Normal,
                    slot: Some(ast::GlobalSlot::ReadSlot(0)),
                    assignment: None,
                }],
            }),
            ast::RootDefinition::GlobalVariable(ast::GlobalVariable {
                global_type: ast::GlobalType(ast::Type(ast::TypeLayout::from_scalar(ast::ScalarType::Int), ast::TypeModifier { is_const: true, .. ast::TypeModifier::default() }), ast::GlobalStorage::Static, None),
                defs: vec![ast::GlobalVariableName {
                    name: "g_myFour".to_string(),
                    bind: ast::VariableBind::Normal,
                    slot: None,
                    assignment: Some(Located::none(ast::Expression::Literal(ast::Literal::UntypedInt(4)))),
                }],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::void(),
                params: vec![ast::FunctionParam { name: "x".to_string(), param_type: ast::Type::uint().into(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::void(),
                params: vec![ast::FunctionParam { name: "x".to_string(), param_type: ast::Type::float().into(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "outFunc".to_string(),
                returntype: ast::Type::void(),
                params: vec![ast::FunctionParam { name: "x".to_string(), param_type: ast::ParamType(ast::Type::float(), ast::InputModifier::Out, None), semantic: None }],
                body: vec![
                    ast::Statement::Var(ast::VarDef::new("local_static".to_string(), ast::LocalType(ast::Type::uint(), ast::LocalStorage::Static, None), None)),
                    ast::Statement::Expression(Located::loc(1, 1,
                        ast::Expression::BinaryOperation(ast::BinOp::Assignment,
                            Box::new(Located::none(ast::Expression::Variable("x".to_string()))),
                            Box::new(Located::none(ast::Expression::Literal(ast::Literal::Float(1.5f32))))
                        )
                    )),
                ],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: ast::Type::void(),
                params: vec![],
                body: vec![
                    ast::Statement::Empty,
                    ast::Statement::Var(ast::VarDef::new("a".to_string(), ast::Type::uint().into(), None)),
                    ast::Statement::Var(ast::VarDef::new("b".to_string(), ast::Type::uint().into(), None)),
                    ast::Statement::Expression(Located::none(
                        ast::Expression::BinaryOperation(ast::BinOp::Assignment,
                            Box::new(Located::none(ast::Expression::Variable("a".to_string()))),
                            Box::new(Located::none(ast::Expression::Variable("b".to_string())))
                        )
                    )),
                    ast::Statement::If(
                        Located::none(ast::Expression::Variable("b".to_string())),
                        Box::new(ast::Statement::Empty),
                    ),
                    ast::Statement::Expression(Located::none(
                        ast::Expression::BinaryOperation(ast::BinOp::Assignment,
                            Box::new(Located::none(ast::Expression::ArraySubscript(
                                Box::new(Located::none(ast::Expression::Variable("g_myInBuffer".to_string()))),
                                Box::new(Located::none(ast::Expression::Literal(ast::Literal::Int(0))))
                            ))),
                            Box::new(Located::none(ast::Expression::Literal(ast::Literal::Int(4))))
                        )
                    )),
                    ast::Statement::Expression(Located::none(
                        ast::Expression::Call(
                            Box::new(Located::none(ast::Expression::Variable("myFunc".to_string()))),
                            vec![
                                Located::none(ast::Expression::Variable("b".to_string()))
                            ]
                        )
                    )),
                    ast::Statement::Var(ast::VarDef::new("testOut".to_string(), ast::Type::float().into(), None)),
                    ast::Statement::Var(ast::VarDef {
                        local_type: ast::Type::from_layout(ast::TypeLayout::float()).into(),
                        defs: vec![ast::LocalVariableName {
                            name: "x".to_string(),
                            bind: ast::VariableBind::Array(Located::none(ast::Expression::Literal(ast::Literal::UntypedInt(3)))),
                            assignment: None,
                        }]
                    }),
                    ast::Statement::Expression(Located::none(
                        ast::Expression::Call(
                            Box::new(Located::none(ast::Expression::Variable("outFunc".to_string()))),
                            vec![Located::none(ast::Expression::Variable("testOut".to_string()))]
                        )
                    )),
                ],
                attributes: vec![ast::FunctionAttribute::NumThreads(8, 8, 1)],
            }),
        ],
    };
    let res = typeparse(&module);
    assert!(res.is_ok(), "{:?}", res);

    let static_global_test = ast::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            ast::RootDefinition::GlobalVariable(ast::GlobalVariable {
                global_type: ast::GlobalType(ast::Type(ast::TypeLayout::from_scalar(ast::ScalarType::Int), ast::TypeModifier { is_const: true, .. ast::TypeModifier::default() }), ast::GlobalStorage::Static, None),
                defs: vec![ast::GlobalVariableName {
                    name: "g_myFour".to_string(),
                    bind: ast::VariableBind::Normal,
                    slot: None,
                    assignment: Some(Located::none(ast::Expression::Literal(ast::Literal::UntypedInt(4)))),
                }],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: ast::Type::void(),
                params: vec![],
                body: vec![
                    ast::Statement::Expression(Located::none(ast::Expression::Variable("g_myFour".to_string())))
                ],
                attributes: vec![ast::FunctionAttribute::NumThreads(8, 8, 1)],
            }),
        ],
    };
    let static_global_result = typeparse(&static_global_test);
    let static_global_expected = Ok(ir::Module {
        entry_point: "CSMAIN".to_string(),
        global_table: ir::GlobalTable::default(),
        global_declarations: ir::GlobalDeclarations {
            functions: HashMap::new(),
            globals: { let mut map = HashMap::new(); map.insert(ir::GlobalId(0), "g_myFour".to_string()); map },
            structs: HashMap::new(),
            constants: HashMap::new(),
        },
        root_definitions: vec![
            ir::RootDefinition::GlobalVariable(ir::GlobalVariable {
                id: ir::GlobalId(0),
                global_type: ir::GlobalType(ir::Type(ir::TypeLayout::from_scalar(ir::ScalarType::Int), ir::TypeModifier { is_const: true, .. ir::TypeModifier::default() }), ir::GlobalStorage::Static, None),
                assignment: Some(ir::Expression::Cast(ir::Type::int(), Box::new(ir::Expression::Literal(ir::Literal::UntypedInt(4))))),
            }),
            ir::RootDefinition::Kernel(ir::Kernel {
                group_dimensions: ir::Dimension(8, 8, 1),
                params: vec![],
                scope_block: ir::ScopeBlock(vec![
                    ir::Statement::Expression(ir::Expression::Global(ir::GlobalId(0)))
                ], ir::ScopedDeclarations { variables: HashMap::new() }),
            }),
        ],
    });
    assert_eq!(static_global_result, static_global_expected);
}
