
use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use super::ir;
use super::ast;
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

    ConstantSlotAlreadyUsed(String),
    ReadResourceSlotAlreadyUsed(ir::VariableId, ir::VariableId),
    ReadWriteResourceSlotAlreadyUsed(ir::VariableId, ir::VariableId),

    UnknownIdentifier(String),
    UnknownType(ErrorType),

    TypeDoesNotHaveMembers(ErrorType),
    UnknownTypeMember(ErrorType, String),

    ArrayIndexingNonArrayType,
    ArraySubscriptIndexNotInteger,

    CallOnNonFunction,

    FunctionPassedToAnotherFunction(ErrorType, ErrorType),
    FunctionArgumentTypeMismatch(Vec<FunctionOverload>, ParamArray),

    UnaryOperationWrongTypes(ir::UnaryOp, ErrorType),
    BinaryOperationWrongTypes(ir::BinOp, ErrorType, ErrorType),

    InvalidCast(ErrorType, ErrorType),
    FunctionTypeInInitExpression,
    WrongTypeInInitExpression,
    FunctionNotCalled,

    KernelNotDefined,
    KernelDefinedMultipleTimes,
    KernelHasNoDispatchDimensions,
    KernelHasParamWithBadSemantic(ast::FunctionParam),
    KernelHasParamWithoutSemantic(ast::FunctionParam),
}

pub type ReturnType = ir::Type;
pub type ParamType = ir::Type;
pub type ParamArray = Vec<ParamType>;
pub type ClassType = ir::Type;

#[derive(PartialEq, Debug, Clone)]
pub enum FunctionName {
    Intrinsic(intrinsics::IntrinsicFactory),
    User(ir::FunctionId),
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionOverload(pub FunctionName, pub ReturnType, pub ParamArray);

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

            TyperError::ConstantSlotAlreadyUsed(_) => "global constant slot already used",
            TyperError::ReadResourceSlotAlreadyUsed(_, _) => "global resource slot already used",
            TyperError::ReadWriteResourceSlotAlreadyUsed(_, _) => "global writable resource slot already used",

            TyperError::UnknownIdentifier(_) => "unknown identifier",
            TyperError::UnknownType(_) => "unknown type name",

            TyperError::TypeDoesNotHaveMembers(_) => "unknown member (type has no members)",
            TyperError::UnknownTypeMember(_, _) => "unknown member",

            TyperError::ArrayIndexingNonArrayType => "array index applied to non-array type",
            TyperError::ArraySubscriptIndexNotInteger => "array subscripts must be integers",

            TyperError::CallOnNonFunction => "function call applied to non-function type",

            TyperError::FunctionPassedToAnotherFunction(_, _) => "functions can not be passed to other functions",
            TyperError::FunctionArgumentTypeMismatch(_, _) => "wrong parameters given to function",

            TyperError::UnaryOperationWrongTypes(_, _) => "operation does not support the given types",
            TyperError::BinaryOperationWrongTypes(_, _, _) => "operation does not support the given types",

            TyperError::InvalidCast(_, _) => "invalid cast",
            TyperError::FunctionTypeInInitExpression => "function not called",
            TyperError::WrongTypeInInitExpression => "wrong type in variable initialization",
            TyperError::FunctionNotCalled => "function not called",

            TyperError::KernelNotDefined => "entry point not found",
            TyperError::KernelDefinedMultipleTimes => "multiple entry points found",
            TyperError::KernelHasNoDispatchDimensions => "compute kernels require a dispatch dimension",
            TyperError::KernelHasParamWithBadSemantic(_) => "kernel parameter did not have a valid kernel semantic",
            TyperError::KernelHasParamWithoutSemantic(_) => "kernel parameter did not have a kernel semantic",
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
    Value(ir::Expression, ir::Type),
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
    struct_definitions: HashMap<ir::StructId, HashMap<String, ir::Type>>,
    next_free_struct_id: ir::StructId,

    cbuffer_ids: HashMap<String, ir::ConstantBufferId>,
    cbuffer_definitions: HashMap<ir::ConstantBufferId, HashMap<String, ir::Type>>,
    next_free_cbuffer_id: ir::ConstantBufferId,
}

#[derive(PartialEq, Debug, Clone)]
struct GlobalContext {
    functions: HashMap<String, UnresolvedFunction>,
    next_free_function_id: ir::FunctionId,

    types: TypeBlock,
    variables: VariableBlock,
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

trait StructIdFinder {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError>;
}

trait ErrorTypeContext {
    fn ir_type_to_error_type(&self, ty: &ir::Type) -> ErrorType;

    fn typed_expression_to_error_type(&self, texp: &TypedExpression) -> ErrorType;
}

impl UnresolvedFunction {
    pub fn get_name(&self) -> String { self.0.clone() }
}

impl UnresolvedMethod {
    pub fn get_name(&self) -> String { self.0.clone() }
}

impl VariableBlock {
    fn new() -> VariableBlock {
        VariableBlock { variables: HashMap::new(), next_free_variable_id: 0 }
    }

    fn insert_variable(&mut self, name: String, typename: ir::Type, context: &ErrorTypeContext) -> Result<ir::VariableId, TyperError> {
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name, context.ir_type_to_error_type(ty), context.ir_type_to_error_type(&typename)))
        };
        match self.variables.entry(name.clone()) {
            Entry::Occupied(occupied) => Err(TyperError::ValueAlreadyDefined(name, context.ir_type_to_error_type(&occupied.get().0), context.ir_type_to_error_type(&typename))),
            Entry::Vacant(vacant) => {
                let id = self.next_free_variable_id;
                self.next_free_variable_id = self.next_free_variable_id + 1;
                vacant.insert((typename, id)); Ok(id)
            },
        }
    }

    fn has_variable(&self, name: &String) -> Option<&(ir::Type, ir::VariableId)> {
        self.variables.get(name)
    }

    fn find_variable(&self, name: &String, scopes_up: u32) -> Option<TypedExpression> {
        match self.variables.get(name) {
            Some(&(ref ty, ref id)) => return Some(TypedExpression::Value(ir::Expression::Variable(ir::VariableRef(id.clone(), scopes_up)), ty.clone())),
            None => None,
        }
    }

    fn destruct(self) -> HashMap<ir::VariableId, String> {
        self.variables.iter().fold(HashMap::new(),
            |mut map, (name, &(_, ref id))| {
                map.insert(id.clone(), name.clone());
                map
            }
        )
    }
}

impl TypeBlock {
    fn new() -> TypeBlock {
        TypeBlock {
            struct_ids: HashMap::new(),
            struct_definitions: HashMap::new(),
            next_free_struct_id: 0,
            cbuffer_ids: HashMap::new(),
            cbuffer_definitions: HashMap::new(),
            next_free_cbuffer_id: 0,
        }
    }

    fn insert_struct(&mut self, name: &String, members: HashMap<String, ir::Type>) -> Option<ir::StructId> {
        let id = self.next_free_struct_id;
        self.next_free_struct_id = self.next_free_struct_id + 1;
        match (self.struct_ids.entry(name.clone()), self.struct_definitions.entry(id.clone())) {
            (Entry::Vacant(id_entry), Entry::Vacant(def_entry)) => {
                id_entry.insert(id.clone());
                def_entry.insert(members);
                Some(id.clone())
            },
            _ => None,
        }
    }

    fn find_struct_member(&self, id: &ir::StructId, member_name: &String) -> Result<ir::Type, TyperError> {
        match self.struct_definitions.get(id) {
            Some(def) => def.get(member_name).map(|ty| ty.clone()).ok_or(
                TyperError::UnknownTypeMember(
                    self.ir_type_to_error_type(&ir::Type::Structured(ir::StructuredType::Struct(id.clone()))),
                    member_name.clone()
                )
            ),
            None => Err(TyperError::UnknownType(
                self.ir_type_to_error_type(&ir::Type::Structured(ir::StructuredType::Struct(id.clone())))
            )),
        }
    }

    fn insert_cbuffer(&mut self, name: &String, members: HashMap<String, ir::Type>) -> Option<ir::ConstantBufferId> {
        let id = self.next_free_cbuffer_id;
        self.next_free_cbuffer_id = self.next_free_cbuffer_id + 1;
        match (self.cbuffer_ids.entry(name.clone()), self.cbuffer_definitions.entry(id.clone())) {
            (Entry::Vacant(id_entry), Entry::Vacant(def_entry)) => {
                id_entry.insert(id.clone());
                def_entry.insert(members);
                Some(id.clone())
            },
            _ => None,
        }
    }

    fn find_variable(&self, name: &String) -> Option<TypedExpression> {
        for (id, members) in &self.cbuffer_definitions {
            for (member_name, ty) in members {
                if member_name == name {
                    return Some(TypedExpression::Value(
                        ir::Expression::ConstantVariable(id.clone(), name.clone()),
                        ty.clone()
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

    fn invert_structuredtype(&self, ty: &ir::StructuredType) -> ast::StructuredType {
        match *ty {
            ir::StructuredType::Data(ref data_type) => ast::StructuredType::Data(data_type.clone()),
            ir::StructuredType::Struct(ref id) => {
                ast::StructuredType::Custom(match self.get_struct_name(&id) {
                    Some(name) => name,
                    None => "<struct>".to_string(),
                })
            },
        }
    }

    fn invert_objecttype(&self, ty: &ir::ObjectType) -> ast::ObjectType {
        match *ty {
            ir::ObjectType::Buffer(ref data_type) => ast::ObjectType::Buffer(data_type.clone()),
            ir::ObjectType::RWBuffer(ref data_type) => ast::ObjectType::RWBuffer(data_type.clone()),
            ir::ObjectType::ByteAddressBuffer => ast::ObjectType::ByteAddressBuffer,
            ir::ObjectType::RWByteAddressBuffer => ast::ObjectType::RWByteAddressBuffer,
            ir::ObjectType::StructuredBuffer(ref structured_type) => ast::ObjectType::StructuredBuffer(self.invert_structuredtype(structured_type)),
            ir::ObjectType::RWStructuredBuffer(ref structured_type) => ast::ObjectType::RWStructuredBuffer(self.invert_structuredtype(structured_type)),
            ir::ObjectType::AppendStructuredBuffer(ref structured_type) => ast::ObjectType::AppendStructuredBuffer(self.invert_structuredtype(structured_type)),
            ir::ObjectType::ConsumeStructuredBuffer(ref structured_type) => ast::ObjectType::ConsumeStructuredBuffer(self.invert_structuredtype(structured_type)),
            ir::ObjectType::Texture1D(ref data_type) => ast::ObjectType::Texture1D(data_type.clone()),
            ir::ObjectType::Texture1DArray(ref data_type) => ast::ObjectType::Texture1DArray(data_type.clone()),
            ir::ObjectType::Texture2D(ref data_type) => ast::ObjectType::Texture2D(data_type.clone()),
            ir::ObjectType::Texture2DArray(ref data_type) => ast::ObjectType::Texture2DArray(data_type.clone()),
            ir::ObjectType::Texture2DMS(ref data_type) => ast::ObjectType::Texture2DMS(data_type.clone()),
            ir::ObjectType::Texture2DMSArray(ref data_type) => ast::ObjectType::Texture2DMSArray(data_type.clone()),
            ir::ObjectType::Texture3D(ref data_type) => ast::ObjectType::Texture3D(data_type.clone()),
            ir::ObjectType::TextureCube(ref data_type) => ast::ObjectType::TextureCube(data_type.clone()),
            ir::ObjectType::TextureCubeArray(ref data_type) => ast::ObjectType::TextureCubeArray(data_type.clone()),
            ir::ObjectType::RWTexture1D(ref data_type) => ast::ObjectType::RWTexture1D(data_type.clone()),
            ir::ObjectType::RWTexture1DArray(ref data_type) => ast::ObjectType::RWTexture1DArray(data_type.clone()),
            ir::ObjectType::RWTexture2D(ref data_type) => ast::ObjectType::RWTexture2D(data_type.clone()),
            ir::ObjectType::RWTexture2DArray(ref data_type) => ast::ObjectType::RWTexture2DArray(data_type.clone()),
            ir::ObjectType::RWTexture3D(ref data_type) => ast::ObjectType::RWTexture3D(data_type.clone()),
            ir::ObjectType::InputPatch => ast::ObjectType::InputPatch,
            ir::ObjectType::OutputPatch => ast::ObjectType::OutputPatch,
        }
    }

    fn invert_type(&self, ty: &ir::Type) -> ast::Type {
        match *ty {
            ir::Type::Void => ast::Type::Void,
            ir::Type::Structured(ref structured_type) => ast::Type::Structured(self.invert_structuredtype(structured_type)),
            ir::Type::SamplerState => ast::Type::SamplerState,
            ir::Type::Object(ref object_type) => ast::Type::Object(self.invert_objecttype(object_type)),
            ir::Type::Array(ref array_type) => ast::Type::Array(Box::new(self.invert_type(array_type))),
        }
    }

    fn destruct(self) -> (HashMap<ir::StructId, String>, HashMap<ir::ConstantBufferId, String>) {
        (self.struct_ids.iter().fold(HashMap::new(),
            |mut map, (name, id)| {
                map.insert(id.clone(), name.clone());
                map
             }
        ),
        self.cbuffer_ids.iter().fold(HashMap::new(),
            |mut map, (name, id)| {
                map.insert(id.clone(), name.clone());
                map
             }
        ))
    }
}

impl StructIdFinder for TypeBlock {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        self.struct_ids.get(name).map(|id| id.clone()).ok_or(
            TyperError::UnknownType(
                ErrorType::Value(ast::Type::Structured(ast::StructuredType::Custom(name.clone())))
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
            TypedExpression::Value(_, ref ty) => ErrorType::Value(TypeBlock::invert_type(self, ty)),
            TypedExpression::Function(ref unresolved) => ErrorType::Function(unresolved.clone()),
            TypedExpression::Method(ref unresolved) => ErrorType::Method(unresolved.clone()),
        }
    }
}

impl GlobalContext {
    fn new() -> GlobalContext {
        GlobalContext {
            functions: get_intrinsics(),
            next_free_function_id: 0,
            types: TypeBlock::new(),
            variables: VariableBlock::new(),
        }
    }

    pub fn insert_function(&mut self, name: String, function_type: FunctionOverload) -> Result<(), TyperError> {
        // Error if a variable of the same name already exists
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name, self.ir_type_to_error_type(ty), ErrorType::Unknown))
        };
        // Try to add the function
        match self.functions.entry(name.clone()) {
            Entry::Occupied(mut occupied) => {
                // Fail if the overload already exists
                for &FunctionOverload(_, _, ref args) in &occupied.get().1 {
                    if *args == function_type.2 {
                        return Err(TyperError::ValueAlreadyDefined(name, ErrorType::Unknown, ErrorType::Unknown))
                    }
                };
                // Insert a new overload
                occupied.get_mut().1.push(function_type);
                Ok(())
            },
            Entry::Vacant(vacant) => {
                // Insert a new function with one overload
                vacant.insert(UnresolvedFunction(name, vec![function_type])); Ok(())
            },
        }
    }

    fn find_variable_recur(&self, name: &String, scopes_up: u32) -> Result<TypedExpression, TyperError> {
        match self.functions.get(name) {
            Some(tys) => return Ok(TypedExpression::Function(tys.clone())),
            None => { },
        };
        match self.types.find_variable(name) {
            Some(tys) => return Ok(tys),
            None => { },
        };
        match self.variables.find_variable(name, scopes_up) {
            Some(tys) => return Ok(tys),
            None => { },
        };
        Err(TyperError::UnknownIdentifier(name.clone()))
    }

    fn make_function_id(&mut self) -> ir::FunctionId {
        let value = self.next_free_function_id;
        self.next_free_function_id = self.next_free_function_id + 1;
        value
    }

    fn insert_variable(&mut self, name: String, typename: ir::Type) -> Result<ir::VariableId, TyperError> {
        self.variables.insert_variable(name, typename, &self.types)
    }

    fn has_variable(&self, name: &String) -> Option<&(ir::Type, ir::VariableId)> {
        self.variables.has_variable(name)
    }

    #[allow(dead_code)]
    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn insert_struct(&mut self, name: &String, members: HashMap<String, ir::Type>) -> Option<ir::StructId> {
        self.types.insert_struct(name, members)
    }

    fn find_struct_member(&self, id: &ir::StructId, member_name: &String) -> Result<ir::Type, TyperError> {
        self.types.find_struct_member(id, member_name)
    }

    fn insert_cbuffer(&mut self, name: &String, members: HashMap<String, ir::Type>) -> Option<ir::ConstantBufferId> {
        self.types.insert_cbuffer(name, members)
    }

    fn get_type_block(&self) -> &TypeBlock {
        &self.types
    }

    fn destruct(self) -> ir::GlobalDeclarations {
        let (structs, constants) = self.types.destruct();
        ir::GlobalDeclarations {
            variables: self.variables.destruct(),
            functions: self.functions.iter().fold(HashMap::new(),
                |mut map, (_, &UnresolvedFunction(ref name, ref overloads))| {
                    for overload in overloads {
                        match overload.0 {
                            FunctionName::User(id) => { map.insert(id, name.clone()); },
                            _ => { },
                        }
                    }
                    map
                }
            ),
            structs: structs,
            constants: constants,
        }
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
        ScopeContext { parent: Box::new(Context::Scope(parent.clone())), variables: VariableBlock::new() }
    }

    fn from_global(parent: &GlobalContext) -> ScopeContext {
        ScopeContext { parent: Box::new(Context::Global(parent.clone())), variables: VariableBlock::new() }
    }

    fn find_variable_recur(&self, name: &String, scopes_up: u32) -> Result<TypedExpression, TyperError> {
        match self.variables.find_variable(name, scopes_up) {
            Some(texp) => return Ok(texp),
            None => self.parent.find_variable_recur(name, scopes_up + 1),
        }
    }

    fn destruct(self) -> ir::ScopedDeclarations {
        ir::ScopedDeclarations {
            variables: self.variables.destruct()
        }
    }

    fn insert_variable(&mut self, name: String, typename: ir::Type) -> Result<ir::VariableId, TyperError> {
         let type_block = self.parent.get_type_block();
         let variables = &mut self.variables;
        variables.insert_variable(name, typename, type_block)
    }

    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct_member(&self, id: &ir::StructId, member_name: &String) -> Result<ir::Type, TyperError> {
        self.parent.find_struct_member(id, member_name)
    }

    fn get_type_block(&self) -> &TypeBlock {
        self.parent.get_type_block()
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

    fn find_variable_recur(&self, name: &String, scopes_up: u32) -> Result<TypedExpression, TyperError> {
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

    fn find_struct_member(&self, id: &ir::StructId, member_name: &String) -> Result<ir::Type, TyperError> {
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

/// Create a map of all the intrinsic functions we need to parse
fn get_intrinsics() -> HashMap<String, UnresolvedFunction> {

    let funcs = intrinsics::get_intrinsics();

    let mut strmap: HashMap<String, UnresolvedFunction> = HashMap::new();
    for &(ref return_type, ref name, ref params, ref factory) in funcs {
        let overload = FunctionOverload(FunctionName::Intrinsic(factory.clone()), return_type.clone(), params.to_vec());
        match strmap.entry(name.to_string()) {
            Entry::Occupied(mut occupied) => {
                let &mut UnresolvedFunction(_, ref mut overloads) = occupied.get_mut();
                overloads.push(overload);
            },
            Entry::Vacant(vacant) => {
                vacant.insert(UnresolvedFunction(name.to_string(), vec![overload]));
            },
        }
    }
    strmap
}

fn parse_structuredtype(ty: &ast::StructuredType, struct_finder: &StructIdFinder) -> Result<ir::StructuredType, TyperError> {
    Ok(match *ty {
        ast::StructuredType::Data(ref data_type) => ir::StructuredType::Data(data_type.clone()),
        ast::StructuredType::Custom(ref name) => ir::StructuredType::Struct(try!(struct_finder.find_struct_id(name))),
    })
}

fn parse_objecttype(ty: &ast::ObjectType, struct_finder: &StructIdFinder) -> Result<ir::ObjectType, TyperError> {
    Ok(match *ty {
        ast::ObjectType::Buffer(ref data_type) => ir::ObjectType::Buffer(data_type.clone()),
        ast::ObjectType::RWBuffer(ref data_type) => ir::ObjectType::RWBuffer(data_type.clone()),
        ast::ObjectType::ByteAddressBuffer => ir::ObjectType::ByteAddressBuffer,
        ast::ObjectType::RWByteAddressBuffer => ir::ObjectType::RWByteAddressBuffer,
        ast::ObjectType::StructuredBuffer(ref structured_type) => ir::ObjectType::StructuredBuffer(try!(parse_structuredtype(structured_type, struct_finder))),
        ast::ObjectType::RWStructuredBuffer(ref structured_type) => ir::ObjectType::RWStructuredBuffer(try!(parse_structuredtype(structured_type, struct_finder))),
        ast::ObjectType::AppendStructuredBuffer(ref structured_type) => ir::ObjectType::AppendStructuredBuffer(try!(parse_structuredtype(structured_type, struct_finder))),
        ast::ObjectType::ConsumeStructuredBuffer(ref structured_type) => ir::ObjectType::ConsumeStructuredBuffer(try!(parse_structuredtype(structured_type, struct_finder))),
        ast::ObjectType::Texture1D(ref data_type) => ir::ObjectType::Texture1D(data_type.clone()),
        ast::ObjectType::Texture1DArray(ref data_type) => ir::ObjectType::Texture1DArray(data_type.clone()),
        ast::ObjectType::Texture2D(ref data_type) => ir::ObjectType::Texture2D(data_type.clone()),
        ast::ObjectType::Texture2DArray(ref data_type) => ir::ObjectType::Texture2DArray(data_type.clone()),
        ast::ObjectType::Texture2DMS(ref data_type) => ir::ObjectType::Texture2DMS(data_type.clone()),
        ast::ObjectType::Texture2DMSArray(ref data_type) => ir::ObjectType::Texture2DMSArray(data_type.clone()),
        ast::ObjectType::Texture3D(ref data_type) => ir::ObjectType::Texture3D(data_type.clone()),
        ast::ObjectType::TextureCube(ref data_type) => ir::ObjectType::TextureCube(data_type.clone()),
        ast::ObjectType::TextureCubeArray(ref data_type) => ir::ObjectType::TextureCubeArray(data_type.clone()),
        ast::ObjectType::RWTexture1D(ref data_type) => ir::ObjectType::RWTexture1D(data_type.clone()),
        ast::ObjectType::RWTexture1DArray(ref data_type) => ir::ObjectType::RWTexture1DArray(data_type.clone()),
        ast::ObjectType::RWTexture2D(ref data_type) => ir::ObjectType::RWTexture2D(data_type.clone()),
        ast::ObjectType::RWTexture2DArray(ref data_type) => ir::ObjectType::RWTexture2DArray(data_type.clone()),
        ast::ObjectType::RWTexture3D(ref data_type) => ir::ObjectType::RWTexture3D(data_type.clone()),
        ast::ObjectType::InputPatch => ir::ObjectType::InputPatch,
        ast::ObjectType::OutputPatch => ir::ObjectType::OutputPatch,
    })
}

fn parse_type(ty: &ast::Type, struct_finder: &StructIdFinder) -> Result<ir::Type, TyperError> {
    Ok(match *ty {
        ast::Type::Void => ir::Type::Void,
        ast::Type::Structured(ref structured_type) => ir::Type::Structured(try!(parse_structuredtype(structured_type, struct_finder))),
        ast::Type::SamplerState => ir::Type::SamplerState,
        ast::Type::Object(ref object_type) => ir::Type::Object(try!(parse_objecttype(object_type, struct_finder))),
        ast::Type::Array(ref array_type) => ir::Type::Array(Box::new(try!(parse_type(array_type, struct_finder)))),
    })
}

fn find_function_type(overloads: &Vec<FunctionOverload>, param_types: &ParamArray) -> Result<(FunctionOverload, Vec<ImplicitConversion>), TyperError> {

    fn find_overload_casts(overload: &FunctionOverload, param_types: &ParamArray) -> Result<Vec<ImplicitConversion>, ()> {
        let mut overload_casts = Vec::with_capacity(param_types.len());
        for (required_type, source_type) in overload.2.iter().zip(param_types.iter()) {
            if let Ok(cast) = ImplicitConversion::find(source_type, required_type) {
                overload_casts.push(cast)
            } else {
                return Err(())
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
    };
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
                    ConversionPriority::Equal => { },
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
    Err(TyperError::FunctionArgumentTypeMismatch(overloads.clone(), param_types.clone()))
}

fn apply_casts(casts: Vec<ImplicitConversion>, values: Vec<ir::Expression>) -> Vec<ir::Expression> {
    assert_eq!(casts.len(), values.len());
    values.iter().enumerate().map(|(index, value)| casts[index].apply(value.clone())).collect::<Vec<_>>()
}

fn write_function(unresolved: UnresolvedFunction, param_types: ParamArray, param_values: Vec<ir::Expression>) -> Result<TypedExpression, TyperError> {
    // Find the matching function overload
    let (FunctionOverload(name, return_type, _), casts) = try!(find_function_type(&unresolved.1, &param_types));
    // Apply implicit casts
    let param_values = apply_casts(casts, param_values);

    match name {
        FunctionName::Intrinsic(factory) => {
            Ok(TypedExpression::Value(
                ir::Expression::Intrinsic(Box::new(factory.create_intrinsic(&param_values))),
                return_type
            ))
        },
        FunctionName::User(id) => {
            Ok(TypedExpression::Value(ir::Expression::Call(Box::new(ir::Expression::Function(id)), param_values), return_type))
        },
    }
}

fn write_method(unresolved: UnresolvedMethod, param_types: ParamArray, param_values: Vec<ir::Expression>) -> Result<TypedExpression, TyperError> {
    // Find the matching method overload
    let (FunctionOverload(name, return_type, _), casts) = try!(find_function_type(&unresolved.2, &param_types));
    // Apply implicit casts
    let mut param_values = apply_casts(casts, param_values);
    // Add struct as implied first argument
    param_values.insert(0, unresolved.3);

    match name {
        FunctionName::Intrinsic(factory) => {
            Ok(TypedExpression::Value(
                ir::Expression::Intrinsic(Box::new(factory.create_intrinsic(&param_values))),
                return_type
            ))
        },
        FunctionName::User(_) => panic!("User defined methods should not exist"),
    }
}

fn parse_literal(ast: &ast::Literal) -> Result<TypedExpression, TyperError> {
    match ast {
        &ast::Literal::Bool(b) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Bool(b)), ir::Type::bool())),
        &ast::Literal::UntypedInt(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::UntypedInt(i)), ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::UntypedInt))))),
        &ast::Literal::Int(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Int(i)), ir::Type::int())),
        &ast::Literal::UInt(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::UInt(i)), ir::Type::uint())),
        &ast::Literal::Long(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Long(i)), ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::UntypedInt))))),
        &ast::Literal::Half(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Half(f)), ir::Type::float())),
        &ast::Literal::Float(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Float(f)), ir::Type::float())),
        &ast::Literal::Double(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Double(f)), ir::Type::double())),
    }
}

fn resolve_arithmetic_types(binop: &ir::BinOp, left: &ir::Type, right: &ir::Type, context: &ScopeContext) -> Result<(ImplicitConversion, ImplicitConversion), TyperError> {
    use hlsl::ir::Type;
    use hlsl::ir::StructuredType;
    use hlsl::ir::DataType;
    use hlsl::ir::ScalarType;

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

        let left = match *left { ScalarType::UntypedInt => ScalarType::Int, ScalarType::Bool => return Err(()), ref scalar => scalar.clone() };
        let right = match *right { ScalarType::UntypedInt => ScalarType::Int, ScalarType::Bool => return Err(()), ref scalar => scalar.clone() };

        let left_order = try!(get_order(&left));
        let right_order = try!(get_order(&right));

        if left_order > right_order { Ok(left) } else { Ok(right) }
    }

    fn do_noerror(_: &ir::BinOp, left: &ir::Type, right: &ir::Type) -> Result<(ImplicitConversion, ImplicitConversion), ()> {
        let (left, right, common) = match (left, right) {
            (&Type::Structured(StructuredType::Data(DataType::Scalar(ref ls))),
                &ir::Type::Structured(ir::StructuredType::Data(DataType::Scalar(ref rs)))) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common = ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(common_scalar)));
                (left, right, common)
            },
            (&Type::Structured(StructuredType::Data(DataType::Vector(ref ls, ref x1))),
                &Type::Structured(StructuredType::Data(DataType::Vector(ref rs, ref x2)))) if x1 == x2 => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common = ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Vector(common_scalar, *x2)));
                (left, right, common)
            },
            (&Type::Structured(StructuredType::Data(DataType::Matrix(ref ls, ref x1, ref y1))),
                &Type::Structured(StructuredType::Data(DataType::Matrix(ref rs, ref x2, ref y2)))) if x1 == x2 && y1 == y2 => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common = ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Matrix(common_scalar, *x2, *y2)));
                (left, right, common)
            },
            _ => return Err(()),
        };
        let lc = try!(ImplicitConversion::find(left, &common));
        let rc = try!(ImplicitConversion::find(right, &common));
        Ok((lc, rc))
    }

    match do_noerror(binop, left, right) {
        Ok(res) => Ok(res),
        Err(_) => Err(TyperError::BinaryOperationWrongTypes(binop.clone(), context.ir_type_to_error_type(left), context.ir_type_to_error_type(right))),
    }
}

fn parse_expr_binop(op: &ast::BinOp, lhs: &ast::Expression, rhs: &ast::Expression, context: &ScopeContext) -> Result<TypedExpression, TyperError> {
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
        ast::BinOp::Modulus => {
            let (lhs_cast, rhs_cast) = try!(resolve_arithmetic_types(op, &lhs_type, &rhs_type, context));
            assert_eq!(lhs_cast.get_target_type(), rhs_cast.get_target_type());
            let lhs_final = lhs_cast.apply(lhs_ir);
            let rhs_final = rhs_cast.apply(rhs_ir);
            Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(), Box::new(lhs_final), Box::new(rhs_final)), rhs_cast.get_target_type()))
        },
        ast::BinOp::LeftShift |
        ast::BinOp::RightShift => {
            Err(TyperError::Unimplemented)
        },
        ast::BinOp::LessThan |
        ast::BinOp::LessEqual |
        ast::BinOp::GreaterThan |
        ast::BinOp::GreaterEqual |
        ast::BinOp::Equality |
        ast::BinOp::Inequality => {
            let (lhs_cast, rhs_cast) = try!(resolve_arithmetic_types(op, &lhs_type, &rhs_type, context));
            assert_eq!(lhs_cast.get_target_type(), rhs_cast.get_target_type());
            let lhs_final = lhs_cast.apply(lhs_ir);
            let rhs_final = rhs_cast.apply(rhs_ir);
            Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(), Box::new(lhs_final), Box::new(rhs_final)), ir::Type::bool()))
        },
        ast::BinOp::Assignment => {
            match ImplicitConversion::find(&rhs_type, &lhs_type) {
                Ok(rhs_cast) => {
                    let rhs_final = rhs_cast.apply(rhs_ir);
                    Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(), Box::new(lhs_ir), Box::new(rhs_final)), lhs_type))
                },
                Err(()) => Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
            }
        },
    }
}

fn parse_expr(ast: &ast::Expression, context: &ScopeContext) -> Result<TypedExpression, TyperError> {
    match ast {
        &ast::Expression::Literal(ref lit) => parse_literal(lit),
        &ast::Expression::Variable(ref s) => Ok(try!(context.find_variable(s))),
        &ast::Expression::UnaryOperation(ref op, ref expr) => {
            match try!(parse_expr(expr, context)) {
                TypedExpression::Value(expr_ir, expr_ty) => {
                    Ok(TypedExpression::Value(ir::Expression::UnaryOperation(op.clone(), Box::new(expr_ir)), expr_ty))
                },
                _ => Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown)),
            }
        },
        &ast::Expression::BinaryOperation(ref op, ref lhs, ref rhs) => parse_expr_binop(op, lhs, rhs, context),
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
            let indexed_type = match array_ty {
                ir::Type::Object(ir::ObjectType::Buffer(data_type)) |
                ir::Type::Object(ir::ObjectType::RWBuffer(data_type)) => {
                    ir::Type::Structured(ir::StructuredType::Data(data_type))
                },
                ir::Type::Object(ir::ObjectType::StructuredBuffer(structured_type)) |
                ir::Type::Object(ir::ObjectType::RWStructuredBuffer(structured_type)) => {
                    ir::Type::Structured(structured_type)
                },
                _ => return Err(TyperError::ArrayIndexingNonArrayType),
            };
            let cast_to_int_result = ImplicitConversion::find(&subscript_ty, &ir::Type::int());
            let subscript_final = match cast_to_int_result {
                Err(_) => return Err(TyperError::ArraySubscriptIndexNotInteger),
                Ok(cast) => cast.apply(subscript_ir),
            };
            Ok(TypedExpression::Value(ir::Expression::ArraySubscript(Box::new(array_ir), Box::new(subscript_final)), indexed_type))
        },
        &ast::Expression::Member(ref composite, ref member) => {
            let composite_texp = try!(parse_expr(composite, context));
            let composite_pt = context.typed_expression_to_error_type(&composite_texp);
            let (composite_ir, composite_ty) = match composite_texp {
                TypedExpression::Value(composite_ir, composite_type) => (composite_ir, composite_type),
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_pt)),
            };
            let ety = match &composite_ty {
                &ir::Type::Structured(ir::StructuredType::Struct(ref id)) => {
                    match context.find_struct_member(id, member) {
                        Ok(ty) => ty,
                        Err(err) => return Err(err),
                    }
                }
                &ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Vector(ref scalar, ref x))) => {
                    // Todo: Swizzling
                    let exists = match &member[..] {
                        "x" | "r" if *x >= 1 => true,
                        "y" | "g" if *x >= 2 => true,
                        "z" | "b" if *x >= 3 => true,
                        "w" | "a" if *x >= 4 => true,
                        _ => false,
                    };
                    if exists {
                        ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(scalar.clone())))
                    } else {
                        return Err(TyperError::UnknownTypeMember(composite_pt, member.clone()));
                    }
                }
                &ir::Type::Object(ir::ObjectType::Buffer(ref data_type)) => {
                    match &member[..] {
                        "Load" => {
                            return Ok(TypedExpression::Method(UnresolvedMethod(
                                "Buffer::Load".to_string(),
                                ir::Type::Object(ir::ObjectType::Buffer(data_type.clone())),
                                vec![FunctionOverload(
                                    FunctionName::Intrinsic(IntrinsicFactory::Intrinsic2(ir::Intrinsic::BufferLoad)),
                                    ir::Type::Structured(ir::StructuredType::Data(data_type.clone())),
                                    vec![ir::Type::int()]
                                )],
                                composite_ir
                            )))
                        },
                        _ => return Err(TyperError::UnknownTypeMember(composite_pt, member.clone())),
                    }
                }
                &ir::Type::Object(ir::ObjectType::StructuredBuffer(ref structured_type)) => {
                    match &member[..] {
                        "Load" => {
                            return Ok(TypedExpression::Method(UnresolvedMethod(
                                "StructuredBuffer::Load".to_string(),
                                ir::Type::Object(ir::ObjectType::StructuredBuffer(structured_type.clone())),
                                vec![FunctionOverload(
                                    FunctionName::Intrinsic(IntrinsicFactory::Intrinsic2(ir::Intrinsic::StructuredBufferLoad)),
                                    ir::Type::Structured(structured_type.clone()),
                                    vec![ir::Type::int()]
                                )],
                                composite_ir
                            )))
                        },
                        _ => return Err(TyperError::UnknownTypeMember(composite_pt, member.clone())),
                    }
                }
                // Todo: Matrix components + Object members
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_pt)),
            };
            Ok(TypedExpression::Value(ir::Expression::Member(Box::new(composite_ir), member.clone()), ety))
        },
        &ast::Expression::Call(ref func, ref params) => {
            let func_texp = try!(parse_expr(func, context));
            let mut params_ir: Vec<ir::Expression> = vec![];
            let mut params_types: Vec<ir::Type> = vec![];
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
            };
            match func_texp {
                TypedExpression::Function(unresolved) => write_function(unresolved, params_types, params_ir),
                TypedExpression::Method(unresolved) => write_method(unresolved, params_types, params_ir),
                _ => return Err(TyperError::CallOnNonFunction),
            }
        },
        &ast::Expression::Cast(ref ty, ref expr) => {
            let expr_texp = try!(parse_expr(expr, context));
            let expr_pt = context.typed_expression_to_error_type(&expr_texp);
            match expr_texp {
                TypedExpression::Value(expr_ir, _) => {
                    let ir_type = try!(parse_type(ty, context));
                    Ok(TypedExpression::Value(ir::Expression::Cast(ir_type.clone(), Box::new(expr_ir)), ir_type))
                },
                _ => Err(TyperError::InvalidCast(expr_pt, ErrorType::Value(ty.clone()))),
            }
        },
    }
}

fn parse_expr_value_only(expr: &ast::Expression, context: &ScopeContext) -> Result<ir::Expression, TyperError> {
    let expr_ir = try!(parse_expr(expr, context));
    match expr_ir {
        TypedExpression::Value(expr, _) => Ok(expr),
        TypedExpression::Function(_) => Err(TyperError::FunctionNotCalled),
        TypedExpression::Method(_) => Err(TyperError::FunctionNotCalled),
    }
}

fn parse_vardef(ast: &ast::VarDef, context: ScopeContext) -> Result<(ir::VarDef, ScopeContext), TyperError> {
    let var_type = try!(parse_type(&ast.typename, &context));
    let assign_ir = match ast.assignment {
        Some(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, expt_ty) => {
                    match ImplicitConversion::find(&expt_ty, &var_type) {
                        Ok(rhs_cast) => Some(rhs_cast.apply(expr_ir)),
                        Err(()) => return Err(TyperError::WrongTypeInInitExpression),
                    }
                },
                _ => return Err(TyperError::FunctionTypeInInitExpression),
            }
        },
        None => None,
    };
    let var_name = ast.name.clone();
    let var_type = try!(parse_type(&ast.typename, &context));
    let mut context = context;
    let var_id = try!(context.insert_variable(var_name.clone(), var_type.clone()));
    let vd_ir = ir::VarDef { id: var_id, typename: var_type, assignment: assign_ir };
    Ok((vd_ir, context))
}

fn parse_condition(ast: &ast::Condition, context: ScopeContext) -> Result<(ir::Condition, ScopeContext), TyperError> {
    match ast {
        &ast::Condition::Expr(ref expr) => {
            let expr_ir = match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => expr_ir,
                _ => return Err(TyperError::FunctionNotCalled),
            };
            Ok((ir::Condition::Expr(expr_ir), context))
        },
        &ast::Condition::Assignment(ref vd) => {
            let (vd_ir, context) = try!(parse_vardef(vd, context));
            Ok((ir::Condition::Assignment(vd_ir), context))
        },
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
fn parse_scopeblock(ast: &ast::Statement, block_context: ScopeContext) -> Result<ir::ScopeBlock, TyperError> {
    match *ast {
        ast::Statement::Block(ref statement_vec) => {
            let (statements, block_context) = try!(parse_statement_vec(statement_vec, block_context));
            Ok(ir::ScopeBlock(statements, block_context.destruct()))
        },
        _ => {
            let (ir_statement, block_context) = try!(parse_statement(ast, block_context));
            let ir_vec = match ir_statement { Some(st) => vec![st], None => vec![] };
            Ok(ir::ScopeBlock(ir_vec, block_context.destruct()))
        },
    }
}

fn parse_statement(ast: &ast::Statement, context: ScopeContext) -> Result<(Option<ir::Statement>, ScopeContext), TyperError> {
    match ast {
        &ast::Statement::Empty => Ok((None, context)),
        &ast::Statement::Expression(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => Ok((Some(ir::Statement::Expression(expr_ir)), context)),
                _ => return Err(TyperError::FunctionNotCalled),
            }
        },
        &ast::Statement::Var(ref vd) => {
            let (vd_ir, context) = try!(parse_vardef(vd, context));
            Ok((Some(ir::Statement::Var(vd_ir)), context))
        },
        &ast::Statement::Block(ref statement_vec) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (statements, scoped_context) = try!(parse_statement_vec(statement_vec, scoped_context));
            let decls = scoped_context.destruct();
            Ok((Some(ir::Statement::Block(ir::ScopeBlock(statements, decls))), context))
        },
        &ast::Statement::If(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let cond_ir = try!(parse_expr_value_only(cond, &scoped_context));
            let scope_block = try!(parse_scopeblock(statement, scoped_context));
            Ok((Some(ir::Statement::If(cond_ir, scope_block)), context))
        },
        &ast::Statement::For(ref init, ref cond, ref iter, ref statement) =>  {
            let scoped_context = ScopeContext::from_scope(&context);
            let (init_ir, scoped_context) = try!(parse_condition(init, scoped_context));
            let cond_ir = try!(parse_expr_value_only(cond, &scoped_context));
            let iter_ir = try!(parse_expr_value_only(iter, &scoped_context));
            let scope_block = try!(parse_scopeblock(statement, scoped_context));
            Ok((Some(ir::Statement::For(init_ir, cond_ir, iter_ir, scope_block)), context))
        },
        &ast::Statement::While(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let cond_ir = try!(parse_expr_value_only(cond, &scoped_context));
            let scope_block = try!(parse_scopeblock(statement, scoped_context));
            Ok((Some(ir::Statement::While(cond_ir, scope_block)), context))
        },
        &ast::Statement::Return(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => Ok((Some(ir::Statement::Return(expr_ir)), context)),
                _ => return Err(TyperError::FunctionNotCalled),
            }
        },
    }
}

fn parse_statement_vec(ast: &[ast::Statement], context: ScopeContext) -> Result<(Vec<ir::Statement>, ScopeContext), TyperError> {
    let mut context = context;
    let mut body_ir = vec![];
    for statement_ast in ast {
        let (statement_ir_opt, next_context) = try!(parse_statement(&statement_ast, context));
        match statement_ir_opt {
            Some(statement_ir) => body_ir.push(statement_ir),
            None => { },
        };
        context = next_context;
    }
    Ok((body_ir, context))
}

fn parse_rootdefinition_struct(sd: &ast::StructDefinition, mut context: GlobalContext) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
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
    };
    let name = &sd.name;
    match context.insert_struct(name, member_map) {
        Some(id) => {
            let struct_def = ir::StructDefinition {
                id: id,
                members: members,
            };
            Ok((ir::RootDefinition::Struct(struct_def), context))
        },
        None => Err(TyperError::StructAlreadyDefined(name.clone())),
    }
}

fn parse_rootdefinition_constantbuffer(cb: &ast::ConstantBuffer, mut context: GlobalContext, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
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
    };
    let id = match context.insert_cbuffer(&cb_name, members_map) {
        Some(id) => id,
        None => return Err(TyperError::ConstantBufferAlreadyDefined(cb_name.clone())),
    };
    let cb_ir = ir::ConstantBuffer { id: id, members: members };
    match cb.slot {
        Some(ast::ConstantSlot(slot)) => {
            match globals.constants.insert(slot, cb_ir.id.clone()) {
                // Todo: ConstantSlotAlreadyUsed should get name of previously used slot
                Some(_) => return Err(TyperError::ConstantSlotAlreadyUsed(cb_name.clone())),
                None => { },
            }
        },
        None => { },
    }
    Ok((ir::RootDefinition::ConstantBuffer(cb_ir), context))
}

fn parse_rootdefinition_globalvariable(gv: &ast::GlobalVariable, mut context: GlobalContext, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let var_name = gv.name.clone();
    let var_type = try!(parse_type(&gv.typename, &context));
    let var_id = try!(context.insert_variable(var_name.clone(), var_type.clone()));
    let gv_ir = ir::GlobalVariable { id: var_id, typename: var_type };
    let entry = ir::GlobalEntry { id: var_id, typename: gv_ir.typename.clone() };
    match gv.slot {
        Some(ast::GlobalSlot::ReadSlot(slot)) => {
            match globals.r_resources.insert(slot, entry) {
                Some(currently_used_by) => return Err(TyperError::ReadResourceSlotAlreadyUsed(currently_used_by.id.clone(), gv_ir.id.clone())),
                None => { },
            }
        },
        Some(ast::GlobalSlot::ReadWriteSlot(slot)) => {
            match globals.rw_resources.insert(slot, entry) {
                Some(currently_used_by) => return Err(TyperError::ReadWriteResourceSlotAlreadyUsed(currently_used_by.id.clone(), gv_ir.id.clone())),
                None => { },
            }
        },
        None => { },
    }
    Ok((ir::RootDefinition::GlobalVariable(gv_ir), context))
}

fn parse_rootdefinition_function(fd: &ast::FunctionDefinition, mut context: GlobalContext) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {

    let mut scoped_context = ScopeContext::from_global(&context);
    let func_params = {
        let mut vec = vec![];
        for param in &fd.params {
            let var_type = try!(parse_type(&param.typename, &context));
            let var_id = try!(scoped_context.insert_variable(param.name.clone(), var_type.clone()));
            vec.push(ir::FunctionParam {
                id: var_id,
                typename: var_type,
            });
        }
        vec
    };
    let (body_ir, scoped_context) = try!(parse_statement_vec(&fd.body, scoped_context));
    let decls = scoped_context.destruct();
    let return_type = try!(parse_type(&fd.returntype, &context));
    let fd_ir = ir::FunctionDefinition {
        id: context.make_function_id(),
        returntype: return_type,
        params: func_params,
        body: body_ir,
        scope: decls,
        attributes: fd.attributes.clone(),
    };
    let func_type = FunctionOverload(
        FunctionName::User(fd_ir.id),
        fd_ir.returntype.clone(),
        fd_ir.params.iter().map(|p| { p.typename.clone() }).collect()
    );
    try!(context.insert_function(fd.name.clone(), func_type));
    Ok((ir::RootDefinition::Function(fd_ir), context))
}

fn parse_rootdefinition_kernel(fd: &ast::FunctionDefinition, context: GlobalContext) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {

    let mut scoped_context = ScopeContext::from_global(&context);
    let kernel_params = {
        let mut vec = vec![];
        for param in &fd.params {
            let var_type = try!(parse_type(&param.typename, &context));
            let var_id = try!(scoped_context.insert_variable(param.name.clone(), var_type.clone()));
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

    fn find_dispatch_dimensions(attributes: &[ast::FunctionAttribute]) -> Result<ir::Dimension, TyperError> {
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
        body: body_ir,
        scope: decls,
    };
    Ok((ir::RootDefinition::Kernel(kernel), context))
}

fn parse_rootdefinition(ast: &ast::RootDefinition, context: GlobalContext, globals: &mut ir::GlobalTable, entry_point: &str) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    match ast {
        &ast::RootDefinition::Struct(ref sd) => parse_rootdefinition_struct(sd, context),
        &ast::RootDefinition::SamplerState => Ok((ir::RootDefinition::SamplerState, context)),
        &ast::RootDefinition::ConstantBuffer(ref cb) => parse_rootdefinition_constantbuffer(cb, context, globals),
        &ast::RootDefinition::GlobalVariable(ref gv) => parse_rootdefinition_globalvariable(gv, context, globals),
        &ast::RootDefinition::Function(ref fd) if fd.name == entry_point => parse_rootdefinition_kernel(fd, context),
        &ast::RootDefinition::Function(ref fd) => parse_rootdefinition_function(fd, context),
    }
}

pub fn typeparse(ast: &ast::Module) -> Result<ir::Module, TyperError> {
    let mut context = GlobalContext::new();

    let mut global_table = ir::GlobalTable::new();
    let mut root_definitions = vec![];

    for def in &ast.root_definitions {
        let (def_ir, next_context) = try!(parse_rootdefinition(&def, context, &mut global_table, &ast.entry_point.clone()));
        root_definitions.push(def_ir);
        context = next_context;
    }

    let global_declarations = context.destruct();

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
            },
            _ => { },
        }
    }
    if !has_kernel {
        return Err(TyperError::KernelNotDefined);
    }

    Ok(ir)
}

#[test]
fn test_typeparse() {
    let module = ast::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            ast::RootDefinition::GlobalVariable(ast::GlobalVariable {
                name: "g_myInBuffer".to_string(),
                typename: ast::Type::Object(ast::ObjectType::Buffer(ast::DataType::Scalar(ast::ScalarType::Int))),
                slot: Some(ast::GlobalSlot::ReadSlot(0)),
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::Void,
                params: vec![ast::FunctionParam { name: "x".to_string(), typename: ast::Type::uint(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::Void,
                params: vec![ast::FunctionParam { name: "x".to_string(), typename: ast::Type::float(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: ast::Type::Void,
                params: vec![],
                body: vec![
                    ast::Statement::Empty,
                    ast::Statement::Var(ast::VarDef { name: "a".to_string(), typename: ast::Type::uint(), assignment: None }),
                    ast::Statement::Var(ast::VarDef { name: "b".to_string(), typename: ast::Type::uint(), assignment: None }),
                    ast::Statement::Expression(
                        ast::Expression::BinaryOperation(ast::BinOp::Assignment,
                            Box::new(ast::Expression::Variable("a".to_string())),
                            Box::new(ast::Expression::Variable("b".to_string()))
                        )
                    ),
                    ast::Statement::If(
                        ast::Expression::Variable("b".to_string()),
                        Box::new(ast::Statement::Empty),
                    ),
                    ast::Statement::Expression(
                        ast::Expression::BinaryOperation(ast::BinOp::Assignment,
                            Box::new(ast::Expression::ArraySubscript(
                                Box::new(ast::Expression::Variable("g_myInBuffer".to_string())),
                                Box::new(ast::Expression::Literal(ast::Literal::Int(0)))
                            )),
                            Box::new(ast::Expression::Literal(ast::Literal::Int(4)))
                        ),
                    ),
                    ast::Statement::Expression(
                        ast::Expression::Call(
                            Box::new(ast::Expression::Variable("myFunc".to_string())),
                            vec![
                                ast::Expression::Variable("b".to_string())
                            ]
                        ),
                    ),
                ],
                attributes: vec![ast::FunctionAttribute::NumThreads(8, 8, 1)],
            }),
        ],
    };
    let res = typeparse(&module);
    assert!(res.is_ok(), "{:?}", res);
}
