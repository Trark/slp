use super::casting::ImplicitConversion;
use super::intrinsics;
use super::intrinsics::IntrinsicFactory;
use crate::pel;
use crate::rel;
use rel::ReduceContext;
use slp_lang_hir as ir;
use slp_lang_hir::ExpressionType;
use slp_lang_hir::Intrinsic;
use slp_lang_hir::ToExpressionType;
use slp_lang_hst as ast;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;

#[derive(PartialEq, Debug, Clone)]
pub enum TyperError {
    Unimplemented,
    ExpressionSequenceOperatorNotImplemented,

    ValueAlreadyDefined(String, ErrorType, ErrorType),
    StructAlreadyDefined(String),
    ConstantBufferAlreadyDefined(String),

    ConstantSlotAlreadyUsed(ir::ConstantBufferId, ir::ConstantBufferId),
    ReadResourceSlotAlreadyUsed(ir::GlobalId, ir::GlobalId),
    ReadWriteResourceSlotAlreadyUsed(ir::GlobalId, ir::GlobalId),
    SamplerResourceSlotAlreadyUsed(ir::GlobalId, ir::GlobalId),

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
    BinaryOperationWrongTypes(ast::BinOp, ErrorType, ErrorType),
    BinaryOperationNonNumericType,
    TernaryConditionRequiresBoolean(ErrorType),
    TernaryArmsMustHaveSameType(ErrorType, ErrorType),

    ExpectedValueExpression(ErrorType),

    InvalidCast(ErrorType, ErrorType),

    InitializerExpressionWrongType,
    InitializerAggregateDoesNotMatchType,
    InitializerAggregateWrongDimension,
    InitializerAggregateWrongElementType,

    WrongTypeInConstructor,
    WrongTypeInReturnStatement,
    FunctionNotCalled,

    KernelNotDefined,
    KernelDefinedMultipleTimes,
    KernelHasNoDispatchDimensions,
    KernelDispatchDimensionMustBeConstantExpression,
    KernelHasParamWithBadSemantic(ast::FunctionParam),
    KernelHasParamWithoutSemantic(ast::FunctionParam),

    LvalueRequired,
    ArrayDimensionsMustBeConstantExpression(ast::Expression),
    ArrayDimensionNotSpecified,

    RelReduceError(rel::ReduceError),
    RelCombineError(rel::CombineError),
}

impl From<rel::ReduceError> for TyperError {
    fn from(err: rel::ReduceError) -> TyperError {
        TyperError::RelReduceError(err)
    }
}

impl From<rel::CombineError> for TyperError {
    fn from(err: rel::CombineError) -> TyperError {
        TyperError::RelCombineError(err)
    }
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
pub struct UnresolvedFunction(pub String, pub Vec<FunctionOverload>);

#[derive(PartialEq, Debug, Clone)]
pub enum ErrorType {
    Untyped(ast::Type),
    Value(ir::Type),
    Function(String, Vec<FunctionOverload>),
    Method(String, ClassType, Vec<FunctionOverload>),
    Unknown,
}

pub trait ToErrorType {
    fn to_error_type(&self) -> ErrorType;
}

impl ToErrorType for ir::Type {
    fn to_error_type(&self) -> ErrorType {
        ErrorType::Value(self.clone())
    }
}

impl ToErrorType for ir::ExpressionType {
    fn to_error_type(&self) -> ErrorType {
        ErrorType::Value(self.0.clone())
    }
}

impl fmt::Display for TyperError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TyperError::Unimplemented => write!(f, "unimplemented"),
            TyperError::ExpressionSequenceOperatorNotImplemented => {
                write!(f, "operator ',' not implemented")
            }

            TyperError::ValueAlreadyDefined(_, _, _) => write!(f, "identifier already defined"),
            TyperError::StructAlreadyDefined(_) => write!(f, "struct aready defined"),
            TyperError::ConstantBufferAlreadyDefined(_) => write!(f, "cbuffer aready defined"),

            TyperError::ConstantSlotAlreadyUsed(_, _) => {
                write!(f, "global constant slot already used")
            }
            TyperError::ReadResourceSlotAlreadyUsed(_, _) => {
                write!(f, "global resource slot already used")
            }
            TyperError::ReadWriteResourceSlotAlreadyUsed(_, _) => {
                write!(f, "global writable resource slot already used")
            }
            TyperError::SamplerResourceSlotAlreadyUsed(_, _) => {
                write!(f, "sampler slot already used")
            }

            TyperError::UnknownIdentifier(_) => write!(f, "unknown identifier"),
            TyperError::UnknownType(_) => write!(f, "unknown type name"),

            TyperError::TypeDoesNotHaveMembers(_) => {
                write!(f, "unknown member (type has no members)")
            }
            TyperError::UnknownTypeMember(_, _) => write!(f, "unknown member"),
            TyperError::InvalidSwizzle(_, _) => write!(f, "invalid swizzle"),

            TyperError::ArrayIndexingNonArrayType => {
                write!(f, "array index applied to non-array type")
            }
            TyperError::ArraySubscriptIndexNotInteger => {
                write!(f, "array subscripts must be integers")
            }

            TyperError::CallOnNonFunction => {
                write!(f, "function call applied to non-function type")
            }

            TyperError::FunctionPassedToAnotherFunction(_, _) => {
                write!(f, "functions can not be passed to other functions")
            }
            TyperError::FunctionArgumentTypeMismatch(_, _) => {
                write!(f, "wrong parameters given to function")
            }
            TyperError::NumericConstructorWrongArgumentCount => {
                write!(f, "wrong number of arguments to constructor")
            }

            TyperError::UnaryOperationWrongTypes(_, _) => {
                write!(f, "operation does not support the given types")
            }
            TyperError::BinaryOperationWrongTypes(_, _, _) => {
                write!(f, "operation does not support the given types")
            }
            TyperError::BinaryOperationNonNumericType => {
                write!(f, "non-numeric type in binary operation")
            }
            TyperError::TernaryConditionRequiresBoolean(_) => {
                write!(f, "ternary condition must be boolean")
            }
            TyperError::TernaryArmsMustHaveSameType(_, _) => {
                write!(f, "ternary arms must have the same type")
            }

            TyperError::ExpectedValueExpression(_) => write!(f, "expected a value expression"),

            TyperError::InvalidCast(_, _) => write!(f, "invalid cast"),

            TyperError::InitializerExpressionWrongType => {
                write!(f, "wrong type in variable initialization")
            }
            TyperError::InitializerAggregateDoesNotMatchType => {
                write!(f, "initializer does not match type")
            }
            TyperError::InitializerAggregateWrongDimension => {
                write!(f, "initializer has incorrect number of elements")
            }
            TyperError::InitializerAggregateWrongElementType => {
                write!(f, "initializer element has incorrect type")
            }

            TyperError::WrongTypeInConstructor => write!(f, "wrong type in numeric constructor"),
            TyperError::WrongTypeInReturnStatement => write!(f, "wrong type in return statement"),
            TyperError::FunctionNotCalled => write!(f, "function not called"),

            TyperError::KernelNotDefined => write!(f, "entry point not found"),
            TyperError::KernelDefinedMultipleTimes => write!(f, "multiple entry points found"),
            TyperError::KernelHasNoDispatchDimensions => {
                write!(f, "compute kernels require a dispatch dimension")
            }
            TyperError::KernelDispatchDimensionMustBeConstantExpression => {
                write!(f, "dispatch dimension must be constant expression")
            }
            TyperError::KernelHasParamWithBadSemantic(_) => {
                write!(f, "kernel parameter did not have a valid kernel semantic")
            }
            TyperError::KernelHasParamWithoutSemantic(_) => {
                write!(f, "kernel parameter did not have a kernel semantic")
            }

            TyperError::LvalueRequired => write!(f, "lvalue is required in this context"),
            TyperError::ArrayDimensionsMustBeConstantExpression(_) => {
                write!(f, "array dimensions must be constant")
            }
            TyperError::ArrayDimensionNotSpecified => write!(f, "array not given any dimensions"),
            TyperError::RelReduceError(ref re) => write!(f, "{}", re),
            TyperError::RelCombineError(ref ce) => write!(f, "{}", ce),
        }
    }
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
    global_slots_s: Vec<(u32, ir::GlobalEntry)>,
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

pub enum VariableExpression {
    Local(ir::VariableRef, ir::Type),
    Global(ir::GlobalId, ir::Type),
    Constant(ir::ConstantBufferId, String, ir::Type),
    Function(UnresolvedFunction),
}

pub trait ExpressionContext: StructIdFinder + ir::TypeContext + ReduceContext {
    fn find_variable(&self, name: &String) -> Result<VariableExpression, TyperError>;
    fn find_struct_member(
        &self,
        id: &ir::StructId,
        member_name: &String,
    ) -> Result<ir::Type, TyperError>;
    fn get_return_type(&self) -> ir::Type;

    fn as_struct_id_finder(&self) -> &dyn StructIdFinder;
    fn as_reduce_context(&self) -> &dyn ReduceContext;
}

pub trait StructIdFinder {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError>;
}

impl VariableBlock {
    fn new() -> VariableBlock {
        VariableBlock {
            variables: HashMap::new(),
            next_free_variable_id: ir::VariableId(0),
        }
    }

    fn insert_variable(
        &mut self,
        name: String,
        typename: ir::Type,
        _: &TypeBlock,
    ) -> Result<ir::VariableId, TyperError> {
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(
                name,
                ty.to_error_type(),
                typename.to_error_type(),
            ));
        };
        match self.variables.entry(name.clone()) {
            Entry::Occupied(occupied) => Err(TyperError::ValueAlreadyDefined(
                name,
                occupied.get().0.to_error_type(),
                typename.to_error_type(),
            )),
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

    fn find_variable(&self, name: &String, scopes_up: u32) -> Option<VariableExpression> {
        match self.variables.get(name) {
            Some(&(ref ty, ref id)) => {
                let var = ir::VariableRef(id.clone(), ir::ScopeRef(scopes_up));
                return Some(VariableExpression::Local(var, ty.clone()));
            }
            None => None,
        }
    }

    fn destruct(self) -> HashMap<ir::VariableId, (String, ir::Type)> {
        self.variables
            .iter()
            .fold(HashMap::new(), |mut map, (name, &(ref ty, ref id))| {
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

    fn insert_struct(
        &mut self,
        name: &String,
        members: HashMap<String, ir::Type>,
    ) -> Option<ir::StructId> {
        let id = self.next_free_struct_id;
        self.next_free_struct_id = ir::StructId(self.next_free_struct_id.0 + 1);
        match (
            self.struct_ids.entry(name.clone()),
            self.struct_names.entry(id.clone()),
            self.struct_definitions.entry(id.clone()),
        ) {
            (Entry::Vacant(id_entry), Entry::Vacant(name_entry), Entry::Vacant(def_entry)) => {
                id_entry.insert(id.clone());
                name_entry.insert(name.clone());
                def_entry.insert(members);
                Some(id.clone())
            }
            _ => None,
        }
    }

    fn find_struct_member(
        &self,
        id: &ir::StructId,
        member_name: &String,
    ) -> Result<ir::Type, TyperError> {
        match self.struct_definitions.get(id) {
            Some(def) => {
                def.get(member_name)
                    .map(|ty| ty.clone())
                    .ok_or(TyperError::UnknownTypeMember(
                        ir::Type::from_struct(id.clone()).to_error_type(),
                        member_name.clone(),
                    ))
            }
            None => Err(TyperError::UnknownType(
                ir::Type::from_struct(id.clone()).to_error_type(),
            )),
        }
    }

    fn insert_cbuffer(
        &mut self,
        name: &String,
        members: HashMap<String, ir::Type>,
    ) -> Option<ir::ConstantBufferId> {
        let id = self.next_free_cbuffer_id;
        self.next_free_cbuffer_id = ir::ConstantBufferId(self.next_free_cbuffer_id.0 + 1);
        match (
            self.cbuffer_ids.entry(name.clone()),
            self.cbuffer_names.entry(id.clone()),
            self.cbuffer_definitions.entry(id.clone()),
        ) {
            (Entry::Vacant(id_entry), Entry::Vacant(name_entry), Entry::Vacant(def_entry)) => {
                id_entry.insert(id.clone());
                name_entry.insert(name.clone());
                def_entry.insert(members);
                Some(id.clone())
            }
            _ => None,
        }
    }

    fn find_variable(&self, name: &String) -> Option<VariableExpression> {
        for (id, members) in &self.cbuffer_definitions {
            for (member_name, ty) in members {
                if member_name == name {
                    return Some(VariableExpression::Constant(
                        id.clone(),
                        name.clone(),
                        ty.clone(),
                    ));
                }
            }
        }
        None
    }
}

impl StructIdFinder for TypeBlock {
    fn find_struct_id(&self, name: &String) -> Result<ir::StructId, TyperError> {
        self.struct_ids
            .get(name)
            .map(|id| id.clone())
            .ok_or(TyperError::UnknownType(ErrorType::Untyped(
                ast::Type::from_layout(ast::TypeLayout::Custom(name.clone())),
            )))
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
            global_slots_s: vec![],
            global_slots_constants: vec![],
        }
    }

    pub fn insert_function(
        &mut self,
        name: String,
        function_type: FunctionOverload,
    ) -> Result<(), TyperError> {
        // Error if a variable of the same name already exists
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(
                name,
                ty.to_error_type(),
                ErrorType::Unknown,
            ));
        };

        fn insert_function_name(
            function_names: &mut HashMap<ir::FunctionId, String>,
            function_type: FunctionOverload,
            name: String,
        ) {
            match function_type.0 {
                FunctionName::User(id) => match function_names.entry(id) {
                    Entry::Occupied(_) => {
                        panic!("function id named twice");
                    }
                    Entry::Vacant(vacant) => {
                        vacant.insert(name);
                    }
                },
                FunctionName::Intrinsic(_) => {}
            }
        }

        // Try to add the function
        match self.functions.entry(name.clone()) {
            Entry::Occupied(mut occupied) => {
                // Fail if the overload already exists
                for &FunctionOverload(_, _, ref args) in &occupied.get().1 {
                    if *args == function_type.2 {
                        return Err(TyperError::ValueAlreadyDefined(
                            name,
                            ErrorType::Unknown,
                            ErrorType::Unknown,
                        ));
                    }
                }
                // Insert a new overload
                insert_function_name(&mut self.function_names, function_type.clone(), name);
                occupied.get_mut().1.push(function_type);
                Ok(())
            }
            Entry::Vacant(vacant) => {
                // Insert a new function with one overload
                insert_function_name(
                    &mut self.function_names,
                    function_type.clone(),
                    name.clone(),
                );
                vacant.insert(UnresolvedFunction(name, vec![function_type]));
                Ok(())
            }
        }
    }

    fn find_variable_recur(
        &self,
        name: &String,
        scopes_up: u32,
    ) -> Result<VariableExpression, TyperError> {
        assert!(scopes_up != 0);
        match self.functions.get(name) {
            Some(tys) => return Ok(VariableExpression::Function(tys.clone())),
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

    fn insert_global(
        &mut self,
        name: String,
        typename: ir::Type,
    ) -> Result<ir::GlobalId, TyperError> {
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(
                name,
                ty.to_error_type(),
                typename.to_error_type(),
            ));
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

    fn find_global(&self, name: &String) -> Option<VariableExpression> {
        match self.globals.get(name) {
            Some(&(ref ty, ref id)) => {
                return Some(VariableExpression::Global(id.clone(), ty.clone()))
            }
            None => None,
        }
    }

    fn insert_struct(
        &mut self,
        name: &String,
        members: HashMap<String, ir::Type>,
    ) -> Option<ir::StructId> {
        self.types.insert_struct(name, members)
    }

    fn insert_cbuffer(
        &mut self,
        name: &String,
        members: HashMap<String, ir::Type>,
    ) -> Option<ir::ConstantBufferId> {
        self.types.insert_cbuffer(name, members)
    }

    fn get_type_block(&self) -> &TypeBlock {
        &self.types
    }
}

impl ExpressionContext for GlobalContext {
    fn find_variable(&self, name: &String) -> Result<VariableExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct_member(
        &self,
        id: &ir::StructId,
        member_name: &String,
    ) -> Result<ir::Type, TyperError> {
        self.types.find_struct_member(id, member_name)
    }

    fn get_return_type(&self) -> ir::Type {
        self.current_return_type
            .clone()
            .expect("not inside function")
    }

    fn as_struct_id_finder(&self) -> &dyn StructIdFinder {
        self
    }

    fn as_reduce_context(&self) -> &dyn ReduceContext {
        self
    }
}

impl ReduceContext for GlobalContext {
    fn find_overload(&self, id: &ir::FunctionId) -> FunctionOverload {
        // Search debug names to get position in name -> unresolved map
        let name = match self.function_names.get(id) {
            Some(name) => name,
            None => panic!("unknown function id name in user call"),
        };
        let unresolved = match self.functions.get(name) {
            Some(unresolved) => unresolved,
            None => panic!("unknown function name in user call"),
        };
        for overload in &unresolved.1 {
            match overload.0 {
                FunctionName::User(ref overload_id) => {
                    if id == overload_id {
                        return overload.clone();
                    }
                }
                _ => {}
            }
        }
        panic!("function id does not exist")
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
    fn get_constant(
        &self,
        id: &ir::ConstantBufferId,
        name: &str,
    ) -> Result<ExpressionType, ir::TypeError> {
        match self.types.cbuffer_definitions.get(id) {
            Some(ref cm) => match cm.get(name) {
                Some(ref ty) => Ok(ty.to_lvalue()),
                None => Err(ir::TypeError::ConstantDoesNotExist(
                    id.clone(),
                    name.to_string(),
                )),
            },
            None => Err(ir::TypeError::ConstantBufferDoesNotExist(id.clone())),
        }
    }
    fn get_struct_member(
        &self,
        id: &ir::StructId,
        name: &str,
    ) -> Result<ExpressionType, ir::TypeError> {
        match self.types.struct_definitions.get(&id) {
            Some(ref cm) => match cm.get(name) {
                Some(ref ty) => Ok(ty.to_lvalue()),
                None => Err(ir::TypeError::StructMemberDoesNotExist(
                    id.clone(),
                    name.to_string(),
                )),
            },
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

    fn find_variable_recur(
        &self,
        name: &String,
        scopes_up: u32,
    ) -> Result<VariableExpression, TyperError> {
        match self.variables.find_variable(name, scopes_up) {
            Some(texp) => return Ok(texp),
            None => self.parent.find_variable_recur(name, scopes_up + 1),
        }
    }

    fn destruct(self) -> ir::ScopedDeclarations {
        ir::ScopedDeclarations {
            variables: self.variables.destruct(),
        }
    }

    fn insert_variable(
        &mut self,
        name: String,
        typename: ir::Type,
    ) -> Result<ir::VariableId, TyperError> {
        let type_block = self.parent.get_type_block();
        let variables = &mut self.variables;
        variables.insert_variable(name, typename, type_block)
    }

    fn get_type_block(&self) -> &TypeBlock {
        self.parent.get_type_block()
    }
}

impl ExpressionContext for ScopeContext {
    fn find_variable(&self, name: &String) -> Result<VariableExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct_member(
        &self,
        id: &ir::StructId,
        member_name: &String,
    ) -> Result<ir::Type, TyperError> {
        self.parent.find_struct_member(id, member_name)
    }

    fn get_return_type(&self) -> ir::Type {
        self.parent.get_return_type()
    }

    fn as_struct_id_finder(&self) -> &dyn StructIdFinder {
        self
    }

    fn as_reduce_context(&self) -> &dyn ReduceContext {
        self
    }
}

impl ReduceContext for ScopeContext {
    fn find_overload(&self, id: &ir::FunctionId) -> FunctionOverload {
        self.parent.find_overload(id)
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
            up => self
                .parent
                .get_local(&ir::VariableRef(id.clone(), ir::ScopeRef(up - 1))),
        }
    }
    fn get_global(&self, id: &ir::GlobalId) -> Result<ExpressionType, ir::TypeError> {
        self.parent.get_global(id)
    }
    fn get_constant(
        &self,
        id: &ir::ConstantBufferId,
        name: &str,
    ) -> Result<ExpressionType, ir::TypeError> {
        self.parent.get_constant(id, name)
    }
    fn get_struct_member(
        &self,
        id: &ir::StructId,
        name: &str,
    ) -> Result<ExpressionType, ir::TypeError> {
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

impl Context {
    fn find_variable_recur(
        &self,
        name: &String,
        scopes_up: u32,
    ) -> Result<VariableExpression, TyperError> {
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

    fn find_overload(&self, id: &ir::FunctionId) -> FunctionOverload {
        match *self {
            Context::Global(ref global) => global.find_overload(id),
            Context::Scope(ref scope) => scope.find_overload(id),
        }
    }

    fn get_return_type(&self) -> ir::Type {
        match *self {
            Context::Global(ref global) => global.get_return_type(),
            Context::Scope(ref scope) => scope.get_return_type(),
        }
    }

    fn find_struct_member(
        &self,
        id: &ir::StructId,
        member_name: &String,
    ) -> Result<ir::Type, TyperError> {
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
    fn get_constant(
        &self,
        id: &ir::ConstantBufferId,
        name: &str,
    ) -> Result<ExpressionType, ir::TypeError> {
        match *self {
            Context::Global(ref global) => global.get_constant(id, name),
            Context::Scope(ref scope) => scope.get_constant(id, name),
        }
    }
    fn get_struct_member(
        &self,
        id: &ir::StructId,
        name: &str,
    ) -> Result<ExpressionType, ir::TypeError> {
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
    for &(ref name, ref params, ref factory) in funcs {
        let return_type = match *factory {
            IntrinsicFactory::Intrinsic0(ref i) => i.get_return_type(),
            IntrinsicFactory::Intrinsic1(ref i) => i.get_return_type(),
            IntrinsicFactory::Intrinsic2(ref i) => i.get_return_type(),
            IntrinsicFactory::Intrinsic3(ref i) => i.get_return_type(),
        };
        let overload = FunctionOverload(
            FunctionName::Intrinsic(factory.clone()),
            return_type.0,
            params.to_vec(),
        );
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

pub fn parse_datalayout(ty: &ast::DataLayout) -> Result<ir::DataLayout, TyperError> {
    Ok(match *ty {
        ast::DataLayout::Scalar(ref scalar) => ir::DataLayout::Scalar(parse_scalartype(scalar)?),
        ast::DataLayout::Vector(ref scalar, ref x) => {
            ir::DataLayout::Vector(parse_scalartype(scalar)?, *x)
        }
        ast::DataLayout::Matrix(ref scalar, ref x, ref y) => {
            ir::DataLayout::Matrix(parse_scalartype(scalar)?, *x, *y)
        }
    })
}

fn parse_datatype(ty: &ast::DataType) -> Result<ir::DataType, TyperError> {
    let &ast::DataType(ref tyl, ref modifier) = ty;
    Ok(ir::DataType(
        parse_datalayout(tyl)?,
        parse_modifier(modifier),
    ))
}

fn parse_structuredlayout(
    ty: &ast::StructuredLayout,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::StructuredLayout, TyperError> {
    Ok(match *ty {
        ast::StructuredLayout::Scalar(ref scalar) => {
            ir::StructuredLayout::Scalar(parse_scalartype(scalar)?)
        }
        ast::StructuredLayout::Vector(ref scalar, ref x) => {
            ir::StructuredLayout::Vector(parse_scalartype(scalar)?, *x)
        }
        ast::StructuredLayout::Matrix(ref scalar, ref x, ref y) => {
            ir::StructuredLayout::Matrix(parse_scalartype(scalar)?, *x, *y)
        }
        ast::StructuredLayout::Custom(ref name) => {
            ir::StructuredLayout::Struct(struct_finder.find_struct_id(name)?)
        }
    })
}

fn parse_structuredtype(
    ty: &ast::StructuredType,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::StructuredType, TyperError> {
    let &ast::StructuredType(ref tyl, ref modifier) = ty;
    Ok(ir::StructuredType(
        parse_structuredlayout(tyl, struct_finder)?,
        parse_modifier(modifier),
    ))
}

fn parse_objecttype(
    ty: &ast::ObjectType,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::ObjectType, TyperError> {
    Ok(match *ty {
        ast::ObjectType::Buffer(ref data_type) => {
            ir::ObjectType::Buffer(parse_datatype(data_type)?)
        }
        ast::ObjectType::RWBuffer(ref data_type) => {
            ir::ObjectType::RWBuffer(parse_datatype(data_type)?)
        }
        ast::ObjectType::ByteAddressBuffer => ir::ObjectType::ByteAddressBuffer,
        ast::ObjectType::RWByteAddressBuffer => ir::ObjectType::RWByteAddressBuffer,
        ast::ObjectType::StructuredBuffer(ref structured_type) => {
            ir::ObjectType::StructuredBuffer(parse_structuredtype(structured_type, struct_finder)?)
        }
        ast::ObjectType::RWStructuredBuffer(ref structured_type) => {
            ir::ObjectType::RWStructuredBuffer(parse_structuredtype(
                structured_type,
                struct_finder,
            )?)
        }
        ast::ObjectType::AppendStructuredBuffer(ref structured_type) => {
            ir::ObjectType::AppendStructuredBuffer(parse_structuredtype(
                structured_type,
                struct_finder,
            )?)
        }
        ast::ObjectType::ConsumeStructuredBuffer(ref structured_type) => {
            ir::ObjectType::ConsumeStructuredBuffer(parse_structuredtype(
                structured_type,
                struct_finder,
            )?)
        }
        ast::ObjectType::Texture1D(ref data_type) => {
            ir::ObjectType::Texture1D(parse_datatype(data_type)?)
        }
        ast::ObjectType::Texture1DArray(ref data_type) => {
            ir::ObjectType::Texture1DArray(parse_datatype(data_type)?)
        }
        ast::ObjectType::Texture2D(ref data_type) => {
            ir::ObjectType::Texture2D(parse_datatype(data_type)?)
        }
        ast::ObjectType::Texture2DArray(ref data_type) => {
            ir::ObjectType::Texture2DArray(parse_datatype(data_type)?)
        }
        ast::ObjectType::Texture2DMS(ref data_type) => {
            ir::ObjectType::Texture2DMS(parse_datatype(data_type)?)
        }
        ast::ObjectType::Texture2DMSArray(ref data_type) => {
            ir::ObjectType::Texture2DMSArray(parse_datatype(data_type)?)
        }
        ast::ObjectType::Texture3D(ref data_type) => {
            ir::ObjectType::Texture3D(parse_datatype(data_type)?)
        }
        ast::ObjectType::TextureCube(ref data_type) => {
            ir::ObjectType::TextureCube(parse_datatype(data_type)?)
        }
        ast::ObjectType::TextureCubeArray(ref data_type) => {
            ir::ObjectType::TextureCubeArray(parse_datatype(data_type)?)
        }
        ast::ObjectType::RWTexture1D(ref data_type) => {
            ir::ObjectType::RWTexture1D(parse_datatype(data_type)?)
        }
        ast::ObjectType::RWTexture1DArray(ref data_type) => {
            ir::ObjectType::RWTexture1DArray(parse_datatype(data_type)?)
        }
        ast::ObjectType::RWTexture2D(ref data_type) => {
            ir::ObjectType::RWTexture2D(parse_datatype(data_type)?)
        }
        ast::ObjectType::RWTexture2DArray(ref data_type) => {
            ir::ObjectType::RWTexture2DArray(parse_datatype(data_type)?)
        }
        ast::ObjectType::RWTexture3D(ref data_type) => {
            ir::ObjectType::RWTexture3D(parse_datatype(data_type)?)
        }
        ast::ObjectType::InputPatch => ir::ObjectType::InputPatch,
        ast::ObjectType::OutputPatch => ir::ObjectType::OutputPatch,
    })
}

fn parse_typelayout(
    ty: &ast::TypeLayout,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::TypeLayout, TyperError> {
    Ok(match *ty {
        ast::TypeLayout::Void => ir::TypeLayout::void(),
        ast::TypeLayout::Scalar(ref scalar) => ir::TypeLayout::Scalar(parse_scalartype(scalar)?),
        ast::TypeLayout::Vector(ref scalar, ref x) => {
            ir::TypeLayout::Vector(parse_scalartype(scalar)?, *x)
        }
        ast::TypeLayout::Matrix(ref scalar, ref x, ref y) => {
            ir::TypeLayout::Matrix(parse_scalartype(scalar)?, *x, *y)
        }
        ast::TypeLayout::Custom(ref name) => {
            ir::TypeLayout::Struct(struct_finder.find_struct_id(name)?)
        }
        ast::TypeLayout::SamplerState => ir::TypeLayout::SamplerState,
        ast::TypeLayout::Object(ref object_type) => {
            ir::TypeLayout::Object(parse_objecttype(object_type, struct_finder)?)
        }
    })
}

pub fn parse_type(
    ty: &ast::Type,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::Type, TyperError> {
    let &ast::Type(ref tyl, ref modifier) = ty;
    Ok(ir::Type(
        parse_typelayout(tyl, struct_finder)?,
        parse_modifier(modifier),
    ))
}

fn parse_interpolationmodifier(
    im: &ast::InterpolationModifier,
) -> Result<ir::InterpolationModifier, TyperError> {
    Ok(match *im {
        ast::InterpolationModifier::NoInterpolation => ir::InterpolationModifier::NoInterpolation,
        ast::InterpolationModifier::Linear => ir::InterpolationModifier::Linear,
        ast::InterpolationModifier::Centroid => ir::InterpolationModifier::Centroid,
        ast::InterpolationModifier::NoPerspective => ir::InterpolationModifier::NoPerspective,
        ast::InterpolationModifier::Sample => ir::InterpolationModifier::Sample,
    })
}

fn parse_globalstorage(ls: &ast::GlobalStorage) -> Result<ir::GlobalStorage, TyperError> {
    Ok(match *ls {
        ast::GlobalStorage::Extern => ir::GlobalStorage::Extern,
        ast::GlobalStorage::Static => ir::GlobalStorage::Static,
        ast::GlobalStorage::GroupShared => ir::GlobalStorage::GroupShared,
    })
}

fn parse_globaltype(
    global_type: &ast::GlobalType,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::GlobalType, TyperError> {
    let ty = parse_type(&global_type.0, struct_finder)?;
    let interp = match global_type.2 {
        Some(ref im) => Some(parse_interpolationmodifier(im)?),
        None => None,
    };
    Ok(ir::GlobalType(
        ty,
        parse_globalstorage(&global_type.1)?,
        interp,
    ))
}

fn parse_inputmodifier(it: &ast::InputModifier) -> Result<ir::InputModifier, TyperError> {
    Ok(match *it {
        ast::InputModifier::In => ir::InputModifier::In,
        ast::InputModifier::Out => ir::InputModifier::Out,
        ast::InputModifier::InOut => ir::InputModifier::InOut,
    })
}

fn parse_paramtype(
    param_type: &ast::ParamType,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::ParamType, TyperError> {
    let ty = parse_type(&param_type.0, struct_finder)?;
    let interp = match param_type.2 {
        Some(ref im) => Some(parse_interpolationmodifier(im)?),
        None => None,
    };
    Ok(ir::ParamType(
        ty,
        parse_inputmodifier(&param_type.1)?,
        interp,
    ))
}

fn parse_localstorage(local_storage: &ast::LocalStorage) -> Result<ir::LocalStorage, TyperError> {
    Ok(match *local_storage {
        ast::LocalStorage::Local => ir::LocalStorage::Local,
        ast::LocalStorage::Static => ir::LocalStorage::Static,
    })
}

fn parse_localtype(
    local_type: &ast::LocalType,
    struct_finder: &dyn StructIdFinder,
) -> Result<ir::LocalType, TyperError> {
    let ty = parse_type(&local_type.0, struct_finder)?;
    let interp = match local_type.2 {
        Some(ref im) => Some(parse_interpolationmodifier(im)?),
        None => None,
    };
    Ok(ir::LocalType(
        ty,
        parse_localstorage(&local_type.1)?,
        interp,
    ))
}

/// Parse an expression
fn parse_expr(
    expr: &ast::Expression,
    context: &dyn ExpressionContext,
) -> Result<(ir::Expression, ExpressionType), TyperError> {
    // Type errors should error out here
    let (expr_pel, expr_ety) = pel::parse_expr_value_only(expr, context)?;

    let expr_ir = match expr_pel.direct_to_hir() {
        Ok(expr_ir) => {
            // If the expression is trivial then go straight from pel to hir
            expr_ir
        }
        Err(_) => {
            // Else reduce the pel expression to a rel sequence
            let expr_rel = rel::reduce(expr_pel.clone(), context.as_reduce_context())?;
            // Then turn it into a hir expression
            match rel::combine(expr_rel.clone(), &mut rel::FakeCombineContext)? {
                rel::CombinedExpression::Single(expr_ir) => {
                    // The expression is representable as a single
                    // output expression
                    expr_ir
                }
                rel::CombinedExpression::Multi(res) => {
                    // The expression requires multiple statements
                    // For example, to create local variables to pass as
                    // out parameters when the source type is different
                    panic!(
                        "rel combine multi unimplemented `{:?}` `{:?}`",
                        expr_rel, res
                    )
                }
            }
        }
    };

    let ety_res = ir::TypeParser::get_expression_type(&expr_ir, context.as_type_context());
    let ety = ety_res.expect("type unknown");

    // Ensure the result type is the same as the type the pel parser returns
    {
        // Only compare Type not ValueType - the rel parser parses self
        // contained expressions and thinks the root level only needs an rvalue,
        // so may modify the output value type (for example from array index to
        // array load)
        let ty = &ety.0;
        let expr_ty = &expr_ety.0;
        assert!(ty == expr_ty, "{:?} == {:?}: {:?}", ty, expr_ty, expr_ir);
    }

    Ok((expr_ir, ety))
}

/// Parse an expression
fn parse_expr_statement(
    expr: &ast::Expression,
    context: &dyn ExpressionContext,
) -> Result<ir::Statement, TyperError> {
    // Type errors should error out here
    let (expr_pel, expr_ety) = pel::parse_expr_value_only(expr, context)?;
    let expr_pel = expr_pel.ignore_return();

    let single = match expr_pel.direct_to_hir() {
        Ok(expr_ir) => {
            // If the expression is trivial then go straight from pel to hir
            expr_ir
        }
        Err(_) => {
            // Else reduce the pel expression to a rel sequence
            let expr_rel = rel::reduce(expr_pel.clone(), context.as_reduce_context())?;
            let expr_rel = expr_rel.ignore_value();
            // Then turn it into a hir expression
            let mut combine_context = rel::ScopeCombineContext::new();
            match rel::combine(expr_rel.clone(), &mut combine_context)? {
                rel::CombinedExpression::Single(expr_ir) => {
                    // The expression is representable as a single
                    // output expression
                    expr_ir
                }
                rel::CombinedExpression::Multi(res) => {
                    // The expression requires multiple statements
                    // For example, to create local variables to pass as
                    // out parameters when the source type is different
                    let block = combine_context.finalize(res);
                    return Ok(ir::Statement::Block(block));
                }
            }
        }
    };

    // If we only returned one expression (most expressions) then ensure the
    // result type is the same as the type the pel parser returns
    {
        let ety_res = ir::TypeParser::get_expression_type(&single, context.as_type_context());
        let ety = ety_res.expect("type unknown");
        // Only compare Type not ValueType - the rel parser parses self
        // contained expressions and thinks the root level only needs an rvalue,
        // so may modify the output value type (for example from array index to
        // array load)
        let ty = &ety.0;
        let expr_ty = &expr_ety.0;
        if ty.0 != ir::TypeLayout::Void {
            assert!(ty == expr_ty, "{:?} == {:?}: {:?}", ty, expr_ty, single);
        }
    }

    Ok(ir::Statement::Expression(single))
}

fn evaluate_constexpr_int(expr: &ast::Expression) -> Result<u64, ()> {
    Ok(match *expr {
        ast::Expression::Literal(ast::Literal::UntypedInt(i)) => i,
        ast::Expression::Literal(ast::Literal::Int(i)) => i,
        ast::Expression::Literal(ast::Literal::UInt(i)) => i,
        ast::Expression::BinaryOperation(ref op, ref left, ref right) => {
            let lc = evaluate_constexpr_int(left)?;
            let rc = evaluate_constexpr_int(right)?;
            match *op {
                ast::BinOp::Add => lc + rc,
                ast::BinOp::Subtract => lc - rc,
                ast::BinOp::Multiply => lc * rc,
                ast::BinOp::Divide => lc / rc,
                ast::BinOp::Modulus => lc % rc,
                ast::BinOp::LeftShift => lc << rc,
                ast::BinOp::RightShift => lc >> rc,
                _ => return Err(()),
            }
        }
        _ => return Err(()),
    })
}

fn apply_variable_bind(
    ty: ir::Type,
    bind: &ast::VariableBind,
    init: &Option<ast::Initializer>,
) -> Result<ir::Type, TyperError> {
    match *bind {
        ast::VariableBind::Array(ref dim) => {
            let ir::Type(layout, modifiers) = ty;

            let constant_dim = match *dim {
                Some(ref dim_expr) => match evaluate_constexpr_int(&**dim_expr) {
                    Ok(val) => val,
                    Err(()) => {
                        let p = (**dim_expr).clone();
                        return Err(TyperError::ArrayDimensionsMustBeConstantExpression(p));
                    }
                },
                None => match *init {
                    Some(ast::Initializer::Aggregate(ref exprs)) => exprs.len() as u64,
                    _ => return Err(TyperError::ArrayDimensionNotSpecified),
                },
            };

            Ok(ir::Type(
                ir::TypeLayout::Array(Box::new(layout), constant_dim),
                modifiers,
            ))
        }
        ast::VariableBind::Normal => Ok(ty),
    }
}

fn parse_initializer(
    init: &ast::Initializer,
    tyl: &ir::TypeLayout,
    context: &dyn ExpressionContext,
) -> Result<ir::Initializer, TyperError> {
    Ok(match *init {
        ast::Initializer::Expression(ref expr) => {
            let ety = ir::Type::from_layout(tyl.clone()).to_rvalue();
            let (expr_ir, expr_ty) = parse_expr(expr, context)?;
            match ImplicitConversion::find(&expr_ty, &ety) {
                Ok(rhs_cast) => ir::Initializer::Expression(rhs_cast.apply(expr_ir)),
                Err(()) => return Err(TyperError::InitializerExpressionWrongType),
            }
        }
        ast::Initializer::Aggregate(ref exprs) => {
            fn build_elements(
                ety: &ExpressionType,
                inits: &[ast::Initializer],
                context: &dyn ExpressionContext,
            ) -> Result<Vec<ir::Initializer>, TyperError> {
                let mut elements = Vec::with_capacity(inits.len());
                for init in inits {
                    let element = parse_initializer(init, &(ety.0).0, context)?;
                    elements.push(element);
                }
                Ok(elements)
            }
            match *tyl {
                ir::TypeLayout::Scalar(_) => {
                    if exprs.len() as u32 != 1 {
                        return Err(TyperError::InitializerAggregateWrongDimension);
                    }

                    // Reparse as if it was a single expression instead of a 1 element aggregate
                    // Meaning '{ x }' is read as if it were 'x'
                    // Will also reduce '{{ x }}' to 'x'
                    parse_initializer(&exprs[0], tyl, context)?
                }
                ir::TypeLayout::Vector(ref scalar, ref dim) => {
                    if exprs.len() as u32 != *dim {
                        return Err(TyperError::InitializerAggregateWrongDimension);
                    }

                    let ety = ir::Type::from_scalar(scalar.clone()).to_rvalue();
                    let elements = build_elements(&ety, exprs, context)?;

                    ir::Initializer::Aggregate(elements)
                }
                ir::TypeLayout::Array(ref inner, ref dim) => {
                    if exprs.len() as u64 != *dim {
                        return Err(TyperError::InitializerAggregateWrongDimension);
                    }

                    let ety = ir::Type::from_layout(*inner.clone()).to_rvalue();
                    let elements = build_elements(&ety, exprs, context)?;

                    ir::Initializer::Aggregate(elements)
                }
                _ => return Err(TyperError::InitializerAggregateDoesNotMatchType),
            }
        }
    })
}

fn parse_initializer_opt(
    init_opt: &Option<ast::Initializer>,
    tyl: &ir::TypeLayout,
    context: &dyn ExpressionContext,
) -> Result<Option<ir::Initializer>, TyperError> {
    Ok(match *init_opt {
        Some(ref init) => Some(parse_initializer(init, tyl, context)?),
        None => None,
    })
}

fn parse_vardef(
    ast: &ast::VarDef,
    context: ScopeContext,
) -> Result<(Vec<ir::VarDef>, ScopeContext), TyperError> {
    let base_type = parse_localtype(&ast.local_type, &context)?;

    // Build multiple output VarDefs for each variable inside the source VarDef
    let mut context = context;
    let mut vardefs = vec![];
    for local_variable in &ast.defs {
        // Get variable name
        let var_name = &local_variable.name.clone();

        // Build type from ast type + bind
        let ir::LocalType(lty, ls, interp) = base_type.clone();
        let bind = &local_variable.bind;
        let lv_tyl = apply_variable_bind(lty, bind, &local_variable.init)?;
        let lv_type = ir::LocalType(lv_tyl, ls, interp);

        // Parse the initializer
        let var_init = parse_initializer_opt(&local_variable.init, &(lv_type.0).0, &context)?;

        // Register the variable
        let var_id = context.insert_variable(var_name.clone(), lv_type.0.clone())?;

        // Add the variables creation node
        vardefs.push(ir::VarDef {
            id: var_id,
            local_type: lv_type,
            init: var_init,
        });
    }

    Ok((vardefs, context))
}

fn parse_for_init(
    ast: &ast::InitStatement,
    context: ScopeContext,
) -> Result<(ir::ForInit, ScopeContext), TyperError> {
    match *ast {
        ast::InitStatement::Empty => Ok((ir::ForInit::Empty, context)),
        ast::InitStatement::Expression(ref expr) => {
            let expr_ir = parse_expr(expr, &context)?.0;
            Ok((ir::ForInit::Expression(expr_ir), context))
        }
        ast::InitStatement::Declaration(ref vd) => {
            let (vd_ir, context) = parse_vardef(vd, context)?;
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
fn parse_scopeblock(
    ast: &ast::Statement,
    block_context: ScopeContext,
) -> Result<ir::ScopeBlock, TyperError> {
    match *ast {
        ast::Statement::Block(ref statement_vec) => {
            let (statements, block_context) = parse_statement_vec(statement_vec, block_context)?;
            Ok(ir::ScopeBlock(statements, block_context.destruct()))
        }
        _ => {
            let (ir_statements, block_context) = parse_statement(ast, block_context)?;
            Ok(ir::ScopeBlock(ir_statements, block_context.destruct()))
        }
    }
}

fn parse_statement(
    ast: &ast::Statement,
    context: ScopeContext,
) -> Result<(Vec<ir::Statement>, ScopeContext), TyperError> {
    match ast {
        &ast::Statement::Empty => Ok((vec![], context)),
        &ast::Statement::Expression(ref expr) => {
            let statement = parse_expr_statement(expr, &context)?;
            Ok((vec![statement], context))
        }
        &ast::Statement::Var(ref vd) => {
            let (vd_ir, context) = parse_vardef(vd, context)?;
            let vars = vd_ir
                .into_iter()
                .map(|v| ir::Statement::Var(v))
                .collect::<Vec<_>>();
            Ok((vars, context))
        }
        &ast::Statement::Block(ref statement_vec) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (statements, scoped_context) = parse_statement_vec(statement_vec, scoped_context)?;
            let decls = scoped_context.destruct();
            Ok((
                vec![ir::Statement::Block(ir::ScopeBlock(statements, decls))],
                context,
            ))
        }
        &ast::Statement::If(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let cond_ir = parse_expr(cond, &scoped_context)?.0;
            let scope_block = parse_scopeblock(statement, scoped_context)?;
            Ok((vec![ir::Statement::If(cond_ir, scope_block)], context))
        }
        &ast::Statement::IfElse(ref cond, ref true_statement, ref false_statement) => {
            let cond_ir = parse_expr(cond, &context)?.0;
            let scoped_context = ScopeContext::from_scope(&context);
            let scope_block = parse_scopeblock(true_statement, scoped_context)?;
            let scoped_context = ScopeContext::from_scope(&context);
            let else_block = parse_scopeblock(false_statement, scoped_context)?;
            Ok((
                vec![ir::Statement::IfElse(cond_ir, scope_block, else_block)],
                context,
            ))
        }
        &ast::Statement::For(ref init, ref cond, ref iter, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (init_ir, scoped_context) = parse_for_init(init, scoped_context)?;
            let cond_ir = parse_expr(cond, &scoped_context)?.0;
            let iter_ir = parse_expr(iter, &scoped_context)?.0;
            let scope_block = parse_scopeblock(statement, scoped_context)?;
            Ok((
                vec![ir::Statement::For(init_ir, cond_ir, iter_ir, scope_block)],
                context,
            ))
        }
        &ast::Statement::While(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let cond_ir = parse_expr(cond, &scoped_context)?.0;
            let scope_block = parse_scopeblock(statement, scoped_context)?;
            Ok((vec![ir::Statement::While(cond_ir, scope_block)], context))
        }
        &ast::Statement::Break => Ok((vec![ir::Statement::Break], context)),
        &ast::Statement::Continue => Ok((vec![ir::Statement::Continue], context)),
        &ast::Statement::Return(ref expr) => {
            let (expr_ir, expr_ty) = parse_expr(expr, &context)?;
            match ImplicitConversion::find(&expr_ty, &context.get_return_type().to_rvalue()) {
                Ok(rhs_cast) => Ok((
                    vec![ir::Statement::Return(rhs_cast.apply(expr_ir))],
                    context,
                )),
                Err(()) => return Err(TyperError::WrongTypeInReturnStatement),
            }
        }
    }
}

fn parse_statement_vec(
    ast: &[ast::Statement],
    context: ScopeContext,
) -> Result<(Vec<ir::Statement>, ScopeContext), TyperError> {
    let mut context = context;
    let mut body_ir = vec![];
    for statement_ast in ast {
        let (mut statement_ir_vec, next_context) = parse_statement(&statement_ast, context)?;
        body_ir.append(&mut statement_ir_vec);
        context = next_context;
    }
    Ok((body_ir, context))
}

fn parse_rootdefinition_struct(
    sd: &ast::StructDefinition,
    mut context: GlobalContext,
) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let mut members = vec![];
    let mut member_map = HashMap::new();
    for ast_member in &sd.members {
        let base_type = parse_type(&ast_member.ty, &context)?;
        for def in &ast_member.defs {
            let name = def.name.clone();
            let ty = apply_variable_bind(base_type.clone(), &def.bind, &None)?;
            member_map.insert(name.clone(), ty.clone());
            members.push(ir::StructMember {
                name: name,
                typename: ty,
            });
        }
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

fn parse_rootdefinition_constantbuffer(
    cb: &ast::ConstantBuffer,
    mut context: GlobalContext,
) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let cb_name = cb.name.clone();
    let mut members = vec![];
    let mut members_map = HashMap::new();
    for member in &cb.members {
        let base_type = parse_type(&member.ty, &context)?;
        for def in &member.defs {
            let var_name = def.name.clone();
            let var_offset = def.offset.clone();
            let var_type = apply_variable_bind(base_type.clone(), &def.bind, &None)?;
            members_map.insert(var_name.clone(), var_type.clone());
            members.push(ir::ConstantVariable {
                name: var_name,
                typename: var_type,
                offset: var_offset,
            });
        }
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
            context
                .global_slots_constants
                .push((slot, cb_ir.id.clone()));
        }
        None => {}
    }
    Ok((ir::RootDefinition::ConstantBuffer(cb_ir), context))
}

fn parse_rootdefinition_globalvariable(
    gv: &ast::GlobalVariable,
    mut context: GlobalContext,
) -> Result<(Vec<ir::RootDefinition>, GlobalContext), TyperError> {
    let var_type = parse_globaltype(&gv.global_type, &context)?;

    let mut defs = vec![];

    for def in &gv.defs {
        // Resolve type
        let ir::GlobalType(lty, gs, interp) = var_type.clone();
        let bind = &def.bind;
        let gv_tyl = apply_variable_bind(lty, bind, &def.init)?;
        let gv_type = ir::GlobalType(gv_tyl, gs, interp);

        // Insert variable
        let var_name = def.name.clone();
        let input_type = gv_type.0.clone();
        let var_id = context.insert_global(var_name.clone(), input_type.clone())?;

        let var_init = parse_initializer_opt(&def.init, &input_type.0, &context)?;
        let gv_ir = ir::GlobalVariable {
            id: var_id,
            global_type: gv_type,
            init: var_init,
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
            Some(ast::GlobalSlot::SamplerSlot(slot)) => {
                context.global_slots_s.push((slot, entry));
            }
            None => {}
        }

        defs.push(ir::RootDefinition::GlobalVariable(gv_ir));
    }

    Ok((defs, context))
}

fn parse_rootdefinition_function(
    fd: &ast::FunctionDefinition,
    mut context: GlobalContext,
) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let return_type = parse_type(&fd.returntype, &context)?;
    // Set the return type of the current function (for return statement parsing)
    assert_eq!(context.current_return_type, None);
    context.current_return_type = Some(return_type.clone());

    let mut scoped_context = ScopeContext::from_global(&context);
    let func_params = {
        let mut vec = vec![];
        for param in &fd.params {
            let var_type = parse_paramtype(&param.param_type, &context)?;
            let var_id = scoped_context.insert_variable(param.name.clone(), var_type.0.clone())?;
            vec.push(ir::FunctionParam {
                id: var_id,
                param_type: var_type,
            });
        }
        vec
    };
    let (body_ir, scoped_context) = parse_statement_vec(&fd.body, scoped_context)?;
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
    let func_type = FunctionOverload(
        FunctionName::User(fd_ir.id),
        fd_ir.returntype.clone(),
        fd_ir.params.iter().map(|p| p.param_type.clone()).collect(),
    );
    context.insert_function(fd.name.clone(), func_type)?;
    Ok((ir::RootDefinition::Function(fd_ir), context))
}

fn parse_rootdefinition_kernel(
    fd: &ast::FunctionDefinition,
    context: GlobalContext,
) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let mut scoped_context = ScopeContext::from_global(&context);
    let kernel_params = {
        let mut vec = vec![];
        for param in &fd.params {
            let var_type = parse_paramtype(&param.param_type, &context)?;
            let var_id = scoped_context.insert_variable(param.name.clone(), var_type.0.clone())?;
            vec.push(ir::KernelParam(
                var_id,
                match &param.semantic {
                    &Some(ast::Semantic::DispatchThreadId) => ir::KernelSemantic::DispatchThreadId,
                    &Some(ast::Semantic::GroupId) => ir::KernelSemantic::GroupId,
                    &Some(ast::Semantic::GroupIndex) => ir::KernelSemantic::GroupIndex,
                    &Some(ast::Semantic::GroupThreadId) => ir::KernelSemantic::GroupThreadId,
                    &None => return Err(TyperError::KernelHasParamWithoutSemantic(param.clone())),
                },
            ));
        }
        vec
    };
    let (body_ir, scoped_context) = parse_statement_vec(&fd.body, scoped_context)?;
    let decls = scoped_context.destruct();

    fn find_dispatch_dimensions(
        attributes: &[ast::FunctionAttribute],
    ) -> Result<ir::Dimension, TyperError> {
        for attribute in attributes {
            match *attribute {
                ast::FunctionAttribute::NumThreads(ref x, ref y, ref z) => {
                    let eval_x = match evaluate_constexpr_int(x) {
                        Ok(val) => val,
                        Err(()) => {
                            return Err(TyperError::KernelDispatchDimensionMustBeConstantExpression)
                        }
                    };
                    let eval_y = match evaluate_constexpr_int(y) {
                        Ok(val) => val,
                        Err(()) => {
                            return Err(TyperError::KernelDispatchDimensionMustBeConstantExpression)
                        }
                    };
                    let eval_z = match evaluate_constexpr_int(z) {
                        Ok(val) => val,
                        Err(()) => {
                            return Err(TyperError::KernelDispatchDimensionMustBeConstantExpression)
                        }
                    };
                    return Ok(ir::Dimension(eval_x, eval_y, eval_z));
                }
            };
        }
        Err(TyperError::KernelHasNoDispatchDimensions)
    }
    let kernel = ir::Kernel {
        group_dimensions: find_dispatch_dimensions(&fd.attributes[..])?,
        params: kernel_params,
        scope_block: ir::ScopeBlock(body_ir, decls),
    };
    Ok((ir::RootDefinition::Kernel(kernel), context))
}

fn parse_rootdefinition(
    ast: &ast::RootDefinition,
    context: GlobalContext,
    entry_point: &str,
) -> Result<(Vec<ir::RootDefinition>, GlobalContext), TyperError> {
    match ast {
        &ast::RootDefinition::Struct(ref sd) => {
            let (def, context) = parse_rootdefinition_struct(sd, context)?;
            Ok((vec![def], context))
        }
        &ast::RootDefinition::SamplerState => unimplemented!(),
        &ast::RootDefinition::ConstantBuffer(ref cb) => {
            let (def, context) = parse_rootdefinition_constantbuffer(cb, context)?;
            Ok((vec![def], context))
        }
        &ast::RootDefinition::GlobalVariable(ref gv) => {
            parse_rootdefinition_globalvariable(gv, context)
        }
        &ast::RootDefinition::Function(ref fd) if fd.name == entry_point => {
            let (def, context) = parse_rootdefinition_kernel(fd, context)?;
            Ok((vec![def], context))
        }
        &ast::RootDefinition::Function(ref fd) => {
            let (def, context) = parse_rootdefinition_function(fd, context)?;
            Ok((vec![def], context))
        }
    }
}

pub fn typeparse(ast: &ast::Module) -> Result<ir::Module, TyperError> {
    use slp_lang_hir::globals_analysis::GlobalUsage;

    let mut context = GlobalContext::new();

    let mut root_definitions = vec![];

    for def in &ast.root_definitions {
        let (mut def_ir, next_context) =
            parse_rootdefinition(&def, context, &ast.entry_point.clone())?;
        root_definitions.append(&mut def_ir);
        context = next_context;
    }

    let analysis = GlobalUsage::analyse(&root_definitions);

    let root_definitions = root_definitions
        .into_iter()
        .filter(|def| match *def {
            ir::RootDefinition::GlobalVariable(ref gv) => analysis.kernel.globals.contains(&gv.id),
            ir::RootDefinition::ConstantBuffer(ref cb) => analysis.kernel.cbuffers.contains(&cb.id),
            ir::RootDefinition::Function(ref func) => analysis.kernel.functions.contains(&func.id),
            _ => true,
        })
        .collect::<Vec<_>>();

    // Gather remaining global declaration names
    let global_declarations = root_definitions.iter().fold(
        ir::GlobalDeclarations {
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
                            panic!("constant buffer name does not exist");
                        }
                    }
                }
                ir::RootDefinition::GlobalVariable(ref gv) => {
                    match context.global_names.get(&gv.id) {
                        Some(name) => {
                            map.globals.insert(gv.id, name.clone());
                        }
                        None => {
                            panic!("global variable name does not exist");
                        }
                    }
                }
                ir::RootDefinition::Function(ref func) => {
                    match context.function_names.get(&func.id) {
                        Some(name) => {
                            map.functions.insert(func.id, name.clone());
                        }
                        None => {
                            panic!("function name does not exist");
                        }
                    }
                }
                _ => {}
            }
            map
        },
    );

    // Resolve used globals into SRV list
    let mut global_table_r = HashMap::new();
    for (slot, entry) in context.global_slots_r {
        if global_declarations.globals.contains_key(&entry.id) {
            let error_id = entry.id.clone();
            match global_table_r.insert(slot, entry) {
                Some(currently_used_by) => {
                    let id = currently_used_by.id.clone();
                    return Err(TyperError::ReadResourceSlotAlreadyUsed(id, error_id));
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
                Some(currently_used_by) => {
                    let id = currently_used_by.id.clone();
                    return Err(TyperError::ReadWriteResourceSlotAlreadyUsed(id, error_id));
                }
                None => {}
            }
        }
    }

    // Resolve used samplers into sampler list
    let mut global_table_s = HashMap::new();
    for (slot, entry) in context.global_slots_s {
        if global_declarations.globals.contains_key(&entry.id) {
            let error_id = entry.id.clone();
            match global_table_s.insert(slot, entry) {
                Some(currently_used_by) => {
                    let id = currently_used_by.id.clone();
                    return Err(TyperError::SamplerResourceSlotAlreadyUsed(id, error_id));
                }
                None => {}
            }
        }
    }

    // Resolve used constant buffers into constant buffer list
    let mut global_table_constants = HashMap::new();
    for (slot, cb_id) in context.global_slots_constants {
        if global_declarations.constants.contains_key(&cb_id) {
            let error_id = cb_id.clone();
            match global_table_constants.insert(slot, cb_id) {
                Some(currently_used_by) => {
                    let by = currently_used_by.clone();
                    return Err(TyperError::ConstantSlotAlreadyUsed(by, error_id));
                }
                None => {}
            }
        }
    }

    // Make the table describing all global bindings
    let global_table = ir::GlobalTable {
        r_resources: global_table_r,
        rw_resources: global_table_rw,
        samplers: global_table_s,
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
                global_type: ast::Type::from_object(ast::ObjectType::Buffer(ast::DataType(
                    ast::DataLayout::Scalar(ast::ScalarType::Int),
                    ast::TypeModifier::default(),
                )))
                .into(),
                defs: vec![ast::GlobalVariableName {
                    name: "g_myInBuffer".to_string(),
                    bind: ast::VariableBind::Normal,
                    slot: Some(ast::GlobalSlot::ReadSlot(0)),
                    init: None,
                }],
            }),
            ast::RootDefinition::GlobalVariable(ast::GlobalVariable {
                global_type: ast::GlobalType(
                    ast::Type(
                        ast::TypeLayout::from_scalar(ast::ScalarType::Int),
                        ast::TypeModifier {
                            is_const: true,
                            ..ast::TypeModifier::default()
                        },
                    ),
                    ast::GlobalStorage::Static,
                    None,
                ),
                defs: vec![ast::GlobalVariableName {
                    name: "g_myFour".to_string(),
                    bind: ast::VariableBind::Normal,
                    slot: None,
                    init: Some(ast::Initializer::Expression(Located::none(
                        ast::Expression::Literal(ast::Literal::UntypedInt(4)),
                    ))),
                }],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::void(),
                params: vec![ast::FunctionParam {
                    name: "x".to_string(),
                    param_type: ast::Type::uint().into(),
                    semantic: None,
                }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::void(),
                params: vec![ast::FunctionParam {
                    name: "x".to_string(),
                    param_type: ast::Type::float().into(),
                    semantic: None,
                }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "outFunc".to_string(),
                returntype: ast::Type::void(),
                params: vec![ast::FunctionParam {
                    name: "x".to_string(),
                    param_type: ast::ParamType(ast::Type::float(), ast::InputModifier::Out, None),
                    semantic: None,
                }],
                body: vec![
                    ast::Statement::Var(ast::VarDef::one(
                        "local_static",
                        ast::LocalType(ast::Type::uint(), ast::LocalStorage::Static, None),
                    )),
                    ast::Statement::Expression(Located::loc(
                        1,
                        1,
                        ast::Expression::BinaryOperation(
                            ast::BinOp::Assignment,
                            Box::new(Located::none(ast::Expression::Variable("x".to_string()))),
                            Box::new(Located::none(ast::Expression::Literal(
                                ast::Literal::Float(1.5f32),
                            ))),
                        ),
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
                    ast::Statement::Var(ast::VarDef::one("a", ast::Type::uint().into())),
                    ast::Statement::Var(ast::VarDef::one("b", ast::Type::uint().into())),
                    ast::Statement::Expression(Located::none(ast::Expression::BinaryOperation(
                        ast::BinOp::Assignment,
                        Box::new(Located::none(ast::Expression::Variable("a".to_string()))),
                        Box::new(Located::none(ast::Expression::Variable("b".to_string()))),
                    ))),
                    ast::Statement::If(
                        Located::none(ast::Expression::Variable("b".to_string())),
                        Box::new(ast::Statement::Empty),
                    ),
                    ast::Statement::Expression(Located::none(ast::Expression::BinaryOperation(
                        ast::BinOp::Assignment,
                        Box::new(Located::none(ast::Expression::ArraySubscript(
                            Box::new(Located::none(ast::Expression::Variable(
                                "g_myInBuffer".to_string(),
                            ))),
                            Box::new(Located::none(ast::Expression::Literal(ast::Literal::Int(
                                0,
                            )))),
                        ))),
                        Box::new(Located::none(ast::Expression::Literal(ast::Literal::Int(
                            4,
                        )))),
                    ))),
                    ast::Statement::Expression(Located::none(ast::Expression::Call(
                        Box::new(Located::none(ast::Expression::Variable(
                            "myFunc".to_string(),
                        ))),
                        vec![Located::none(ast::Expression::Variable("b".to_string()))],
                    ))),
                    ast::Statement::Var(ast::VarDef::one("testOut", ast::Type::float().into())),
                    ast::Statement::Var(ast::VarDef {
                        local_type: ast::Type::from_layout(ast::TypeLayout::float()).into(),
                        defs: vec![ast::LocalVariableName {
                            name: "x".to_string(),
                            bind: ast::VariableBind::Array(Some(Located::none(
                                ast::Expression::Literal(ast::Literal::UntypedInt(3)),
                            ))),
                            init: None,
                        }],
                    }),
                    ast::Statement::Expression(Located::none(ast::Expression::Call(
                        Box::new(Located::none(ast::Expression::Variable(
                            "outFunc".to_string(),
                        ))),
                        vec![Located::none(ast::Expression::Variable(
                            "testOut".to_string(),
                        ))],
                    ))),
                ],
                attributes: vec![ast::FunctionAttribute::numthreads(8, 8, 1)],
            }),
        ],
    };
    let res = typeparse(&module);
    assert!(res.is_ok(), "{:?}", res);

    let static_global_test = ast::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            ast::RootDefinition::GlobalVariable(ast::GlobalVariable {
                global_type: ast::GlobalType(
                    ast::Type(
                        ast::TypeLayout::from_scalar(ast::ScalarType::Int),
                        ast::TypeModifier {
                            is_const: true,
                            ..ast::TypeModifier::default()
                        },
                    ),
                    ast::GlobalStorage::Static,
                    None,
                ),
                defs: vec![ast::GlobalVariableName {
                    name: "g_myFour".to_string(),
                    bind: ast::VariableBind::Normal,
                    slot: None,
                    init: Some(ast::Initializer::Expression(Located::none(
                        ast::Expression::Literal(ast::Literal::UntypedInt(4)),
                    ))),
                }],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: ast::Type::void(),
                params: vec![],
                body: vec![ast::Statement::Expression(Located::none(
                    ast::Expression::Variable("g_myFour".to_string()),
                ))],
                attributes: vec![ast::FunctionAttribute::numthreads(8, 8, 1)],
            }),
        ],
    };
    let static_global_result = typeparse(&static_global_test);
    let static_global_expected = Ok(ir::Module {
        entry_point: "CSMAIN".to_string(),
        global_table: ir::GlobalTable::default(),
        global_declarations: ir::GlobalDeclarations {
            functions: HashMap::new(),
            globals: {
                let mut map = HashMap::new();
                map.insert(ir::GlobalId(0), "g_myFour".to_string());
                map
            },
            structs: HashMap::new(),
            constants: HashMap::new(),
        },
        root_definitions: vec![
            ir::RootDefinition::GlobalVariable(ir::GlobalVariable {
                id: ir::GlobalId(0),
                global_type: ir::GlobalType(
                    ir::Type(
                        ir::TypeLayout::from_scalar(ir::ScalarType::Int),
                        ir::TypeModifier {
                            is_const: true,
                            ..ir::TypeModifier::default()
                        },
                    ),
                    ir::GlobalStorage::Static,
                    None,
                ),
                init: Some(ir::Initializer::Expression(ir::Expression::Cast(
                    ir::Type::int(),
                    Box::new(ir::Expression::Literal(ir::Literal::UntypedInt(4))),
                ))),
            }),
            ir::RootDefinition::Kernel(ir::Kernel {
                group_dimensions: ir::Dimension(8, 8, 1),
                params: vec![],
                scope_block: ir::ScopeBlock(
                    vec![ir::Statement::Expression(ir::Expression::Global(
                        ir::GlobalId(0),
                    ))],
                    ir::ScopedDeclarations {
                        variables: HashMap::new(),
                    },
                ),
            }),
        ],
    });
    assert_eq!(static_global_result, static_global_expected);
}
