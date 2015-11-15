
use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use super::ir;
use super::ast;

#[derive(PartialEq, Debug, Clone)]
pub enum TyperError {
    Unimplemented,

    ValueAlreadyDefined(String, ParseType, ParseType),
    StructAlreadyDefined(String),

    ConstantSlotAlreadyUsed(String, String),
    ReadResourceSlotAlreadyUsed(ir::VariableId, ir::VariableId),
    ReadWriteResourceSlotAlreadyUsed(ir::VariableId, ir::VariableId),

    UnknownIdentifier(String),
    UnknownType(ParseType),

    TypeDoesNotHaveMembers(ParseType),
    UnknownTypeMember(ParseType, String),

    ArrayIndexingNonArrayType,
    ArraySubscriptIndexNotInteger,

    CallOnNonFunction,

    FunctionPassedToAnotherFunction(ParseType, ParseType),
    FunctionArgumentTypeMismatch(UnresolvedFunction, ParamArray),
    MethodArgumentTypeMismatch(UnresolvedMethod, ParamArray),

    UnaryOperationWrongTypes(ir::UnaryOp, ParseType),
    BinaryOperationWrongTypes(ir::BinOp, ParseType, ParseType),

    InvalidCast(ParseType, ParseType),
    FunctionTypeInInitExpression,
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
pub enum IntrinsicFunction {
    Float4,
}

#[derive(PartialEq, Debug, Clone)]
pub enum FunctionName {
    Intrinsic(IntrinsicFunction),
    User(ir::FunctionId),
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionOverload(pub FunctionName, pub ReturnType, pub ParamArray);

#[derive(PartialEq, Debug, Clone)]
pub struct ResolvedFunction(pub FunctionOverload);

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedFunction(pub String, pub Vec<FunctionOverload>);

#[derive(PartialEq, Debug, Clone)]
pub enum IntrinsicMethod {
    BufferLoad,
    StructuredBufferLoad,
}

#[derive(PartialEq, Debug, Clone)]
pub enum MethodName {
    Intrinsic(IntrinsicMethod),
    User(ir::FunctionId),
}

#[derive(PartialEq, Debug, Clone)]
pub struct MethodOverload(pub MethodName, pub ReturnType, pub ParamArray);

#[derive(PartialEq, Debug, Clone)]
pub struct ResolvedMethod(pub ClassType, pub MethodOverload, ir::Expression);

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedMethod(pub String, pub ClassType, pub Vec<MethodOverload>, ir::Expression);

#[derive(PartialEq, Debug, Clone)]
pub enum ParseType {
    Value(ir::Type),
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

            TyperError::ConstantSlotAlreadyUsed(_, _) => "global constant slot already used",
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
            TyperError::MethodArgumentTypeMismatch(_, _) => "wrong paramters given to method",

            TyperError::UnaryOperationWrongTypes(_, _) => "operation does not support the given types",
            TyperError::BinaryOperationWrongTypes(_, _, _) => "operation does not support the given types",

            TyperError::InvalidCast(_, _) => "invalid cast",
            TyperError::FunctionTypeInInitExpression => "function not called",
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
struct GlobalContext {
    structs: HashMap<String, ir::StructDefinition>,

    functions: HashMap<String, UnresolvedFunction>,
    next_free_function_id: ir::FunctionId,

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

impl UnresolvedFunction {
    pub fn get_name(&self) -> String { self.0.clone() }
}

impl UnresolvedMethod {
    pub fn get_name(&self) -> String { self.0.clone() }
}

impl TypedExpression {
    fn to_parsetype(&self) -> ParseType {
        match self {
            &TypedExpression::Value(_, ref ty) => ParseType::Value(ty.clone()),
            &TypedExpression::Function(ref unresolved) => ParseType::Function(unresolved.clone()),
            &TypedExpression::Method(ref unresolved) => ParseType::Method(unresolved.clone()),
        }
    }
}

impl VariableBlock {
    fn new() -> VariableBlock {
        VariableBlock { variables: HashMap::new(), next_free_variable_id: 0 }
    }

    fn insert_variable(&mut self, name: String, typename: ir::Type) -> Result<ir::VariableId, TyperError> {
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name, ParseType::Value(ty.clone()), ParseType::Value(typename)))
        };
        match self.variables.entry(name.clone()) {
            Entry::Occupied(occupied) => Err(TyperError::ValueAlreadyDefined(name, ParseType::Value(occupied.get().0.clone()), ParseType::Value(typename))),
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

impl GlobalContext {
    fn new() -> GlobalContext {
        GlobalContext { structs: HashMap::new(), functions: get_intrinsics(), next_free_function_id: 0, variables: VariableBlock::new() }
    }

    pub fn insert_function(&mut self, name: String, function_type: FunctionOverload) -> Result<(), TyperError> {
        // Error if a variable of the same name already exists
        if let Some(&(ref ty, _)) = self.has_variable(&name) {
            return Err(TyperError::ValueAlreadyDefined(name, ParseType::Value(ty.clone()), ParseType::Unknown))
        };
        // Try to add the function
        match self.functions.entry(name.clone()) {
            Entry::Occupied(mut occupied) => {
                // Fail if the overload already exists
                for &FunctionOverload(_, _, ref args) in &occupied.get().1 {
                    if *args == function_type.2 {
                        return Err(TyperError::ValueAlreadyDefined(name, ParseType::Unknown, ParseType::Unknown))
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
            Some(tys) => Ok(TypedExpression::Function(tys.clone())),
            None => {
                match self.variables.find_variable(name, scopes_up) {
                    Some(tys) => Ok(tys),
                    None => Err(TyperError::UnknownIdentifier(name.clone()))
                }
            },
        }
    }

    fn make_function_id(&mut self) -> ir::FunctionId {
        let value = self.next_free_function_id;
        self.next_free_function_id = self.next_free_function_id + 1;
        value
    }

    fn insert_variable(&mut self, name: String, typename: ir::Type) -> Result<ir::VariableId, TyperError> {
        self.variables.insert_variable(name, typename)
    }

    fn has_variable(&self, name: &String) -> Option<&(ir::Type, ir::VariableId)> {
        self.variables.has_variable(name)
    }

    #[allow(dead_code)]
    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct(&self, name: &String) -> Option<&ir::StructDefinition> {
        self.structs.get(name)
    }

    fn destruct(self) -> ir::GlobalDeclarations {
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
        }
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
        self.variables.insert_variable(name, typename)
    }

    fn find_variable(&self, name: &String) -> Result<TypedExpression, TyperError> {
        self.find_variable_recur(name, 0)
    }

    fn find_struct(&self, name: &String) -> Option<&ir::StructDefinition> {
        self.parent.find_struct(name)
    }

}

impl Context {

    fn find_variable_recur(&self, name: &String, scopes_up: u32) -> Result<TypedExpression, TyperError> {
        match *self {
            Context::Global(ref global) => global.find_variable_recur(name, scopes_up),
            Context::Scope(ref scope) => scope.find_variable_recur(name, scopes_up),
        }
    }

    fn find_struct(&self, name: &String) -> Option<&ir::StructDefinition> {
        match *self {
            Context::Global(ref global) => global.find_struct(name),
            Context::Scope(ref scope) => scope.find_struct(name),
        }
    }
}

fn get_intrinsics() -> HashMap<String, UnresolvedFunction> {
    let mut map = HashMap::new();

    fn t_float() -> ir::Type { ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::Float))) };
    fn t_float4() -> ir::Type { ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Vector(ir::ScalarType::Float, 4))) };

    let (float4_str, float4_func) = ("float4".to_string(), FunctionName::Intrinsic(IntrinsicFunction::Float4));
    map.insert(float4_str.clone(), UnresolvedFunction(float4_str.clone(), vec![
        FunctionOverload(float4_func, t_float4(), vec![t_float(), t_float(), t_float(), t_float()])
    ]));

    map
}

fn find_function_type(function: UnresolvedFunction, actual_params: ParamArray) -> Result<ResolvedFunction, TyperError> {
    for overload in &function.1 {
        if overload.2 == actual_params {
            return Ok(ResolvedFunction(overload.clone()))
        };
    };
    Err(TyperError::FunctionArgumentTypeMismatch(function, actual_params))
}

fn find_method_type(method: UnresolvedMethod, actual_params: ParamArray) -> Result<ResolvedMethod, TyperError> {
    for overload in &method.2 {
        if overload.2 == actual_params {
            return Ok(ResolvedMethod(method.1, overload.clone(), method.3))
        };
    };
    Err(TyperError::MethodArgumentTypeMismatch(method, actual_params))
}

fn write_function(function: ResolvedFunction, param_values: Vec<ir::Expression>) -> Result<TypedExpression, TyperError> {
    let ResolvedFunction(FunctionOverload(name, return_type, _)) = function;
    let intrinsic = match name {
        FunctionName::Intrinsic(intrinsic) => intrinsic,
        FunctionName::User(id) => {
            return Ok(TypedExpression::Value(ir::Expression::Call(Box::new(ir::Expression::Function(id)), param_values), return_type))
        },
    };
    Ok(TypedExpression::Value(match intrinsic {
        IntrinsicFunction::Float4 => {
            assert_eq!(param_values.len(), 4);
            ir::Expression::Intrinsic(ir::Intrinsic::Float4(
                Box::new(param_values[0].clone()),
                Box::new(param_values[1].clone()),
                Box::new(param_values[2].clone()),
                Box::new(param_values[3].clone())
            ))
        },
    }, return_type))
}

fn write_method(method: ResolvedMethod, param_values: Vec<ir::Expression>) -> Result<TypedExpression, TyperError> {
    let ResolvedMethod(_, MethodOverload(name, return_type, _), object_ir) = method;
    let intrinsic = match name {
        MethodName::Intrinsic(intrinsic) => intrinsic,
        _ => unreachable!(),
    };
    Ok(TypedExpression::Value(match intrinsic {
        IntrinsicMethod::BufferLoad => {
            assert_eq!(param_values.len(), 1);
            ir::Expression::Intrinsic(ir::Intrinsic::BufferLoad(
                Box::new(object_ir),
                Box::new(param_values[0].clone())
            ))
        },
        IntrinsicMethod::StructuredBufferLoad => {
            assert_eq!(param_values.len(), 1);
            ir::Expression::Intrinsic(ir::Intrinsic::StructuredBufferLoad(
                Box::new(object_ir),
                Box::new(param_values[0].clone())
            ))
        },
    }, return_type))
}

fn parse_literal(ast: &ast::Literal) -> Result<TypedExpression, TyperError> {
    match ast {
        &ast::Literal::Int(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Int(i)), ir::Type::int())),
        &ast::Literal::Uint(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Uint(i)), ir::Type::uint())),
        &ast::Literal::Long(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Long(i)), ir::Type::long())),
        &ast::Literal::Half(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Half(f)), ir::Type::float())),
        &ast::Literal::Float(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Float(f)), ir::Type::float())),
        &ast::Literal::Double(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Double(f)), ir::Type::double())),
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
                _ => Err(TyperError::UnaryOperationWrongTypes(op.clone(), ParseType::Unknown)),
            }
        },
        &ast::Expression::BinaryOperation(ref op, ref lhs, ref rhs) => {
            let lhs_texp = try!(parse_expr(lhs, context));
            let rhs_texp = try!(parse_expr(rhs, context));
            let lhs_pt = lhs_texp.to_parsetype();
            let rhs_pt = rhs_texp.to_parsetype();
            let (lhs_ir, lhs_type) = match lhs_texp {
                TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                _ => return Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
            };
            let (rhs_ir, rhs_type) = match rhs_texp {
                TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                _ => return Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
            };
            if lhs_type != rhs_type {
                Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt))
            } else {
                Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(), Box::new(lhs_ir), Box::new(rhs_ir)), lhs_type.clone()))
            }
        },
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
            match subscript_ty {
                ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::Int))) |
                ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::UInt))) => { },
                _ => return Err(TyperError::ArraySubscriptIndexNotInteger),
            };
            Ok(TypedExpression::Value(ir::Expression::ArraySubscript(Box::new(array_ir), Box::new(subscript_ir)), indexed_type))
        },
        &ast::Expression::Member(ref composite, ref member) => {
            let composite_texp = try!(parse_expr(composite, context));
            let composite_pt = composite_texp.to_parsetype();
            let (composite_ir, composite_ty) = match composite_texp {
                TypedExpression::Value(composite_ir, composite_type) => (composite_ir, composite_type),
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_texp.to_parsetype())),
            };
            let ety = match &composite_ty {
                &ir::Type::Structured(ir::StructuredType::Custom(ref user_defined_name)) => {
                    match context.find_struct(user_defined_name) {
                        Some(struct_def) => {
                            fn find_struct_member(struct_def: &ir::StructDefinition, member: &String, struct_type: &ir::Type) -> Result<ir::Type, TyperError> {
                                for struct_member in &struct_def.members {
                                    if &struct_member.name == member {
                                        return Ok(struct_member.typename.clone())
                                    }
                                }
                                Err(TyperError::UnknownTypeMember(ParseType::Value(struct_type.clone()), member.clone()))
                            }
                            try!(find_struct_member(struct_def, member, &composite_ty))
                        },
                        None => return Err(TyperError::UnknownType(composite_pt)),
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
                                vec![MethodOverload(
                                    MethodName::Intrinsic(IntrinsicMethod::BufferLoad),
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
                                vec![MethodOverload(
                                    MethodName::Intrinsic(IntrinsicMethod::StructuredBufferLoad),
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
                    texp => return Err(TyperError::FunctionPassedToAnotherFunction(func_texp.to_parsetype(), texp.to_parsetype())),
                };
                params_ir.push(expr_ir);
                params_types.push(expr_ty);
            };
            match func_texp {
                TypedExpression::Function(unresolved) => {
                    let function = try!(find_function_type(unresolved, params_types));
                    write_function(function, params_ir)
                },
                TypedExpression::Method(unresolved) => {
                    let method = try!(find_method_type(unresolved, params_types));
                    write_method(method, params_ir)
                },
                _ => return Err(TyperError::CallOnNonFunction),
            }
        },
        &ast::Expression::Cast(ref ty, ref expr) => {
            let expr_texp = try!(parse_expr(expr, context));
            let expr_pt = expr_texp.to_parsetype();
            match expr_texp {
                TypedExpression::Value(expr_ir, _) => {
                    Ok(TypedExpression::Value(ir::Expression::Cast(ty.clone(), Box::new(expr_ir)), ty.clone()))
                },
                _ => Err(TyperError::InvalidCast(expr_pt, ParseType::Value(ty.clone()))),
            }
        },
    }
}

fn parse_vardef(ast: &ast::VarDef, context: ScopeContext) -> Result<(ir::VarDef, ScopeContext), TyperError> {
    let assign_ir = match ast.assignment {
        Some(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => Some(expr_ir),
                _ => return Err(TyperError::FunctionTypeInInitExpression),
            }
        },
        None => None,
    };
    let var_name = ast.name.clone();
    let var_type = ast.typename.clone();
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

fn parse_statement(ast: &ast::Statement, context: ScopeContext) -> Result<(Option<ir::Statement>, ScopeContext), TyperError> {
    fn empty_block() -> ir::Statement { ir::Statement::Block(vec![], ir::ScopedDeclarations { variables: HashMap::new() }) }
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
            Ok((Some(ir::Statement::Block(statements, decls)), context))
        },
        &ast::Statement::If(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (cond_ir, scoped_context) = try!(parse_condition(cond, scoped_context));
            let (statement_ir_opt, scoped_context) = try!(parse_statement(statement, scoped_context));
            let decls = scoped_context.destruct();
            let statement_ir = Box::new(match statement_ir_opt { Some(statement_ir) => statement_ir, None => empty_block() });
            Ok((Some(ir::Statement::If(cond_ir, statement_ir, decls)), context))
        },
        &ast::Statement::For(ref init, ref cond, ref iter, ref statement) =>  {
            let scoped_context = ScopeContext::from_scope(&context);
            let (init_ir, scoped_context) = try!(parse_condition(init, scoped_context));
            let (cond_ir, scoped_context) = try!(parse_condition(cond, scoped_context));
            let (iter_ir, scoped_context) = try!(parse_condition(iter, scoped_context));
            let (statement_ir_opt, scoped_context) = try!(parse_statement(statement, scoped_context));
            let decls = scoped_context.destruct();
            let statement_ir = Box::new(match statement_ir_opt { Some(statement_ir) => statement_ir, None => empty_block() });
            Ok((Some(ir::Statement::For(init_ir, cond_ir, iter_ir, statement_ir, decls)), context))
        },
        &ast::Statement::While(ref cond, ref statement) => {
            let scoped_context = ScopeContext::from_scope(&context);
            let (cond_ir, scoped_context) = try!(parse_condition(cond, scoped_context));
            let (statement_ir_opt, scoped_context) = try!(parse_statement(statement, scoped_context));
            let decls = scoped_context.destruct();
            let statement_ir = Box::new(match statement_ir_opt { Some(statement_ir) => statement_ir, None => empty_block() });
            Ok((Some(ir::Statement::While(cond_ir, statement_ir, decls)), context))
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
    let struct_def = sd.clone();
    match context.structs.insert(struct_def.name.clone(), struct_def.clone()) {
        Some(_) => return Err(TyperError::StructAlreadyDefined(struct_def.name.clone())),
        None => { },
    };
    Ok((ir::RootDefinition::Struct(struct_def), context))
}

fn parse_rootdefinition_constantbuffer(cb: &ast::ConstantBuffer, mut context: GlobalContext, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let cb_name = cb.name.clone();
    let cb_type = ir::Type::custom(&cb_name);
    let cb_id = try!(context.insert_variable(cb_name.clone(), cb_type.clone()));
    let cb_ir = ir::ConstantBuffer { id: cb_id, members: cb.members.clone() };
    match cb.slot {
        Some(ast::ConstantSlot(slot)) => {
            match globals.constants.insert(slot, cb_name.clone()) {
                Some(currently_used_by) => return Err(TyperError::ConstantSlotAlreadyUsed(currently_used_by.clone(), cb_name.clone())),
                None => { },
            }
        },
        None => { },
    }
    Ok((ir::RootDefinition::ConstantBuffer(cb_ir), context))
}

fn parse_rootdefinition_globalvariable(gv: &ast::GlobalVariable, mut context: GlobalContext, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, GlobalContext), TyperError> {
    let var_name = gv.name.clone();
    let var_type = gv.typename.clone();
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
            let var_id = try!(scoped_context.insert_variable(param.name.clone(), param.typename.clone()));
            vec.push(ir::FunctionParam {
                id: var_id,
                typename: param.typename.clone(),
            });
        }
        vec
    };
    let (body_ir, scoped_context) = try!(parse_statement_vec(&fd.body, scoped_context));
    let decls = scoped_context.destruct();

    let fd_ir = ir::FunctionDefinition {
        id: context.make_function_id(),
        returntype: fd.returntype.clone(),
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
            let var_id = try!(scoped_context.insert_variable(param.name.clone(), param.typename.clone()));
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
                    ast::Statement::Var(ast::VarDef { name: "a".to_string(), typename: ir::Type::uint(), assignment: None }),
                    ast::Statement::Var(ast::VarDef { name: "b".to_string(), typename: ir::Type::uint(), assignment: None }),
                    ast::Statement::Expression(
                        ast::Expression::BinaryOperation(ast::BinOp::Assignment,
                            Box::new(ast::Expression::Variable("a".to_string())),
                            Box::new(ast::Expression::Variable("b".to_string()))
                        )
                    ),
                    ast::Statement::If(
                        ast::Condition::Assignment(ast::VarDef {
                            name: "c".to_string(),
                            typename: ir::Type::uint(),
                            assignment: Some(ast::Expression::Variable("a".to_string()))
                        }),
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
