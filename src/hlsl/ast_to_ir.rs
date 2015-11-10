
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use super::ir;
use super::ast;

#[derive(PartialEq, Debug, Clone)]
pub enum ParseError {
    Unimplemented,

    ValueAlreadyDefined(String, ParseType, ParseType),
    StructAlreadyDefined(String),

    ConstantSlotAlreadyUsed(String, String),
    ReadResourceSlotAlreadyUsed(String, String),
    ReadWriteResourceSlotAlreadyUsed(String, String),

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
    User(String),
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionOverload(pub ReturnType, pub ParamArray);

#[derive(PartialEq, Debug, Clone)]
pub struct ResolvedFunction(pub FunctionName, pub FunctionOverload);

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedFunction(pub FunctionName, pub Vec<FunctionOverload>);

#[derive(PartialEq, Debug, Clone)]
pub enum IntrinsicMethod {
    BufferLoad,
    StructuredBufferLoad,
}

#[derive(PartialEq, Debug, Clone)]
pub enum MethodName {
    Intrinsic(IntrinsicMethod),
    User(String),
}

#[derive(PartialEq, Debug, Clone)]
pub struct MethodOverload(pub ReturnType, pub ParamArray);

#[derive(PartialEq, Debug, Clone)]
pub struct ResolvedMethod(pub MethodName, pub ClassType, pub MethodOverload, ir::Expression);

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedMethod(pub MethodName, pub ClassType, pub Vec<MethodOverload>, ir::Expression);

#[derive(PartialEq, Debug, Clone)]
pub enum ParseType {
    Value(ir::Type),
    Function(UnresolvedFunction),
    Method(UnresolvedMethod),
    Unknown,
}

#[derive(PartialEq, Debug, Clone)]
pub enum TypedExpression {
    // Expression + Type
    Value(ir::Expression, ir::Type),
    // Name of function + overloads
    Function(UnresolvedFunction),
    // Name of function + overloads + object
    Method(UnresolvedMethod),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Context {
    structs: HashMap<String, ir::StructDefinition>,

    functions: HashMap<String, UnresolvedFunction>,

    // Variables (and functions) visible in the current context. Maps names to types
    // One map per scope level (nearer the front is a wider scope)
    variables: Vec<HashMap<String, ir::Type>>,
}

impl FunctionName {
    pub fn get_name(&self) -> String {
        match self {
            &FunctionName::User(ref s) => s.clone(),
            &FunctionName::Intrinsic(_) => "<unknown>".to_string(),
        }
    }
}

impl UnresolvedFunction {
    pub fn get_name(&self) -> String { self.0.get_name() }
}

impl MethodName {
    pub fn get_name(&self) -> String {
        match self {
            &MethodName::User(ref s) => s.clone(),
            &MethodName::Intrinsic(_) => "<unknown-method>".to_string(),
        }
    }
}

impl UnresolvedMethod {
    pub fn get_name(&self) -> String { self.0.get_name() }
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

impl Context {
    pub fn new() -> Context {
        Context { structs: HashMap::new(), functions: get_intrinsics(), variables: vec![HashMap::new()] }
    }

    pub fn insert_variable(&mut self, name: String, typename: ir::Type) -> Result<(), ParseError> {
        assert!(self.variables.len() > 0);
        if let Ok(ety) = self.find_variable(&name) {
            return Err(ParseError::ValueAlreadyDefined(name, ety.to_parsetype(), ParseType::Value(typename)))
        };
        let last = self.variables.len() - 1;
        match self.variables[last].entry(name.clone()) {
            Entry::Occupied(occupied) => Err(ParseError::ValueAlreadyDefined(name, ParseType::Value(occupied.get().clone()), ParseType::Value(typename))),
            Entry::Vacant(vacant) => { vacant.insert(typename); Ok(()) },
        }
    }

    pub fn insert_function(&mut self, name: String, function_type: FunctionOverload) -> Result<(), ParseError> {
        // Error if a variable of the same name already exists
        assert_eq!(self.variables.len(), 1);
        for map in self.variables.iter().rev() {
            match map.get(&name) {
                Some(ty) => return Err(ParseError::ValueAlreadyDefined(name, ParseType::Value(ty.clone()), ParseType::Unknown)),
                None => { },
            }
        }
        // Try to add the function
        match self.functions.entry(name.clone()) {
            Entry::Occupied(mut occupied) => {
                // Fail if the overload already exists
                for &FunctionOverload(_, ref args) in &occupied.get().1 {
                    if *args == function_type.1 {
                        return Err(ParseError::ValueAlreadyDefined(name, ParseType::Unknown, ParseType::Unknown))
                    }
                };
                // Insert a new overload
                occupied.get_mut().1.push(function_type);
                Ok(())
            },
            Entry::Vacant(vacant) => {
                // Insert a new function with one overload
                vacant.insert(UnresolvedFunction(FunctionName::User(name), vec![function_type])); Ok(())
            },
        }
    }

    pub fn find_variable(&self, name: &String) -> Result<TypedExpression, ParseError> {
        match self.functions.get(name) {
            Some(tys) => return Ok(TypedExpression::Function(tys.clone())),
            None => { },
        }
        for map in self.variables.iter().rev() {
            match map.get(name) {
                Some(ty) => return Ok(TypedExpression::Value(ir::Expression::Variable(name.clone()), ty.clone())),
                None => { },
            }
        }
        Err(ParseError::UnknownIdentifier(name.clone()))
    }

    pub fn scoped(&self) -> Context {
        let mut scoped_vars = self.variables.clone();
        scoped_vars.push(HashMap::new());
        Context {
            structs: self.structs.clone(),
            functions: self.functions.clone(),
            variables: scoped_vars,
        }
    }
}

fn get_intrinsics() -> HashMap<String, UnresolvedFunction> {
    let mut map = HashMap::new();

    fn t_float() -> ir::Type { ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::Float))) };
    fn t_float4() -> ir::Type { ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Vector(ir::ScalarType::Float, 4))) };

    map.insert("float4".to_string(), UnresolvedFunction(FunctionName::Intrinsic(IntrinsicFunction::Float4), vec![
        FunctionOverload(t_float4(), vec![t_float(), t_float(), t_float(), t_float()])
    ]));

    map
}

fn find_function_type(function: UnresolvedFunction, actual_params: ParamArray) -> Result<ResolvedFunction, ParseError> {
    for overload in &function.1 {
        if overload.1 == actual_params {
            return Ok(ResolvedFunction(function.0, overload.clone()))
        };
    };
    Err(ParseError::FunctionArgumentTypeMismatch(function, actual_params))
}

fn find_method_type(method: UnresolvedMethod, actual_params: ParamArray) -> Result<ResolvedMethod, ParseError> {
    for overload in &method.2 {
        if overload.1 == actual_params {
            return Ok(ResolvedMethod(method.0, method.1, overload.clone(), method.3))
        };
    };
    Err(ParseError::MethodArgumentTypeMismatch(method, actual_params))
}

fn write_function(function: ResolvedFunction, param_values: Vec<ir::Expression>) -> Result<TypedExpression, ParseError> {
    let ResolvedFunction(name, FunctionOverload(return_type, _)) = function;
    let intrinsic = match name {
        FunctionName::Intrinsic(intrinsic) => intrinsic,
        FunctionName::User(name) => {
            return Ok(TypedExpression::Value(ir::Expression::Call(Box::new(ir::Expression::Variable(name)), param_values), return_type))
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

fn write_method(method: ResolvedMethod, param_values: Vec<ir::Expression>) -> Result<TypedExpression, ParseError> {
    let ResolvedMethod(name, _, MethodOverload(return_type, _), object_ir) = method;
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

fn parse_literal(ast: &ast::Literal) -> Result<TypedExpression, ParseError> {
    match ast {
        &ast::Literal::Int(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Int(i)), ir::Type::int())),
        &ast::Literal::Uint(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Uint(i)), ir::Type::uint())),
        &ast::Literal::Long(i) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Long(i)), ir::Type::long())),
        &ast::Literal::Half(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Half(f)), ir::Type::float())),
        &ast::Literal::Float(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Float(f)), ir::Type::float())),
        &ast::Literal::Double(f) => Ok(TypedExpression::Value(ir::Expression::Literal(ir::Literal::Double(f)), ir::Type::double())),
    }
}

fn parse_expr(ast: &ast::Expression, context: &Context) -> Result<TypedExpression, ParseError> {
    match ast {
        &ast::Expression::Literal(ref lit) => parse_literal(lit),
        &ast::Expression::Variable(ref s) => Ok(try!(context.find_variable(s))),
        &ast::Expression::UnaryOperation(ref op, ref expr) => {
            match try!(parse_expr(expr, context)) {
                TypedExpression::Value(expr_ir, expr_ty) => {
                    Ok(TypedExpression::Value(ir::Expression::UnaryOperation(op.clone(), Box::new(expr_ir)), expr_ty))
                },
                _ => Err(ParseError::UnaryOperationWrongTypes(op.clone(), ParseType::Unknown)),
            }
        },
        &ast::Expression::BinaryOperation(ref op, ref lhs, ref rhs) => {
            let lhs_texp = try!(parse_expr(lhs, context));
            let rhs_texp = try!(parse_expr(rhs, context));
            let lhs_pt = lhs_texp.to_parsetype();
            let rhs_pt = rhs_texp.to_parsetype();
            let (lhs_ir, lhs_type) = match lhs_texp {
                TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                _ => return Err(ParseError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
            };
            let (rhs_ir, rhs_type) = match rhs_texp {
                TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                _ => return Err(ParseError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt)),
            };
            if lhs_type != rhs_type {
                Err(ParseError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt))
            } else {
                Ok(TypedExpression::Value(ir::Expression::BinaryOperation(op.clone(), Box::new(lhs_ir), Box::new(rhs_ir)), lhs_type.clone()))
            }
        },
        &ast::Expression::ArraySubscript(ref array, ref subscript) => {
            let array_texp = try!(parse_expr(array, context));
            let subscript_texp = try!(parse_expr(subscript, context));
            let (array_ir, array_ty) = match array_texp {
                TypedExpression::Value(array_ir, array_ty) => (array_ir, array_ty),
                _ => return Err(ParseError::ArrayIndexingNonArrayType),
            };
            let (subscript_ir, subscript_ty) = match subscript_texp {
                TypedExpression::Value(subscript_ir, subscript_ty) => (subscript_ir, subscript_ty),
                _ => return Err(ParseError::ArrayIndexingNonArrayType),
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
                _ => return Err(ParseError::ArrayIndexingNonArrayType),
            };
            match subscript_ty {
                ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::Int))) |
                ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::UInt))) => { },
                _ => return Err(ParseError::ArraySubscriptIndexNotInteger),
            };
            Ok(TypedExpression::Value(ir::Expression::ArraySubscript(Box::new(array_ir), Box::new(subscript_ir)), indexed_type))
        },
        &ast::Expression::Member(ref composite, ref member) => {
            let composite_texp = try!(parse_expr(composite, context));
            let composite_pt = composite_texp.to_parsetype();
            let (composite_ir, composite_ty) = match composite_texp {
                TypedExpression::Value(composite_ir, composite_type) => (composite_ir, composite_type),
                _ => return Err(ParseError::TypeDoesNotHaveMembers(composite_texp.to_parsetype())),
            };
            let ety = match &composite_ty {
                &ir::Type::Structured(ir::StructuredType::Custom(ref user_defined_name)) => {
                    match context.structs.get(user_defined_name) {
                        Some(struct_def) => {
                            fn find_struct_member(struct_def: &ir::StructDefinition, member: &String, struct_type: &ir::Type) -> Result<ir::Type, ParseError> {
                                for struct_member in &struct_def.members {
                                    if &struct_member.name == member {
                                        return Ok(struct_member.typename.clone())
                                    }
                                }
                                Err(ParseError::UnknownTypeMember(ParseType::Value(struct_type.clone()), member.clone()))
                            }
                            try!(find_struct_member(struct_def, member, &composite_ty))
                        },
                        None => return Err(ParseError::UnknownType(composite_pt)),
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
                        return Err(ParseError::UnknownTypeMember(composite_pt, member.clone()));
                    }
                }
                &ir::Type::Object(ir::ObjectType::Buffer(ref data_type)) => {
                    match &member[..] {
                        "Load" => {
                            return Ok(TypedExpression::Method(UnresolvedMethod(
                                MethodName::Intrinsic(IntrinsicMethod::BufferLoad),
                                ir::Type::Object(ir::ObjectType::Buffer(data_type.clone())),
                                vec![MethodOverload(
                                    ir::Type::Structured(ir::StructuredType::Data(data_type.clone())),
                                    vec![ir::Type::int()]
                                )],
                                composite_ir
                            )))
                        },
                        _ => return Err(ParseError::UnknownTypeMember(composite_pt, member.clone())),
                    }
                }
                &ir::Type::Object(ir::ObjectType::StructuredBuffer(ref structured_type)) => {
                    match &member[..] {
                        "Load" => {
                            return Ok(TypedExpression::Method(UnresolvedMethod(
                                MethodName::Intrinsic(IntrinsicMethod::StructuredBufferLoad),
                                ir::Type::Object(ir::ObjectType::StructuredBuffer(structured_type.clone())),
                                vec![MethodOverload(
                                    ir::Type::Structured(structured_type.clone()),
                                    vec![ir::Type::int()]
                                )],
                                composite_ir
                            )))
                        },
                        _ => return Err(ParseError::UnknownTypeMember(composite_pt, member.clone())),
                    }
                }
                // Todo: Matrix components + Object members
                _ => return Err(ParseError::TypeDoesNotHaveMembers(composite_pt)),
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
                    texp => return Err(ParseError::FunctionPassedToAnotherFunction(func_texp.to_parsetype(), texp.to_parsetype())),
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
                _ => return Err(ParseError::CallOnNonFunction),
            }
        },
        &ast::Expression::Cast(ref ty, ref expr) => {
            let expr_texp = try!(parse_expr(expr, context));
            let expr_pt = expr_texp.to_parsetype();
            match expr_texp {
                TypedExpression::Value(expr_ir, _) => {
                    Ok(TypedExpression::Value(ir::Expression::Cast(ty.clone(), Box::new(expr_ir)), ty.clone()))
                },
                _ => Err(ParseError::InvalidCast(expr_pt, ParseType::Value(ty.clone()))),
            }
        },
    }
}

fn parse_vardef(ast: &ast::VarDef, context: Context) -> Result<(ir::VarDef, Context), ParseError> {
    let assign_ir = match ast.assignment {
        Some(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => Some(expr_ir),
                _ => return Err(ParseError::FunctionTypeInInitExpression),
            }
        },
        None => None,
    };
    let vd_ir = ir::VarDef { name: ast.name.clone(), typename: ast.typename.clone(), assignment: assign_ir };
    let mut context = context;
    try!(context.insert_variable(vd_ir.name.clone(), vd_ir.typename.clone()));
    Ok((vd_ir, context))
}

fn parse_condition(ast: &ast::Condition, context: Context) -> Result<(ir::Condition, Context), ParseError> {
    match ast {
        &ast::Condition::Expr(ref expr) => {
            let expr_ir = match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => expr_ir,
                _ => return Err(ParseError::FunctionNotCalled),
            };
            Ok((ir::Condition::Expr(expr_ir), context))
        },
        &ast::Condition::Assignment(ref vd) => {
            let (vd_ir, context) = try!(parse_vardef(vd, context));
            Ok((ir::Condition::Assignment(vd_ir), context))
        },
    }
}

fn parse_statement(ast: &ast::Statement, context: Context) -> Result<(Option<ir::Statement>, Context), ParseError> {
    match ast {
        &ast::Statement::Empty => Ok((None, context)),
        &ast::Statement::Expression(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => Ok((Some(ir::Statement::Expression(expr_ir)), context)),
                _ => return Err(ParseError::FunctionNotCalled),
            }
        },
        &ast::Statement::Var(ref vd) => {
            let (vd_ir, context) = try!(parse_vardef(vd, context));
            Ok((Some(ir::Statement::Var(vd_ir)), context))
        },
        &ast::Statement::Block(ref statement_vec) => {
            let scoped_context = context.scoped();
            let statements = try!(parse_statement_vec(statement_vec, scoped_context));
            Ok((Some(ir::Statement::Block(statements)), context))
        },
        &ast::Statement::If(ref cond, ref statement) => {
            let (cond_ir, context) = try!(parse_condition(cond, context));
            let (statement_ir_opt, context) = try!(parse_statement(statement, context));
            let statement_ir = Box::new(match statement_ir_opt { Some(statement_ir) => statement_ir, None => ir::Statement::Block(vec![]) });
            Ok((Some(ir::Statement::If(cond_ir, statement_ir)), context))
        },
        &ast::Statement::For(ref init, ref cond, ref iter, ref statement) =>  {
            let (init_ir, context) = try!(parse_condition(init, context));
            let (cond_ir, context) = try!(parse_condition(cond, context));
            let (iter_ir, context) = try!(parse_condition(iter, context));
            let (statement_ir_opt, context) = try!(parse_statement(statement, context));
            let statement_ir = Box::new(match statement_ir_opt { Some(statement_ir) => statement_ir, None => ir::Statement::Block(vec![]) });
            Ok((Some(ir::Statement::For(init_ir, cond_ir, iter_ir, statement_ir)), context))
        },
        &ast::Statement::While(ref cond, ref statement) => {
            let (cond_ir, context) = try!(parse_condition(cond, context));
            let (statement_ir_opt, context) = try!(parse_statement(statement, context));
            let statement_ir = Box::new(match statement_ir_opt { Some(statement_ir) => statement_ir, None => ir::Statement::Block(vec![]) });
            Ok((Some(ir::Statement::While(cond_ir, statement_ir)), context))
        },
        &ast::Statement::Return(ref expr) => {
            match try!(parse_expr(expr, &context)) {
                TypedExpression::Value(expr_ir, _) => Ok((Some(ir::Statement::Return(expr_ir)), context)),
                _ => return Err(ParseError::FunctionNotCalled),
            }
        },
    }
}

fn parse_statement_vec(ast: &Vec<ast::Statement>, context: Context) -> Result<Vec<ir::Statement>, ParseError> {
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
    Ok(body_ir)
}

fn parse_rootdefinition_struct(sd: &ast::StructDefinition, mut context: Context) -> Result<(ir::RootDefinition, Context), ParseError> {
    let struct_def = sd.clone();
    match context.structs.insert(struct_def.name.clone(), struct_def.clone()) {
        Some(_) => return Err(ParseError::StructAlreadyDefined(struct_def.name.clone())),
        None => { },
    };
    Ok((ir::RootDefinition::Struct(struct_def), context))
}

fn parse_rootdefinition_constantbuffer(cb: &ast::ConstantBuffer, mut context: Context, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, Context), ParseError> {
    let cb_ir = ir::ConstantBuffer { name: cb.name.clone(), members: cb.members.clone() };
    try!(context.insert_variable(cb_ir.name.clone(), ir::Type::custom(&cb_ir.name[..])));
    match cb.slot {
        Some(ast::ConstantSlot(slot)) => {
            match globals.constants.insert(slot, cb_ir.name.clone()) {
                Some(currently_used_by) => return Err(ParseError::ConstantSlotAlreadyUsed(currently_used_by.clone(), cb_ir.name.clone())),
                None => { },
            }
        },
        None => { },
    }
    Ok((ir::RootDefinition::ConstantBuffer(cb_ir), context))
}

fn parse_rootdefinition_globalvariable(gv: &ast::GlobalVariable, mut context: Context, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, Context), ParseError> {
    let gv_ir = ir::GlobalVariable { name: gv.name.clone(), typename: gv.typename.clone() };
    try!(context.insert_variable(gv_ir.name.clone(), gv_ir.typename.clone()));
    let entry = ir::GlobalEntry { name: gv_ir.name.clone(), typename: gv_ir.typename.clone() };
    match gv.slot {
        Some(ast::GlobalSlot::ReadSlot(slot)) => {
            match globals.r_resources.insert(slot, entry) {
                Some(currently_used_by) => return Err(ParseError::ReadResourceSlotAlreadyUsed(currently_used_by.name.clone(), gv_ir.name.clone())),
                None => { },
            }
        },
        Some(ast::GlobalSlot::ReadWriteSlot(slot)) => {
            match globals.rw_resources.insert(slot, entry) {
                Some(currently_used_by) => return Err(ParseError::ReadWriteResourceSlotAlreadyUsed(currently_used_by.name.clone(), gv_ir.name.clone())),
                None => { },
            }
        },
        None => { },
    }
    Ok((ir::RootDefinition::GlobalVariable(gv_ir), context))
}

fn parse_rootdefinition_function(fd: &ast::FunctionDefinition, mut context: Context) -> Result<(ir::RootDefinition, Context), ParseError> {
    let body_ir = {
        let mut scoped_context = context.scoped();
        for param in &fd.params {
            try!(scoped_context.insert_variable(param.name.clone(), param.typename.clone()));
        }
        try!(parse_statement_vec(&fd.body, scoped_context))
    };
    let fd_ir = ir::FunctionDefinition {
        name: fd.name.clone(),
        returntype: fd.returntype.clone(),
        params: fd.params.clone(),
        body: body_ir,
        attributes: fd.attributes.clone(),
    };
    let func_type = FunctionOverload(
        fd_ir.returntype.clone(),
        fd_ir.params.iter().map(|p| { p.typename.clone() }).collect()
    );
    try!(context.insert_function(fd_ir.name.clone(), func_type));
    Ok((ir::RootDefinition::Function(fd_ir), context))
}

fn parse_rootdefinition_kernel(fd: &ast::FunctionDefinition, context: Context) -> Result<(ir::RootDefinition, Context), ParseError> {
    let body_ir = {
        let mut scoped_context = context.scoped();
        for param in &fd.params {
            try!(scoped_context.insert_variable(param.name.clone(), param.typename.clone()));
        }
        try!(parse_statement_vec(&fd.body, scoped_context))
    };
    fn find_dispatch_dimensions(attributes: &[ast::FunctionAttribute]) -> Result<ir::Dimension, ParseError> {
        for attribute in attributes {
            match attribute {
                &ast::FunctionAttribute::NumThreads(x, y, z) => return Ok(ir::Dimension(x, y, z)),
            };
        }
        Err(ParseError::KernelHasNoDispatchDimensions)
    }
    let kernel = ir::Kernel {
        group_dimensions: try!(find_dispatch_dimensions(&fd.attributes[..])),
        params: vec![],
        body: body_ir,
    };
    Ok((ir::RootDefinition::Kernel(kernel), context))
}

fn parse_rootdefinition(ast: &ast::RootDefinition, context: Context, globals: &mut ir::GlobalTable, entry_point: &str) -> Result<(ir::RootDefinition, Context), ParseError> {
    match ast {
        &ast::RootDefinition::Struct(ref sd) => parse_rootdefinition_struct(sd, context),
        &ast::RootDefinition::SamplerState => Ok((ir::RootDefinition::SamplerState, context)),
        &ast::RootDefinition::ConstantBuffer(ref cb) => parse_rootdefinition_constantbuffer(cb, context, globals),
        &ast::RootDefinition::GlobalVariable(ref gv) => parse_rootdefinition_globalvariable(gv, context, globals),
        &ast::RootDefinition::Function(ref fd) if fd.name == entry_point => parse_rootdefinition_kernel(fd, context),
        &ast::RootDefinition::Function(ref fd) => parse_rootdefinition_function(fd, context),
    }
}

pub fn parse(ast: &ast::Module) -> Result<ir::Module, ParseError> {
    let mut context = Context::new();

    let mut ir = ir::Module { entry_point: ast.entry_point.clone(), global_table: ir::GlobalTable::new(), root_definitions: vec![] };

    for def in &ast.root_definitions {
        let (def_ir, next_context) = try!(parse_rootdefinition(&def, context, &mut ir.global_table, &ir.entry_point[..]));
        ir.root_definitions.push(def_ir);
        context = next_context;
    }

    // Ensure we have one kernel function
    let mut has_kernel = false;
    for root_def in &ir.root_definitions {
        match root_def {
            &ir::RootDefinition::Kernel(_) => {
                if has_kernel {
                    return Err(ParseError::KernelDefinedMultipleTimes);
                } else {
                    has_kernel = true;
                }
            },
            _ => { },
        }
    }
    if !has_kernel {
        return Err(ParseError::KernelNotDefined);
    }

    Ok(ir)
}

#[test]
fn test_parse() {
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
                params: vec![ast::FunctionParam { name: "x".to_string(), typename: ast::Type::uint() }],
                body: vec![],
                attributes: vec![],
            }),
            ast::RootDefinition::Function(ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: ast::Type::Void,
                params: vec![ast::FunctionParam { name: "x".to_string(), typename: ast::Type::float() }],
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
    let res = parse(&module);
    assert!(res.is_ok(), "{:?}", res);
}
