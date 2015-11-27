use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use super::cir as dst;
use super::super::hlsl::ir as src;

#[derive(PartialEq, Debug, Clone)]
pub enum TranspileError {
    Unknown,

    TypeIsNotAllowedAsGlobal(src::Type),
    CouldNotGetEquivalentType(src::Type),

    GlobalFoundThatIsntInKernelParams(src::GlobalVariable),

    UnknownFunctionId(src::FunctionId),
    UnknownStructId(src::StructId),
    UnknownConstantBufferId(src::ConstantBufferId),
    InvalidVariableRef,
    UnknownVariableId,

    BoolVectorsNotSupported,
}

impl error::Error for TranspileError {
    fn description(&self) -> &str {
        match *self {
            TranspileError::Unknown => "unknown transpiler error",
            TranspileError::TypeIsNotAllowedAsGlobal(_) => "global variable has unsupported type",
            TranspileError::CouldNotGetEquivalentType(_) => "could not find equivalent clc type",
            TranspileError::GlobalFoundThatIsntInKernelParams(_) => "non-parameter global found",
            TranspileError::UnknownFunctionId(_) => "unknown function id",
            TranspileError::UnknownStructId(_) => "unknown struct id",
            TranspileError::UnknownConstantBufferId(_) => "unknown cbuffer id",
            TranspileError::InvalidVariableRef => "invalid variable ref",
            TranspileError::UnknownVariableId => "unknown variable id",
            TranspileError::BoolVectorsNotSupported => "bool vectors not supported",
        }
    }
}

impl fmt::Display for TranspileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

type KernelParams = Vec<dst::KernelParam>;

struct Context {
    kernel_params: KernelParams,
    function_name_map: HashMap<src::FunctionId, String>,
    struct_name_map: HashMap<src::StructId, String>,
    cbuffer_type_name_map: HashMap<src::ConstantBufferId, String>,
    cbuffer_instance_name_map: HashMap<src::ConstantBufferId, String>,
    variable_scopes: Vec<HashMap<src::VariableId, String>>,
}

impl Context {

    fn new(globals: &src::GlobalDeclarations) -> Context {
        let mut context = Context {
            kernel_params: vec![],
            function_name_map: HashMap::new(),
            struct_name_map: HashMap::new(),
            cbuffer_type_name_map: HashMap::new(),
            cbuffer_instance_name_map: HashMap::new(),
            variable_scopes: vec![globals.variables.clone()],
        };

        // Insert functions (try to generate [name]_[overload number] for overloads)
        {
            let mut grouped_functions: HashMap<String, Vec<src::FunctionId>> = HashMap::new();
            for (id, name) in globals.functions.iter() {
                match grouped_functions.entry(name.clone()) {
                    Entry::Occupied(mut occupied) => { occupied.get_mut().push(id.clone()); },
                    Entry::Vacant(vacant) => { vacant.insert(vec![id.clone()]); },
                }
            };
            for (key, mut ids) in grouped_functions {
                assert!(ids.len() > 0);
                if ids.len() == 1 {
                    let identifier = context.make_identifier(&key);
                    let ret = context.function_name_map.insert(ids[0], identifier);
                    assert_eq!(ret, None);
                } else {
                    ids.sort();
                    for (index, id) in ids.iter().enumerate() {
                        let gen = key.clone() + "_" + &index.to_string();
                        let identifier = context.make_identifier(&gen);
                        let ret = context.function_name_map.insert(*id, identifier);
                        assert_eq!(ret, None);
                    }
                }
            };
        }

        // Insert structs (name collisions possible with function overloads + globals)
        {
            for (id, struct_name) in globals.structs.iter() {
                let identifier = context.make_identifier(struct_name);
                let ret = context.struct_name_map.insert(id.clone(), identifier);
                assert_eq!(ret, None);
            };
        }

        // Insert cbuffers
        {
            for (id, cbuffer_name) in globals.constants.iter() {
                let identifier_instance = context.make_identifier(cbuffer_name);
                let ret = context.cbuffer_instance_name_map.insert(id.clone(), identifier_instance);
                assert_eq!(ret, None);
                let identifier_type = context.make_identifier(&(cbuffer_name.clone() + "_t"));
                let ret = context.cbuffer_type_name_map.insert(id.clone(), identifier_type);
                assert_eq!(ret, None);
            };
        }

        context
    }

    fn get_function(&self, id: &src::FunctionId) -> Result<dst::Expression, TranspileError> {
        Ok(dst::Expression::Variable(try!(self.get_function_name(id))))
    }

    fn get_function_name(&self, id: &src::FunctionId) -> Result<String, TranspileError> {
        match self.function_name_map.get(id) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownFunctionId(id.clone())),
        }
    }

    /// Get the expression to access an in scope variable
    fn get_variable_ref(&self, var_ref: &src::VariableRef) -> Result<dst::Expression, TranspileError> {
        let scopes_up = var_ref.1 as usize;
        if scopes_up >= self.variable_scopes.len() {
            return Err(TranspileError::UnknownVariableId)
        };
        let scope = self.variable_scopes.len() - scopes_up - 1;
        match self.variable_scopes[scope].get(&var_ref.0) {
            Some(v) => Ok(dst::Expression::Variable(v.clone())),
            None => Err(TranspileError::UnknownVariableId),
        }
    }

    /// Get the name of a variable declared in the current block
    fn get_variable_id(&self, id: &src::VariableId) -> Result<String, TranspileError> {
        match self.variable_scopes[self.variable_scopes.len() - 1].get(id) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownVariableId),
        }
    }

    /// Get the name of a struct
    fn get_struct_name(&self, id: &src::StructId) -> Result<String, TranspileError> {
        match self.struct_name_map.get(id) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownStructId(id.clone())),
        }
    }

    /// Get the name of the struct used for a cbuffer
    fn get_cbuffer_struct_name(&self, id: &src::ConstantBufferId) -> Result<String, TranspileError> {
        match self.cbuffer_type_name_map.get(id) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
        }
    }

        /// Get the name of the cbuffer instance
    fn get_cbuffer_instance_name(&self, id: &src::ConstantBufferId) -> Result<String, TranspileError> {
        match self.cbuffer_instance_name_map.get(id) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
        }
    }

    fn is_free(&self, identifier: &str) -> bool {
        for func_name in self.function_name_map.values() {
            if func_name == identifier {
                return false;
            }
        }
        for struct_name in self.struct_name_map.values() {
            if struct_name == identifier {
                return false;
            }
        }
        for cbuffer_instance in self.cbuffer_instance_name_map.values() {
            if cbuffer_instance == identifier {
                return false;
            }
        }
        for cbuffer_type in self.cbuffer_type_name_map.values() {
            if cbuffer_type == identifier {
                return false;
            }
        }
        for scope in &self.variable_scopes {
            for var in scope.values() {
                if var == identifier {
                    return false;
                }
            }
        }
        return true;
    }

    fn make_identifier(&self, name: &str) -> String {
        if self.is_free(name) {
            return name.to_string();
        }
        let mut index = 0;
        loop {
            let test_name = name.to_string() + "_" + &index.to_string();
            if self.is_free(&test_name) {
                return test_name;
            }
            index = index + 1;
        }
    }

    fn push_scope(&mut self, decls: &src::ScopedDeclarations) {
        self.variable_scopes.push(HashMap::new());
        for (var_id, var_name) in &decls.variables {
            let identifier = self.make_identifier(&var_name);
            let map = self.variable_scopes.last_mut().unwrap();
            map.insert(var_id.clone(), identifier);
        }
    }

    fn pop_scope(&mut self) {
        assert!(self.variable_scopes.len() > 1);
        self.variable_scopes.pop();
    }
}

fn transpile_scalartype(scalartype: &src::ScalarType) -> Result<dst::Scalar, TranspileError> {
    match scalartype {
        &src::ScalarType::Bool => Err(TranspileError::BoolVectorsNotSupported),
        &src::ScalarType::Int => Ok(dst::Scalar::Int),
        &src::ScalarType::UInt => Ok(dst::Scalar::UInt),
        &src::ScalarType::Half => Ok(dst::Scalar::Half),
        &src::ScalarType::Float => Ok(dst::Scalar::Float),
        &src::ScalarType::Double => Ok(dst::Scalar::Double),
        &src::ScalarType::UntypedInt => panic!(),
    }
}

fn transpile_datatype(datatype: &src::DataType, context: &Context) -> Result<dst::Type, TranspileError> {
    transpile_type(&src::Type::from(datatype.clone()), context)
}

fn transpile_type(hlsltype: &src::Type, context: &Context) -> Result<dst::Type, TranspileError> {
    let &src::Type(ref ty, _) = hlsltype;
    match ty {
        &src::TypeLayout::Void => Ok(dst::Type::Void),
        &src::TypeLayout::Scalar(src::ScalarType::Bool) => Ok(dst::Type::Bool),
        &src::TypeLayout::Scalar(ref scalar) => Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar)))),
        &src::TypeLayout::Vector(ref scalar, 1) => Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar)))),
        &src::TypeLayout::Vector(ref scalar, 2) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Two)),
        &src::TypeLayout::Vector(ref scalar, 3) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Three)),
        &src::TypeLayout::Vector(ref scalar, 4) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Four)),
        &src::TypeLayout::Struct(ref id) => {
            let struct_name = try!(context.get_struct_name(id));
            Ok(dst::Type::Struct(struct_name))
        },
        _ => return Err(TranspileError::CouldNotGetEquivalentType(hlsltype.clone())),
    }
}

fn transpile_unaryop(unaryop: &src::UnaryOp) -> Result<dst::UnaryOp, TranspileError> {
    match *unaryop {
        src::UnaryOp::PrefixIncrement => Ok(dst::UnaryOp::PrefixIncrement),
        src::UnaryOp::PrefixDecrement => Ok(dst::UnaryOp::PrefixDecrement),
        src::UnaryOp::PostfixIncrement => Ok(dst::UnaryOp::PostfixIncrement),
        src::UnaryOp::PostfixDecrement => Ok(dst::UnaryOp::PostfixDecrement),
        src::UnaryOp::Plus => Ok(dst::UnaryOp::Plus),
        src::UnaryOp::Minus => Ok(dst::UnaryOp::Minus),
        src::UnaryOp::LogicalNot => Ok(dst::UnaryOp::LogicalNot),
        src::UnaryOp::BitwiseNot => Ok(dst::UnaryOp::BitwiseNot),
    }
}

fn transpile_binop(binop: &src::BinOp) -> Result<dst::BinOp, TranspileError> {
    match *binop {
        src::BinOp::Add => Ok(dst::BinOp::Add),
        src::BinOp::Subtract => Ok(dst::BinOp::Subtract),
        src::BinOp::Multiply => Ok(dst::BinOp::Multiply),
        src::BinOp::Divide => Ok(dst::BinOp::Divide),
        src::BinOp::Modulus => Ok(dst::BinOp::Modulus),
        src::BinOp::LeftShift => Ok(dst::BinOp::LeftShift),
        src::BinOp::RightShift => Ok(dst::BinOp::RightShift),
        src::BinOp::LessThan => Ok(dst::BinOp::LessThan),
        src::BinOp::LessEqual => Ok(dst::BinOp::LessEqual),
        src::BinOp::GreaterThan => Ok(dst::BinOp::GreaterThan),
        src::BinOp::GreaterEqual => Ok(dst::BinOp::GreaterEqual),
        src::BinOp::Equality => Ok(dst::BinOp::Equality),
        src::BinOp::Inequality => Ok(dst::BinOp::Inequality),
        src::BinOp::Assignment => Ok(dst::BinOp::Assignment),
    }
}

fn transpile_literal(lit: &src::Literal) -> Result<dst::Literal, TranspileError> {
    match lit {
        &src::Literal::Bool(b) => Ok(dst::Literal::Bool(b)),
        &src::Literal::UntypedInt(i) => Ok(dst::Literal::Int(i)),
        &src::Literal::Int(i) => Ok(dst::Literal::Int(i)),
        &src::Literal::UInt(i) => Ok(dst::Literal::UInt(i)),
        &src::Literal::Long(i) => Ok(dst::Literal::Long(i)),
        &src::Literal::Half(f) => Ok(dst::Literal::Half(f)),
        &src::Literal::Float(f) => Ok(dst::Literal::Float(f)),
        &src::Literal::Double(f) => Ok(dst::Literal::Double(f)),
    }
}

fn write_func(name: &'static str, args: &[&src::Expression], context: &Context) -> Result<dst::Expression, TranspileError> {
    Ok(dst::Expression::Call(
        Box::new(dst::Expression::Variable(name.to_string())),
        try!(args.iter().fold(Ok(vec![]), |result, exp| {
            let mut vec = try!(result);
            vec.push(try!(transpile_expression(exp, context)));
            Ok(vec)
        }))
    ))
}

fn transpile_intrinsic(intrinsic: &src::Intrinsic, context: &Context) -> Result<dst::Expression, TranspileError> {
    match *intrinsic {
        src::Intrinsic::AllMemoryBarrier | src::Intrinsic::AllMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::Call(
                Box::new(dst::Expression::Variable("barrier".to_string())),
                vec![dst::Expression::BinaryOperation(dst::BinOp::BitwiseOr,
                    Box::new(dst::Expression::Variable("CLK_LOCAL_MEM_FENCE".to_string())),
                    Box::new(dst::Expression::Variable("CLK_GLOBAL_MEM_FENCE".to_string()))
                )]
            ))
        },
        src::Intrinsic::AsIntU(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntU1(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntU2(ref e) => write_func("as_int2", &[e], context),
        src::Intrinsic::AsIntU3(ref e) => write_func("as_int3", &[e], context),
        src::Intrinsic::AsIntU4(ref e) => write_func("as_int4", &[e], context),
        src::Intrinsic::AsIntF(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntF1(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntF2(ref e) => write_func("as_int2", &[e], context),
        src::Intrinsic::AsIntF3(ref e) => write_func("as_int3", &[e], context),
        src::Intrinsic::AsIntF4(ref e) => write_func("as_int4", &[e], context),
        src::Intrinsic::AsUIntI(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntI1(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntI2(ref e) => write_func("as_uint2", &[e], context),
        src::Intrinsic::AsUIntI3(ref e) => write_func("as_uint3", &[e], context),
        src::Intrinsic::AsUIntI4(ref e) => write_func("as_uint4", &[e], context),
        src::Intrinsic::AsUIntF(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntF1(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntF2(ref e) => write_func("as_uint2", &[e], context),
        src::Intrinsic::AsUIntF3(ref e) => write_func("as_uint3", &[e], context),
        src::Intrinsic::AsUIntF4(ref e) => write_func("as_uint4", &[e], context),
        src::Intrinsic::AsFloatI(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatI1(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatI2(ref e) => write_func("as_float2", &[e], context),
        src::Intrinsic::AsFloatI3(ref e) => write_func("as_float3", &[e], context),
        src::Intrinsic::AsFloatI4(ref e) => write_func("as_float4", &[e], context),
        src::Intrinsic::AsFloatU(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatU1(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatU2(ref e) => write_func("as_float2", &[e], context),
        src::Intrinsic::AsFloatU3(ref e) => write_func("as_float3", &[e], context),
        src::Intrinsic::AsFloatU4(ref e) => write_func("as_float4", &[e], context),
        src::Intrinsic::AsFloatF(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatF1(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatF2(ref e) => write_func("as_float2", &[e], context),
        src::Intrinsic::AsFloatF3(ref e) => write_func("as_float3", &[e], context),
        src::Intrinsic::AsFloatF4(ref e) => write_func("as_float4", &[e], context),
        src::Intrinsic::ClampI(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI1(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI2(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI3(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI4(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF1(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF2(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF3(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF4(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::Float4(ref x, ref y, ref z, ref w) => {
            Ok(dst::Expression::Constructor(dst::Constructor::Float4(
                Box::new(try!(transpile_expression(x, context))),
                Box::new(try!(transpile_expression(y, context))),
                Box::new(try!(transpile_expression(z, context))),
                Box::new(try!(transpile_expression(w, context)))
            )))
        },
        src::Intrinsic::BufferLoad(ref buffer, ref loc) => {
            let cl_buffer = Box::new(try!(transpile_expression(buffer, context)));
            let cl_loc = Box::new(try!(transpile_expression(loc, context)));
            Ok(dst::Expression::ArraySubscript(cl_buffer, cl_loc))
        },
        src::Intrinsic::StructuredBufferLoad(_, _) => unimplemented!(),
        _ => unimplemented!(),
    }
}

fn transpile_expression(expression: &src::Expression, context: &Context) -> Result<dst::Expression, TranspileError> {
    match expression {
        &src::Expression::Literal(ref lit) => Ok(dst::Expression::Literal(try!(transpile_literal(lit)))),
        &src::Expression::Variable(ref var_ref) => context.get_variable_ref(var_ref),
        &src::Expression::ConstantVariable(ref id, ref name) => {
            let cbuffer_instance_name = try!(context.get_cbuffer_instance_name(id));
            Ok(dst::Expression::MemberDeref(Box::new(dst::Expression::Variable(cbuffer_instance_name)), name.clone()))
        },
        &src::Expression::Function(ref id) => context.get_function(id),
        &src::Expression::UnaryOperation(ref unaryop, ref expr) => {
            let cl_unaryop = try!(transpile_unaryop(unaryop));
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::UnaryOperation(cl_unaryop, cl_expr))
        }
        &src::Expression::BinaryOperation(ref binop, ref lhs, ref rhs) => {
            let cl_binop = try!(transpile_binop(binop));
            let cl_lhs = Box::new(try!(transpile_expression(lhs, context)));
            let cl_rhs = Box::new(try!(transpile_expression(rhs, context)));
            Ok(dst::Expression::BinaryOperation(cl_binop, cl_lhs, cl_rhs))
        }
        &src::Expression::ArraySubscript(ref expr, ref sub) => {
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            let cl_sub = Box::new(try!(transpile_expression(sub, context)));
            Ok(dst::Expression::ArraySubscript(cl_expr, cl_sub))
        },
        &src::Expression::Member(ref expr, ref member_name) => {
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::Member(cl_expr, member_name.clone()))
        },
        &src::Expression::Call(ref func, ref params) => {
            let func_expr = try!(transpile_expression(func, context));
            let mut params_exprs: Vec<dst::Expression> = vec![];
            for param in params {
                let param_expr = try!(transpile_expression(param, context));
                params_exprs.push(param_expr);
            };
            Ok(dst::Expression::Call(Box::new(func_expr), params_exprs))
        },
        &src::Expression::Cast(ref cast_type, ref expr) => {
            let cl_type = try!(transpile_type(cast_type, context));
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::Cast(cl_type, cl_expr))
        },
        &src::Expression::Intrinsic(ref intrinsic) => transpile_intrinsic(intrinsic, context),
    }
}

fn transpile_vardef(vardef: &src::VarDef, context: &Context) -> Result<dst::VarDef, TranspileError> {
    Ok(dst::VarDef {
        name: try!(context.get_variable_id(&vardef.id)),
        typename: try!(transpile_type(&vardef.typename, context)),
        assignment: match &vardef.assignment { &None => None, &Some(ref expr) => Some(try!(transpile_expression(expr, context))) },
    })
}

#[allow(dead_code)]
fn transpile_condition(cond: &src::Condition, context: &Context) -> Result<dst::Condition, TranspileError> {
    match *cond {
        src::Condition::Expr(ref expr) => {
            let expr_ir = try!(transpile_expression(expr, &context));
            Ok(dst::Condition::Expr(expr_ir))
        },
        src::Condition::Assignment(ref vd) => {
            let cl_vardef = try!(transpile_vardef(vd, &context));
            Ok(dst::Condition::Assignment(cl_vardef))
        },
    }
}

fn transpile_statement(statement: &src::Statement, context: &mut Context) -> Result<dst::Statement, TranspileError> {
    match statement {
        &src::Statement::Expression(ref expr) => Ok(dst::Statement::Expression(try!(transpile_expression(expr, context)))),
        &src::Statement::Var(ref vd) => Ok(dst::Statement::Var(try!(transpile_vardef(vd, context)))),
        &src::Statement::Block(src::ScopeBlock(ref statements, ref decls)) => {
            context.push_scope(decls);
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::Block(cl_statements))
        },
        &src::Statement::If(ref cond, src::ScopeBlock(ref statements, ref decls)) => {
            context.push_scope(decls);
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::If(cl_cond, Box::new(dst::Statement::Block(cl_statements))))
        },
        &src::Statement::For(ref init, ref cond, ref update, src::ScopeBlock(ref statements, ref decls)) => {
            context.push_scope(decls);
            let cl_init = try!(transpile_condition(init, context));
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_update = try!(transpile_expression(update, context));
            let cl_statements= try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::For(cl_init, cl_cond, cl_update, Box::new(dst::Statement::Block(cl_statements))))
        },
        &src::Statement::While(ref cond, src::ScopeBlock(ref statements, ref decls)) => {
            context.push_scope(decls);
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::While(cl_cond, Box::new(dst::Statement::Block(cl_statements))))
        },
        &src::Statement::Return(_) => unimplemented!(),
    }
}

fn transpile_statements(statements: &[src::Statement], context: &mut Context) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_statements = vec![];
    for statement in statements {
        cl_statements.push(try!(transpile_statement(statement, context)));
    }
    Ok(cl_statements)
}

fn transpile_param(param: &src::FunctionParam, context: &Context) -> Result<dst::FunctionParam, TranspileError> {
    Ok(dst::FunctionParam {
        name: try!(context.get_variable_id(&param.id)),
        typename: try!(transpile_type(&param.typename, context)),
    })
}

fn transpile_params(params: &[src::FunctionParam], context: &Context) -> Result<Vec<dst::FunctionParam>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_param(param, context)));
    }
    Ok(cl_params)
}

fn transpile_kernel_input_semantic(param: &src::KernelParam, context: &Context) -> Result<dst::Statement, TranspileError> {
    match &param.1 {
        &src::KernelSemantic::DispatchThreadId => {
            let typename = try!(transpile_type(&param.1.get_type(), context));
            let assign = match &param.1 {
                &src::KernelSemantic::DispatchThreadId => {
                    let x = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(Box::new(dst::Expression::Literal(dst::Literal::UInt(0)))));
                    let y = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(Box::new(dst::Expression::Literal(dst::Literal::UInt(1)))));
                    let z = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(Box::new(dst::Expression::Literal(dst::Literal::UInt(2)))));
                    dst::Expression::Constructor(dst::Constructor::UInt3(Box::new(x), Box::new(y), Box::new(z)))
                },
                _ => unimplemented!(),
            };
            Ok(dst::Statement::Var(dst::VarDef {
                name: try!(context.get_variable_id(&param.0)),
                typename: typename,
                assignment: Some(assign),
            }))
        },
        &src::KernelSemantic::GroupId => unimplemented!(),
        &src::KernelSemantic::GroupIndex => unimplemented!(),
        &src::KernelSemantic::GroupThreadId => unimplemented!(),
    }
}

fn transpile_kernel_input_semantics(params: &[src::KernelParam], context: &Context) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_kernel_input_semantic(param, context)));
    }
    Ok(cl_params)
}

fn transpile_rootdefinition(rootdef: &src::RootDefinition, context: &mut Context) -> Result<Option<dst::RootDefinition>, TranspileError> {
    match rootdef {
        &src::RootDefinition::Struct(ref structdefinition) => {
            Ok(Some(dst::RootDefinition::Struct(dst::StructDefinition {
                name: try!(context.get_struct_name(&structdefinition.id)),
                members: try!(structdefinition.members.iter().fold(
                    Ok(vec![]),
                    |result, member| {
                        let mut vec = try!(result);
                        let dst_member = dst::StructMember {
                            name: member.name.clone(),
                            typename: try!(transpile_type(&member.typename, context)),
                        };
                        vec.push(dst_member);
                        Ok(vec)
                    }
                )),
            })))
        },
        &src::RootDefinition::SamplerState => unimplemented!{},
        &src::RootDefinition::ConstantBuffer(ref cb) => {
            let mut members = vec![];
            for member in &cb.members {
                let var_name = member.name.clone();
                let var_type = try!(transpile_type(&member.typename, context));
                members.push(dst::StructMember {
                    name: var_name,
                    typename: var_type,
                });
            };
            Ok(Some(dst::RootDefinition::Struct(dst::StructDefinition {
                name: try!(context.get_cbuffer_struct_name(&cb.id)),
                members: members,
            })))
        },
        &src::RootDefinition::GlobalVariable(ref gv) => {
            let global_name = try!(context.get_variable_id(&gv.id));
            if context.kernel_params.iter().any(|gp| { gp.name == global_name }) {
                return Ok(None)
            } else {
                return Err(TranspileError::GlobalFoundThatIsntInKernelParams(gv.clone()))
            }
        },
        &src::RootDefinition::Function(ref func) => {
            context.push_scope(&func.scope);
            let params = try!(transpile_params(&func.params, context));
            let body = try!(transpile_statements(&func.body, context));
            context.pop_scope();
            let cl_func = dst::FunctionDefinition {
                name: try!(context.get_function_name(&func.id)),
                returntype: try!(transpile_type(&func.returntype, context)),
                params: params,
                body: body,
            };
            Ok(Some(dst::RootDefinition::Function(cl_func)))
        }
        &src::RootDefinition::Kernel(ref kernel) => {
            context.push_scope(&kernel.scope);
            let mut body = try!(transpile_kernel_input_semantics(&kernel.params, context));
            let mut main_body = try!(transpile_statements(&kernel.body, context));
            context.pop_scope();
            body.append(&mut main_body);
            let cl_kernel = dst::Kernel {
                params: context.kernel_params.clone(),
                body: body,
                group_dimensions: dst::Dimension(kernel.group_dimensions.0, kernel.group_dimensions.1, kernel.group_dimensions.2),
            };
            Ok(Some(dst::RootDefinition::Kernel(cl_kernel)))
        }
    }
}

fn transpile_global(table: &src::GlobalTable, context: &Context) -> Result<KernelParams, TranspileError> {
    let mut global_params: KernelParams = vec![];
    // Todo: Pick slot numbers better
    for (_, id) in &table.constants {
        let cl_type = dst::Type::Pointer(
            dst::AddressSpace::Constant,
            Box::new(dst::Type::Struct(try!(context.get_cbuffer_struct_name(id))))
        );
        let cl_var = try!(context.get_cbuffer_instance_name(id));
        let param = dst::KernelParam {
            name: cl_var,
            typename: cl_type,
        };
        global_params.push(param);
    }
    for (_, global_entry) in &table.r_resources {
        let &src::Type(ref tyl, _) = &global_entry.typename;
        let cl_type = match tyl {
            &src::TypeLayout::Object(src::ObjectType::Buffer(ref data_type)) => {
                dst::Type::Pointer(dst::AddressSpace::Constant, Box::new(try!(transpile_datatype(data_type, context))))
            }
            _ => return Err(TranspileError::TypeIsNotAllowedAsGlobal(global_entry.typename.clone())),
        };
        let param = dst::KernelParam {
            name: try!(context.get_variable_id(&global_entry.id)),
            typename: cl_type,
        };
        global_params.push(param);
    }
    for (_, global_entry) in &table.rw_resources {
        let &src::Type(ref tyl, _) = &global_entry.typename;
        let cl_type = match tyl {
            &src::TypeLayout::Object(src::ObjectType::RWBuffer(ref data_type)) => {
                dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_datatype(data_type, context))))
            }
            _ => return Err(TranspileError::TypeIsNotAllowedAsGlobal(global_entry.typename.clone())),
        };
        let param = dst::KernelParam {
            name: try!(context.get_variable_id(&global_entry.id)),
            typename: cl_type,
        };
        global_params.push(param);
    }
    Ok(global_params)
}

pub fn transpile(module: &src::Module) -> Result<dst::Module, TranspileError> {

    let mut context = Context::new(&module.global_declarations);
    context.kernel_params = try!(transpile_global(&module.global_table, &context));

    let mut cl_defs = vec![];
    for rootdef in &module.root_definitions {
        match try!(transpile_rootdefinition(rootdef, &mut context)) {
            Some(def) => cl_defs.push(def),
            None => { },
        }
    }

    let cl_module = dst::Module {
        root_definitions: cl_defs,
    };

    Ok(cl_module)
}

#[test]
fn test_transpile() {

    use super::super::hlsl;
    use super::super::hlsl::typer::typeparse;

    let module = hlsl::ast::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            hlsl::ast::RootDefinition::GlobalVariable(hlsl::ast::GlobalVariable {
                name: "g_myInBuffer".to_string(),
                typename: hlsl::ast::Type::Object(hlsl::ast::ObjectType::Buffer(hlsl::ast::DataType::Scalar(hlsl::ast::ScalarType::Int))),
                slot: Some(hlsl::ast::GlobalSlot::ReadSlot(0)),
            }),
            hlsl::ast::RootDefinition::Function(hlsl::ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: hlsl::ast::Type::Void,
                params: vec![hlsl::ast::FunctionParam { name: "x".to_string(), typename: hlsl::ast::Type::uint(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            hlsl::ast::RootDefinition::Function(hlsl::ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: hlsl::ast::Type::Void,
                params: vec![hlsl::ast::FunctionParam { name: "x".to_string(), typename: hlsl::ast::Type::float(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            hlsl::ast::RootDefinition::Function(hlsl::ast::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: hlsl::ast::Type::Void,
                params: vec![],
                body: vec![
                    hlsl::ast::Statement::Empty,
                    hlsl::ast::Statement::Var(hlsl::ast::VarDef { name: "a".to_string(), typename: hlsl::ast::Type::uint(), assignment: None }),
                    hlsl::ast::Statement::Var(hlsl::ast::VarDef { name: "b".to_string(), typename: hlsl::ast::Type::uint(), assignment: None }),
                    hlsl::ast::Statement::Expression(
                        hlsl::ast::Expression::BinaryOperation(hlsl::ast::BinOp::Assignment,
                            Box::new(hlsl::ast::Expression::Variable("a".to_string())),
                            Box::new(hlsl::ast::Expression::Variable("b".to_string()))
                        )
                    ),
                    hlsl::ast::Statement::If(
                        hlsl::ast::Expression::Variable("b".to_string()),
                        Box::new(hlsl::ast::Statement::Empty),
                    ),
                    //hlsl::ast::Statement::Expression(
                    //    hlsl::ast::Expression::BinaryOperation(hlsl::ast::BinOp::Assignment,
                    //        Box::new(hlsl::ast::Expression::ArraySubscript(
                    //            Box::new(hlsl::ast::Expression::Variable("g_myInBuffer".to_string())),
                    //            Box::new(hlsl::ast::Expression::Literal(hlsl::ast::Literal::Int(0)))
                    //        )),
                    //        Box::new(hlsl::ast::Expression::Literal(hlsl::ast::Literal::Int(4)))
                    //    ),
                    //),
                    //hlsl::ast::Statement::Expression(
                    //    hlsl::ast::Expression::Call(
                    //        Box::new(hlsl::ast::Expression::Variable("myFunc".to_string())),
                    //        vec![
                    //            hlsl::ast::Expression::Variable("b".to_string())
                    //        ]
                    //    ),
                    //),
                ],
                attributes: vec![hlsl::ast::FunctionAttribute::NumThreads(8, 8, 1)],
            }),
        ],
    };
    let res = typeparse(&module);
    assert!(res.is_ok(), "{:?}", res);

    let clc_res = transpile(&res.unwrap());
    assert!(clc_res.is_ok(), "{:?}", clc_res);
}