
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use super::ir;
use super::ast;

#[derive(PartialEq, Debug, Clone)]
pub enum ParseError {
    Unimplemented,

    ValueAlreadyDefined(String, ExpressionType, ExpressionType),
    StructAlreadyDefined(String),

    ConstantSlotAlreadyUsed(String, String),
    ReadResourceSlotAlreadyUsed(String, String),
    ReadWriteResourceSlotAlreadyUsed(String, String),

    UnknownIdentifier(String),

    ArrayIndexingNonArrayType,
    ArraySubscriptIndexNotInteger,

    CallOnNonFunction(String),

    FunctionPassedToAnotherFunction(String),
    FunctionArgumentTypeMismatch(Vec<FunctionType>, Vec<ir::Type>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionType(pub Box<ir::Type>, pub Vec<ir::Type>);

#[derive(PartialEq, Debug, Clone)]
pub enum ExpressionType {
    Value(ir::Type),
    Function(Vec<FunctionType>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Context {
    structs: HashMap<String, ir::StructDefinition>,

    functions: HashMap<String, Vec<FunctionType>>,

    // Variables (and functions) visible in the current context. Maps names to types
    // One map per scope level (nearer the front is a wider scope)
    variables: Vec<HashMap<String, ir::Type>>,
}

impl Context {
    pub fn new() -> Context {
        Context { structs: HashMap::new(), functions: HashMap::new(), variables: vec![HashMap::new()] }
    }

    pub fn insert_variable(&mut self, name: String, typename: ir::Type) -> Result<(), ParseError> {
        assert!(self.variables.len() > 0);
        if let Ok(ety) = self.find_variable(&name) {
            return Err(ParseError::ValueAlreadyDefined(name, ety, ExpressionType::Value(typename)))
        };
        let last = self.variables.len() - 1;
        match self.variables[last].entry(name.clone()) {
            Entry::Occupied(occupied) => Err(ParseError::ValueAlreadyDefined(name, ExpressionType::Value(occupied.get().clone()), ExpressionType::Value(typename))),
            Entry::Vacant(vacant) => { vacant.insert(typename); Ok(()) },
        }
    }

    pub fn insert_function(&mut self, name: String, function_type: FunctionType) -> Result<(), ParseError> {
        if let Ok(ExpressionType::Value(ty)) = self.find_variable(&name) {
            return Err(ParseError::ValueAlreadyDefined(name, ExpressionType::Value(ty), ExpressionType::Function(vec![function_type])))
        };
        match self.functions.entry(name.clone()) {
            Entry::Occupied(mut occupied) => {
                for &FunctionType(_, ref args) in occupied.get() {
                    if *args == function_type.1 {
                        return Err(ParseError::ValueAlreadyDefined(name, ExpressionType::Function(occupied.get().clone()), ExpressionType::Function(vec![function_type])))
                    }
                };
                occupied.get_mut().push(function_type);
                Ok(())
            },
            Entry::Vacant(vacant) => { vacant.insert(vec![function_type]); Ok(()) },
        }
    }

    pub fn find_variable(&self, name: &String) -> Result<ExpressionType, ParseError> {
        match self.functions.get(name) {
            Some(tys) => return Ok(ExpressionType::Function(tys.clone())),
            None => { },
        }
        for map in self.variables.iter().rev() {
            match map.get(name) {
                Some(ty) => return Ok(ExpressionType::Value(ty.clone())),
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

struct TypedExpression(ir::Expression, ExpressionType);

fn get_function_debug_name(expr: &ir::Expression) -> String {
    match expr {
        &ir::Expression::Variable(ref s) => s.clone(),
        _ => "<unknown>".to_string()
    }
}

fn find_function_type(func_expr: &ir::Expression, overloads: ExpressionType, actual_params: Vec<ir::Type>) -> Result<ir::Type, ParseError> {
    match overloads {
        ExpressionType::Function(fts) => {
            for ft in &fts {
                if ft.1 == actual_params {
                    return Ok(*ft.0.clone())
                };
            };
            Err(ParseError::FunctionArgumentTypeMismatch(fts, actual_params))
        },
        _ => Err(ParseError::CallOnNonFunction(get_function_debug_name(func_expr))),
    }
}

fn parse_expr(ast: &ast::Expression, context: &Context) -> Result<TypedExpression, ParseError> {
    match ast {
        &ast::Expression::LiteralInt(i) => Ok(TypedExpression(ir::Expression::LiteralInt(i), ExpressionType::Value(ir::Type::int()))),
        &ast::Expression::LiteralUint(i) => Ok(TypedExpression(ir::Expression::LiteralUint(i), ExpressionType::Value(ir::Type::uint()))),
        &ast::Expression::LiteralLong(i) => Ok(TypedExpression(ir::Expression::LiteralLong(i), ExpressionType::Value(ir::Type::long()))),
        &ast::Expression::LiteralFloat(f) => Ok(TypedExpression(ir::Expression::LiteralFloat(f), ExpressionType::Value(ir::Type::float()))),
        &ast::Expression::LiteralDouble(f) => Ok(TypedExpression(ir::Expression::LiteralDouble(f), ExpressionType::Value(ir::Type::double()))),
        &ast::Expression::Variable(ref s) => {
            let var_type = try!(context.find_variable(s));
            Ok(TypedExpression(ir::Expression::Variable(s.clone()), var_type))
        },
        &ast::Expression::UnaryOperation(ref op, ref expr) => {
            let expr_ir = try!(parse_expr(expr, context));
            Ok(TypedExpression(ir::Expression::UnaryOperation(op.clone(), Box::new(expr_ir.0)), expr_ir.1.clone()))
        },
        &ast::Expression::BinaryOperation(ref op, ref lhs, ref rhs) => {
            let lhs_ir = try!(parse_expr(lhs, context));
            let rhs_ir = try!(parse_expr(rhs, context));
            Ok(TypedExpression(ir::Expression::BinaryOperation(op.clone(), Box::new(lhs_ir.0), Box::new(rhs_ir.0)), lhs_ir.1.clone()))
        },
        &ast::Expression::ArraySubscript(ref array, ref subscript) => {
            let array_ir = try!(parse_expr(array, context));
            let subscript_ir = try!(parse_expr(subscript, context));
            let indexed_type = match array_ir.1 {
                ExpressionType::Value(ir::Type::Object(ir::ObjectType::Buffer(data_type))) |
                ExpressionType::Value(ir::Type::Object(ir::ObjectType::RWBuffer(data_type))) => {
                    ir::Type::Structured(ir::StructuredType::Data(data_type))
                },
                ExpressionType::Value(ir::Type::Object(ir::ObjectType::StructuredBuffer(structured_type))) |
                ExpressionType::Value(ir::Type::Object(ir::ObjectType::RWStructuredBuffer(structured_type))) => {
                    ir::Type::Structured(structured_type)
                },
                _ => return Err(ParseError::ArrayIndexingNonArrayType),
            };
            match subscript_ir.1 {
                ExpressionType::Value(ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::Int)))) |
                ExpressionType::Value(ir::Type::Structured(ir::StructuredType::Data(ir::DataType::Scalar(ir::ScalarType::UInt)))) => { },
                _ => return Err(ParseError::ArraySubscriptIndexNotInteger),
            };
            Ok(TypedExpression(ir::Expression::ArraySubscript(Box::new(array_ir.0), Box::new(subscript_ir.0)), ExpressionType::Value(indexed_type)))
        },
        &ast::Expression::Member(ref composite, ref member) => {
            let composite_ir = try!(parse_expr(composite, context));
            Ok(TypedExpression(ir::Expression::Member(Box::new(composite_ir.0), member.clone()), ExpressionType::Value(ir::Type::Void)))
        },
        &ast::Expression::Call(ref func, ref params) => {
            let func_ir = try!(parse_expr(func, context));
            let mut params_ir: Vec<ir::Expression> = vec![];
            let mut params_types: Vec<ir::Type> = vec![];
            for param in params {
                let TypedExpression(expr_ir, expr_type) = try!(parse_expr(param, context));
                params_ir.push(expr_ir);
                match expr_type {
                    ExpressionType::Value(ty) => params_types.push(ty),
                    ExpressionType::Function(_) => return Err(ParseError::FunctionPassedToAnotherFunction(get_function_debug_name(&func_ir.0)))
                };
            };
            let ty = try!(find_function_type(&func_ir.0, func_ir.1, params_types));
            Ok(TypedExpression(ir::Expression::Call(Box::new(func_ir.0), params_ir), ExpressionType::Value(ty)))
        },
        &ast::Expression::Cast(ref ty, ref expr) => {
            let expr_ir = try!(parse_expr(expr, context));
            Ok(TypedExpression(ir::Expression::Cast(ty.clone(), Box::new(expr_ir.0)), ExpressionType::Value(ir::Type::Void)))
        },
    }
}

fn parse_vardef(ast: &ast::VarDef, context: Context) -> Result<(ir::VarDef, Context), ParseError> {
    let assign_ir = match ast.assignment {
        Some(ref expr) => Some(try!(parse_expr(expr, &context)).0),
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
            let expr_ir = try!(parse_expr(expr, &context)).0;
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
            Ok((Some(ir::Statement::Expression(try!(parse_expr(expr, &context)).0)), context))
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
            Ok((Some(ir::Statement::Return(try!(parse_expr(expr, &context)).0)), context))
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

fn parse_rootdefinition(ast: &ast::RootDefinition, context: Context, globals: &mut ir::GlobalTable) -> Result<(ir::RootDefinition, Context), ParseError> {
    let mut next_context = context;
    let res = match ast {
        &ast::RootDefinition::Struct(ref sd) => {
            let struct_def = sd.clone();
            match next_context.structs.insert(struct_def.name.clone(), struct_def.clone()) {
                Some(_) => return Err(ParseError::StructAlreadyDefined(struct_def.name.clone())),
                None => { },
            };
            ir::RootDefinition::Struct(struct_def)
        },
        &ast::RootDefinition::SamplerState => ir::RootDefinition::SamplerState,
        &ast::RootDefinition::ConstantBuffer(ref cb) => {
            let cb_ir = ir::ConstantBuffer { name: cb.name.clone(), members: cb.members.clone() };
            try!(next_context.insert_variable(cb_ir.name.clone(), ir::Type::custom(&cb_ir.name[..])));
            match cb.slot {
                Some(ast::ConstantSlot(slot)) => {
                    match globals.constants.insert(slot, cb_ir.name.clone()) {
                        Some(currently_used_by) => return Err(ParseError::ConstantSlotAlreadyUsed(currently_used_by.clone(), cb_ir.name.clone())),
                        None => { },
                    }
                },
                None => { },
            }
            ir::RootDefinition::ConstantBuffer(cb_ir)
        },
        &ast::RootDefinition::GlobalVariable(ref gv) => {
            let gv_ir = ir::GlobalVariable { name: gv.name.clone(), typename: gv.typename.clone() };
            try!(next_context.insert_variable(gv_ir.name.clone(), gv_ir.typename.clone()));
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
            ir::RootDefinition::GlobalVariable(gv_ir)
        }
        &ast::RootDefinition::Function(ref fd) => {
            let body_ir = {
                let mut scoped_context = next_context.scoped();
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
            let func_type = FunctionType(
                Box::new(fd_ir.returntype.clone()),
                fd_ir.params.iter().map(|p| { p.typename.clone() }).collect()
            );
            try!(next_context.insert_function(fd_ir.name.clone(), func_type));
            ir::RootDefinition::Function(fd_ir)
        }

    };
    Ok((res, next_context))
}

pub fn parse(ast: &ast::Module) -> Result<ir::Module, ParseError> {
    let mut context = Context::new();

    let mut ir = ir::Module { entry_point: ast.entry_point.clone(), global_table: ir::GlobalTable::new(), root_definitions: vec![] };

    for def in &ast.root_definitions {
        let (def_ir, next_context) = try!(parse_rootdefinition(&def, context, &mut ir.global_table));
        ir.root_definitions.push(def_ir);
        context = next_context;
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
                typename: ast::Type::Object(ast::ObjectType::Buffer(ast::DataType::Vector(ast::ScalarType::Float, 4))),
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
                                Box::new(ast::Expression::LiteralInt(0))
                            )),
                            Box::new(ast::Expression::LiteralInt(4))
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
                attributes: vec![],
            }),
        ],
    };
    let res = parse(&module);
    assert!(res.is_ok());
}
