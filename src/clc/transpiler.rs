
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use super::cir as dst;
use super::super::hlsl::ir as src;

#[derive(PartialEq, Debug, Clone)]
pub enum TranspileError {
    Unknown,

    TypeIsNotAllowedAsGlobal(src::Type),
    CouldNotGetEquivalentType(src::Type),
    CouldNotGetEquivalentDataType(src::DataType),

    GlobalFoundThatIsntInKernelParams(src::GlobalVariable),

    UnknownFunctionId(src::FunctionId),
}

type KernelParams = Vec<dst::KernelParam>;

struct Context {
    kernel_params: KernelParams,
    function_name_map: HashMap<src::FunctionId, String>,
}

impl Context {
    fn get_function(&self, id: &src::FunctionId) -> Result<dst::Expression, TranspileError> {
        match self.function_name_map.get(id) {
            Some(v) => Ok(dst::Expression::Variable(v.clone())),
            None => Err(TranspileError::UnknownFunctionId(id.clone())),
        }
    }
}

fn transpile_scalartype(scalartype: &src::ScalarType) -> Result<dst::Scalar, TranspileError> {
    match scalartype {
        &src::ScalarType::Bool => Ok(dst::Scalar::Bool),
        &src::ScalarType::Int => Ok(dst::Scalar::Int),
        &src::ScalarType::UInt => Ok(dst::Scalar::UInt),
        &src::ScalarType::Half => Ok(dst::Scalar::Half),
        &src::ScalarType::Float => Ok(dst::Scalar::Float),
        &src::ScalarType::Double => Ok(dst::Scalar::Double),
    }
}

fn transpile_datatype(datatype: &src::DataType) -> Result<dst::Type, TranspileError> {
    match datatype {
        &src::DataType::Scalar(ref scalar) => Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar)))),
        &src::DataType::Vector(ref scalar, 1) => Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar)))),
        &src::DataType::Vector(ref scalar, 2) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Two)),
        &src::DataType::Vector(ref scalar, 3) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Three)),
        &src::DataType::Vector(ref scalar, 4) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Four)),
        ty => return Err(TranspileError::CouldNotGetEquivalentDataType(ty.clone())),
    }
}

fn transpile_type(hlsltype: &src::Type) -> Result<dst::Type, TranspileError> {
    match hlsltype {
        &src::Type::Void => Ok(dst::Type::Void),
        &src::Type::Structured(src::StructuredType::Data(ref data_type)) => transpile_datatype(data_type),
        ty => return Err(TranspileError::CouldNotGetEquivalentType(ty.clone())),
    }
}

fn transpile_binop(binop: &src::BinOp) -> Result<dst::BinOp, TranspileError> {
    match binop {
        &src::BinOp::Add => Ok(dst::BinOp::Add),
        &src::BinOp::Subtract => Ok(dst::BinOp::Subtract),
        &src::BinOp::Multiply => Ok(dst::BinOp::Multiply),
        &src::BinOp::Divide => Ok(dst::BinOp::Divide),
        &src::BinOp::Modulus => Ok(dst::BinOp::Modulus),
        &src::BinOp::Assignment => Ok(dst::BinOp::Assignment),
    }
}

fn transpile_literal(lit: &src::Literal) -> Result<dst::Literal, TranspileError> {
    match lit {
        &src::Literal::Int(i) => Ok(dst::Literal::Int(i)),
        &src::Literal::Uint(i) => Ok(dst::Literal::UInt(i)),
        &src::Literal::Long(i) => Ok(dst::Literal::Long(i)),
        &src::Literal::Half(f) => Ok(dst::Literal::Half(f)),
        &src::Literal::Float(f) => Ok(dst::Literal::Float(f)),
        &src::Literal::Double(f) => Ok(dst::Literal::Double(f)),
    }
}

fn transpile_intrinsic(intrinsic: &src::Intrinsic, context: &Context) -> Result<dst::Expression, TranspileError> {
    match intrinsic {
        &src::Intrinsic::Float4(ref x, ref y, ref z, ref w) => {
            Ok(dst::Expression::Constructor(dst::Constructor::Float4(
                Box::new(try!(transpile_expression(x, context))),
                Box::new(try!(transpile_expression(y, context))),
                Box::new(try!(transpile_expression(z, context))),
                Box::new(try!(transpile_expression(w, context)))
            )))
        },
        &src::Intrinsic::BufferLoad(ref buffer, ref loc) => {
            let cl_buffer = Box::new(try!(transpile_expression(buffer, context)));
            let cl_loc = Box::new(try!(transpile_expression(loc, context)));
            Ok(dst::Expression::ArraySubscript(cl_buffer, cl_loc))
        },
        &src::Intrinsic::StructuredBufferLoad(_, _) => unimplemented!(),
    }
}

fn transpile_expression(expression: &src::Expression, context: &Context) -> Result<dst::Expression, TranspileError> {
    match expression {
        &src::Expression::Literal(ref lit) => Ok(dst::Expression::Literal(try!(transpile_literal(lit)))),
        &src::Expression::Variable(ref name) => Ok(dst::Expression::Variable(name.clone())),
        &src::Expression::Function(ref id) => context.get_function(id),
        &src::Expression::UnaryOperation(_, _) => unimplemented!(),
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
        &src::Expression::Member(_, _) => unimplemented!(),
        &src::Expression::Call(ref func, ref params) => {
            let func_expr = try!(transpile_expression(func, context));
            let mut params_exprs: Vec<dst::Expression> = vec![];
            for param in params {
                let param_expr = try!(transpile_expression(param, context));
                params_exprs.push(param_expr);
            };
            Ok(dst::Expression::Call(Box::new(func_expr), params_exprs))
        },
        &src::Expression::Cast(_, _) => unimplemented!(),
        &src::Expression::Intrinsic(ref intrinsic) => transpile_intrinsic(intrinsic, context),
    }
}

fn transpile_vardef(vardef: &src::VarDef, context: &Context) -> Result<dst::VarDef, TranspileError> {
    Ok(dst::VarDef {
        name: vardef.name.clone(),
        typename: try!(transpile_type(&vardef.typename)),
        assignment: match &vardef.assignment { &None => None, &Some(ref expr) => Some(try!(transpile_expression(expr, context))) },
    })
}

fn transpile_statement(statement: &src::Statement, context: &Context) -> Result<dst::Statement, TranspileError> {
    match statement {
        &src::Statement::Expression(ref expr) => Ok(dst::Statement::Expression(try!(transpile_expression(expr, context)))),
        &src::Statement::Var(ref vd) => Ok(dst::Statement::Var(try!(transpile_vardef(vd, context)))),
        &src::Statement::Block(_) => unimplemented!(),
        &src::Statement::If(_, _) => unimplemented!(),
        &src::Statement::For(_, _, _, _) => unimplemented!(),
        &src::Statement::While(_, _) => unimplemented!(),
        &src::Statement::Return(_) => unimplemented!(),
    }
}

fn transpile_statements(statements: &[src::Statement], context: &Context) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_statements = vec![];
    for statement in statements {
        cl_statements.push(try!(transpile_statement(statement, context)));
    }
    Ok(cl_statements)
}

fn transpile_param(param: &src::FunctionParam) -> Result<dst::FunctionParam, TranspileError> {
    Ok(dst::FunctionParam {
        name: param.name.clone(),
        typename: try!(transpile_type(&param.typename)),
    })
}

fn transpile_params(params: &[src::FunctionParam]) -> Result<Vec<dst::FunctionParam>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_param(param)));
    }
    Ok(cl_params)
}

fn transpile_kernel_input_semantic(param: &src::KernelParam) -> Result<dst::Statement, TranspileError> {
    match &param.1 {
        &src::KernelSemantic::DispatchThreadId => {
            let typename = try!(transpile_type(&param.1.get_type()));
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
                name: param.0.clone(),
                typename: typename,
                assignment: Some(assign),
            }))
        },
        &src::KernelSemantic::GroupId => unimplemented!(),
        &src::KernelSemantic::GroupIndex => unimplemented!(),
        &src::KernelSemantic::GroupThreadId => unimplemented!(),
    }
}

fn transpile_kernel_input_semantics(params: &[src::KernelParam]) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_kernel_input_semantic(param)));
    }
    Ok(cl_params)
}

fn transpile_rootdefinition(rootdef: &src::RootDefinition, context: &Context) -> Result<Option<dst::RootDefinition>, TranspileError> {
    match rootdef {
        &src::RootDefinition::Struct(_) => unimplemented!{},
        &src::RootDefinition::SamplerState => unimplemented!{},
        &src::RootDefinition::ConstantBuffer(_) => unimplemented!{},
        &src::RootDefinition::GlobalVariable(ref gv) => {
            if context.kernel_params.iter().any(|gp| { gp.name == gv.name }) {
                return Ok(None)
            } else {
                return Err(TranspileError::GlobalFoundThatIsntInKernelParams(gv.clone()))
            }
        },
        &src::RootDefinition::Function(ref func) => {
            let cl_func = dst::FunctionDefinition {
                name: func.original_name.clone() + "_" + &func.id.to_string(),
                returntype: try!(transpile_type(&func.returntype)),
                params: try!(transpile_params(&func.params)),
                body: try!(transpile_statements(&func.body, context)),
            };
            Ok(Some(dst::RootDefinition::Function(cl_func)))
        }
        &src::RootDefinition::Kernel(ref kernel) => {
            let mut body = try!(transpile_kernel_input_semantics(&kernel.params));
            let mut main_body = try!(transpile_statements(&kernel.body, context));
            body.append(&mut main_body);
            let cl_kernel = dst::Kernel {
                params: context.kernel_params.clone(),
                body: body,
            };
            Ok(Some(dst::RootDefinition::Kernel(cl_kernel)))
        }
    }
}

fn transpile_global(table: &src::GlobalTable) -> Result<KernelParams, TranspileError> {
    let mut global_params: KernelParams = vec![];
    // Todo: Pick slot numbers better
    for (_, global_entry) in &table.r_resources {
        let cl_type = match &global_entry.typename {
            &src::Type::Object(src::ObjectType::Buffer(ref data_type)) => {
                dst::Type::Pointer(dst::AddressSpace::Constant, Box::new(try!(transpile_datatype(data_type))))
            }
            ty => return Err(TranspileError::TypeIsNotAllowedAsGlobal(ty.clone())),
        };
        let param = dst::KernelParam {
            name: global_entry.name.clone(),
            typename: cl_type,
        };
        global_params.push(param);
    }
    for (_, global_entry) in &table.rw_resources {
        let cl_type = match &global_entry.typename {
            &src::Type::Object(src::ObjectType::RWBuffer(ref data_type)) => {
                dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_datatype(data_type))))
            }
            ty => return Err(TranspileError::TypeIsNotAllowedAsGlobal(ty.clone())),
        };
        let param = dst::KernelParam {
            name: global_entry.name.clone(),
            typename: cl_type,
        };
        global_params.push(param);
    }
    Ok(global_params)
}

fn create_function_names(rootdefs: &[src::RootDefinition]) -> Result<HashMap<src::FunctionId, String>, TranspileError> {
    let mut grouped_functions: HashMap<String, Vec<src::FunctionId>> = HashMap::new();
    for rootdef in rootdefs {
        match rootdef {
            &src::RootDefinition::Function(ref func) => {
                match grouped_functions.entry(func.original_name.clone()) {
                    Entry::Occupied(mut occupied) => { occupied.get_mut().push(func.id); },
                    Entry::Vacant(vacant) => { vacant.insert(vec![func.id]); },
                }
            },
            _ => { },
        }
    };
    let mut name_map: HashMap<src::FunctionId, String> = HashMap::new();
    for (key, ids) in grouped_functions {
        assert!(ids.len() > 0);
        if ids.len() == 1 {
            let ret = name_map.insert(ids[0], key);
            assert_eq!(ret, None);
        } else {
            for (idx, id) in ids.iter().enumerate() {
                let gen = key.clone() + "_" + &idx.to_string();
                let ret = name_map.insert(*id, gen);
                assert_eq!(ret, None);
            }
        }
    };
    Ok(name_map)
}

pub fn transpile(module: &src::Module) -> Result<dst::Module, TranspileError> {

    let context = Context {
        kernel_params: try!(transpile_global(&module.global_table)),
        function_name_map: try!(create_function_names(&module.root_definitions)),
    };

    let mut cl_defs = vec![];
    for rootdef in &module.root_definitions {
        match try!(transpile_rootdefinition(rootdef, &context)) {
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
                    //hlsl::ast::Statement::If(
                    //    hlsl::ast::Condition::Assignment(hlsl::ast::VarDef {
                    //        name: "c".to_string(),
                    //        typename: hlsl::ast::Type::uint(),
                    //        assignment: Some(hlsl::ast::Expression::Variable("a".to_string()))
                    //    }),
                    //    Box::new(hlsl::ast::Statement::Empty),
                    //),
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