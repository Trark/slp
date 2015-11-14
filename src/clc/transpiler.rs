
use super::cir as dst;
use super::super::hlsl::ir as src;

#[derive(PartialEq, Debug, Clone)]
pub enum TranspileError {
    Unknown,

    TypeIsNotAllowedAsGlobal(src::Type),
    CouldNotGetEquivalentType(src::Type),
    CouldNotGetEquivalentDataType(src::DataType),

    GlobalFoundThatIsntInKernelParams(src::GlobalVariable),
}

type KernelParams = Vec<dst::KernelParam>;

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

fn transpile_intrinsic(intrinsic: &src::Intrinsic) -> Result<dst::Expression, TranspileError> {
    match intrinsic {
        &src::Intrinsic::Float4(_, _, _, _) => unimplemented!(),
        &src::Intrinsic::BufferLoad(ref buffer, ref loc) => {
            let cl_buffer = Box::new(try!(transpile_expression(buffer)));
            let cl_loc = Box::new(try!(transpile_expression(loc)));
            Ok(dst::Expression::ArraySubscript(cl_buffer, cl_loc))
        },
        &src::Intrinsic::StructuredBufferLoad(_, _) => unimplemented!(),
    }
}

fn transpile_expression(expression: &src::Expression) -> Result<dst::Expression, TranspileError> {
    match expression {
        &src::Expression::Literal(ref lit) => Ok(dst::Expression::Literal(try!(transpile_literal(lit)))),
        &src::Expression::Variable(ref name) => Ok(dst::Expression::Variable(name.clone())),
        &src::Expression::UnaryOperation(_, _) => Err(TranspileError::Unknown),
        &src::Expression::BinaryOperation(ref binop, ref lhs, ref rhs) => {
            let cl_binop = try!(transpile_binop(binop));
            let cl_lhs = Box::new(try!(transpile_expression(lhs)));
            let cl_rhs = Box::new(try!(transpile_expression(rhs)));
            Ok(dst::Expression::BinaryOperation(cl_binop, cl_lhs, cl_rhs))
        }
        &src::Expression::ArraySubscript(ref expr, ref sub) => {
            let cl_expr = Box::new(try!(transpile_expression(expr)));
            let cl_sub = Box::new(try!(transpile_expression(sub)));
            Ok(dst::Expression::ArraySubscript(cl_expr, cl_sub))
        },
        &src::Expression::Member(_, _) => Err(TranspileError::Unknown),
        &src::Expression::Call(_, _) => Err(TranspileError::Unknown),
        &src::Expression::Cast(_, _) => Err(TranspileError::Unknown),
        &src::Expression::Intrinsic(ref intrinsic) => transpile_intrinsic(intrinsic),
    }
}

fn transpile_vardef(vardef: &src::VarDef) -> Result<dst::VarDef, TranspileError> {
    Ok(dst::VarDef {
        name: vardef.name.clone(),
        typename: try!(transpile_type(&vardef.typename)),
        assignment: match &vardef.assignment { &None => None, &Some(ref expr) => Some(try!(transpile_expression(expr))) },
    })
}

fn transpile_statement(statement: &src::Statement) -> Result<dst::Statement, TranspileError> {
    match statement {
        &src::Statement::Expression(ref expr) => Ok(dst::Statement::Expression(try!(transpile_expression(expr)))),
        &src::Statement::Var(ref vd) => Ok(dst::Statement::Var(try!(transpile_vardef(vd)))),
        &src::Statement::Block(_) => Err(TranspileError::Unknown),
        &src::Statement::If(_, _) => Err(TranspileError::Unknown),
        &src::Statement::For(_, _, _, _) => Err(TranspileError::Unknown),
        &src::Statement::While(_, _) => Err(TranspileError::Unknown),
        &src::Statement::Return(_) => Err(TranspileError::Unknown),
    }
}

fn transpile_statements(statements: &[src::Statement]) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_statements = vec![];
    for statement in statements {
        cl_statements.push(try!(transpile_statement(statement)));
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

fn transpile_rootdefinition(rootdef: &src::RootDefinition, global_params: &KernelParams) -> Result<Option<dst::RootDefinition>, TranspileError> {
    match rootdef {
        &src::RootDefinition::Struct(_) => unimplemented!{},
        &src::RootDefinition::SamplerState => unimplemented!{},
        &src::RootDefinition::ConstantBuffer(_) => unimplemented!{},
        &src::RootDefinition::GlobalVariable(ref gv) => {
            if global_params.iter().any(|gp| { gp.name == gv.name }) {
                return Ok(None)
            } else {
                return Err(TranspileError::GlobalFoundThatIsntInKernelParams(gv.clone()))
            }
        },
        &src::RootDefinition::Function(ref func) => {
            let cl_func = dst::FunctionDefinition {
                name: func.name.clone(),
                returntype: try!(transpile_type(&func.returntype)),
                params: try!(transpile_params(&func.params)),
                body: try!(transpile_statements(&func.body)),
            };
            Ok(Some(dst::RootDefinition::Function(cl_func)))
        }
        &src::RootDefinition::Kernel(ref kernel) => {
            let mut body = try!(transpile_kernel_input_semantics(&kernel.params));
            let mut main_body = try!(transpile_statements(&kernel.body));
            body.append(&mut main_body);
            let cl_kernel = dst::Kernel {
                params: global_params.clone(),
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

pub fn transpile(module: &src::Module) -> Result<dst::Module, TranspileError> {

    let global_params = try!(transpile_global(&module.global_table));
    let mut cl_defs = vec![];

    for rootdef in &module.root_definitions {
        match try!(transpile_rootdefinition(rootdef, &global_params)) {
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
    use super::super::hlsl::ast_to_ir;

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
    let res = ast_to_ir::parse(&module);
    assert!(res.is_ok(), "{:?}", res);

    let clc_res = transpile(&res.unwrap());
    assert!(clc_res.is_ok(), "{:?}", clc_res);
}