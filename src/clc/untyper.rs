
use std::error;
use std::fmt;
use super::cil as src;
use super::cst as dst;

#[derive(PartialEq, Debug, Clone)]
pub enum UntyperError {
}

impl error::Error for UntyperError {
    fn description(&self) -> &str {
        match *self {
        }
    }
}

impl fmt::Display for UntyperError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}


fn result_map<T, G, F>(func: F, inputs: &[T]) -> Result<Vec<G>, UntyperError> where F: Fn(&T) -> Result<G, UntyperError> {
    inputs.iter().fold(Ok(vec![]), |vec, next| {
            let mut vec = match vec { Ok(vec) => vec, Err(err) => return Err(err) };
            vec.push(try!(func(next)));
            Ok(vec)
        }
    )
}

fn untype_type(ty: &src::Type) -> Result<dst::Type, UntyperError> {
    Ok(match *ty {
        src::Type::Void => dst::Type::Void,
        src::Type::Bool => dst::Type::Bool,
        src::Type::Scalar(ref scalar) => dst::Type::Scalar(scalar.clone()),
        src::Type::Vector(ref scalar, ref dim) => dst::Type::Vector(scalar.clone(), dim.clone()),
        src::Type::SizeT => dst::Type::SizeT,
        src::Type::PtrDiffT => dst::Type::PtrDiffT,
        src::Type::IntPtrT => dst::Type::IntPtrT,
        src::Type::UIntPtrT => dst::Type::UIntPtrT,
        src::Type::Struct(ref identifier) => dst::Type::Struct(identifier.clone()),
        src::Type::Pointer(ref address_space, ref inner) => dst::Type::Pointer(address_space.clone(), Box::new(try!(untype_type(inner)))),
        src::Type::Array(ref inner, dim) => dst::Type::Array(Box::new(try!(untype_type(inner))), dim),
        src::Type::Image1D(ref address_space) => dst::Type::Image1D(address_space.clone()),
        src::Type::Image1DBuffer(ref address_space) => dst::Type::Image1DBuffer(address_space.clone()),
        src::Type::Image1DArray(ref address_space) => dst::Type::Image1DArray(address_space.clone()),
        src::Type::Image2D(ref address_space) => dst::Type::Image2D(address_space.clone()),
        src::Type::Image2DArray(ref address_space) => dst::Type::Image2DArray(address_space.clone()),
        src::Type::Image2DDepth(ref address_space) => dst::Type::Image2DDepth(address_space.clone()),
        src::Type::Image2DArrayDepth(ref address_space) => dst::Type::Image2DArrayDepth(address_space.clone()),
        src::Type::Image3D(ref address_space) => dst::Type::Image3D(address_space.clone()),
        src::Type::Image3DArray(ref address_space) => dst::Type::Image3DArray(address_space.clone()),
        src::Type::Sampler => dst::Type::Sampler,
        src::Type::Queue => dst::Type::Queue,
        src::Type::NDRange => dst::Type::NDRange,
        src::Type::ClkEvent => dst::Type::ClkEvent,
        src::Type::ReserveId => dst::Type::ReserveId,
        src::Type::Event => dst::Type::Event,
        src::Type::MemFenceFlags => dst::Type::MemFenceFlags,
    })
}

fn untype_constructor(cons: &src::Constructor) -> Result<dst::Constructor, UntyperError> {
    Ok(match *cons {
        src::Constructor::UInt3(ref e1, ref e2, ref e3) => {
            let u1 = Box::new(try!(untype_expression(e1)));
            let u2 = Box::new(try!(untype_expression(e2)));
            let u3 = Box::new(try!(untype_expression(e3)));
            dst::Constructor::UInt3(u1, u2, u3)
        },
        src::Constructor::Float4(ref e1, ref e2, ref e3, ref e4) => {
            let u1 = Box::new(try!(untype_expression(e1)));
            let u2 = Box::new(try!(untype_expression(e2)));
            let u3 = Box::new(try!(untype_expression(e3)));
            let u4 = Box::new(try!(untype_expression(e4)));
            dst::Constructor::Float4(u1, u2, u3, u4)
        },
    })
}

fn untype_intrinsic(instrinic: &src::Intrinsic) -> Result<dst::Intrinsic, UntyperError> {
    Ok(match *instrinic {
        src::Intrinsic::GetGlobalId(ref expr) => dst::Intrinsic::GetGlobalId(Box::new(try!(untype_expression(expr)))),
    })
}

fn untype_expression(expression: &src::Expression) -> Result<dst::Expression, UntyperError> {
    Ok(match *expression {
        src::Expression::Literal(ref literal) => dst::Expression::Literal(literal.clone()),
        src::Expression::Constructor(ref cons) => dst::Expression::Constructor(try!(untype_constructor(cons))),
        src::Expression::Variable(ref name) => dst::Expression::Variable(name.clone()),
        src::Expression::UnaryOperation(ref un, ref expr) => dst::Expression::UnaryOperation(
            un.clone(),
            Box::new(try!(untype_expression(expr)))
        ),
        src::Expression::BinaryOperation(ref bin, ref e1, ref e2) => dst::Expression::BinaryOperation(
            bin.clone(),
            Box::new(try!(untype_expression(e1))),
            Box::new(try!(untype_expression(e2)))
        ),
        src::Expression::TernaryConditional(ref e1, ref e2, ref e3) => dst::Expression::TernaryConditional(
            Box::new(try!(untype_expression(e1))),
            Box::new(try!(untype_expression(e2))),
            Box::new(try!(untype_expression(e3)))
        ),
        src::Expression::ArraySubscript(ref arr, ref ind) => dst::Expression::ArraySubscript(
            Box::new(try!(untype_expression(arr))),
            Box::new(try!(untype_expression(ind)))
        ),
        src::Expression::Member(ref expr, ref name) => dst::Expression::Member(Box::new(try!(untype_expression(expr))), name.clone()),
        src::Expression::Deref(ref expr) => dst::Expression::Deref(Box::new(try!(untype_expression(expr)))),
        src::Expression::MemberDeref(ref expr, ref name) => dst::Expression::MemberDeref(Box::new(try!(untype_expression(expr))), name.clone()),
        src::Expression::AddressOf(ref expr) => dst::Expression::AddressOf(Box::new(try!(untype_expression(expr)))),
        src::Expression::Call(ref func, ref params) => dst::Expression::Call(
            Box::new(try!(untype_expression(func))),
            try!(result_map(untype_expression, params))
        ),
        src::Expression::Cast(ref ty, ref expr) => dst::Expression::Cast(try!(untype_type(ty)), Box::new(try!(untype_expression(expr)))),
        src::Expression::Intrinsic(ref intrinsic) => dst::Expression::Intrinsic(try!(untype_intrinsic(intrinsic))),
    })
}

fn untype_vardef(vd: &src::VarDef) -> Result<dst::VarDef, UntyperError> {
    Ok(dst::VarDef {
        name: vd.name.clone(),
        typename: try!(untype_type(&vd.typename)),
        assignment: match vd.assignment { None => None, Some(ref expr) => Some(try!(untype_expression(expr))) },
    })
}

fn untype_init_expression(member: &src::Condition) -> Result<dst::Condition, UntyperError> {
    Ok(match *member {
        src::Condition::Expr(ref expr) => dst::Condition::Expr(try!(untype_expression(expr))),
        src::Condition::Assignment(ref vd) => dst::Condition::Assignment(try!(untype_vardef(vd))),
    })
}

fn untype_statement(statement: &src::Statement) -> Result<dst::Statement, UntyperError> {
    Ok(match *statement {
        src::Statement::Empty => dst::Statement::Empty,
        src::Statement::Expression(ref expr) => dst::Statement::Expression(try!(untype_expression(expr))),
        src::Statement::Var(ref vd) => dst::Statement::Var(try!(untype_vardef(vd))),
        src::Statement::Block(ref block) => dst::Statement::Block(try!(result_map(untype_statement, block))),
        src::Statement::If(ref cond, ref statement) => {
            dst::Statement::If(
                try!(untype_expression(cond)),
                Box::new(try!(untype_statement(statement)))
            )
        },
        src::Statement::For(ref init, ref cond, ref update, ref statement) => {
            dst::Statement::For(
                try!(untype_init_expression(init)),
                try!(untype_expression(cond)),
                try!(untype_expression(update)),
                Box::new(try!(untype_statement(statement)))
            )
        },
        src::Statement::While(ref cond, ref statement) => {
            dst::Statement::While(
                try!(untype_expression(cond)),
                Box::new(try!(untype_statement(statement)))
            )
        },
        src::Statement::Return(ref expr) => dst::Statement::Return(try!(untype_expression(expr))),
    })
}

fn untype_struct_member(member: &src::StructMember) -> Result<dst::StructMember, UntyperError> {
    Ok(dst::StructMember {
        name: member.name.clone(),
        typename: try!(untype_type(&member.typename)),
    })
}

fn untype_function_param(param: &src::FunctionParam) -> Result<dst::FunctionParam, UntyperError> {
    Ok(dst::FunctionParam {
        name: param.name.clone(),
        typename: try!(untype_type(&param.typename)),
    })
}

fn untype_kernel_param(param: &src::KernelParam) -> Result<dst::KernelParam, UntyperError> {
    Ok(dst::KernelParam {
        name: param.name.clone(),
        typename: try!(untype_type(&param.typename)),
    })
}

fn untype_root_definition(root: &src::RootDefinition) -> Result<dst::RootDefinition, UntyperError> {
    Ok(match *root {
        src::RootDefinition::GlobalVariable(ref gv) => {
            dst::RootDefinition::GlobalVariable(dst::GlobalVariable {
                name: gv.name.clone(),
                ty: try!(untype_type(&gv.ty)),
                address_space: gv.address_space.clone(),
                init: match gv.init { None => None, Some(ref expr) => Some(try!(untype_expression(expr))) },
            })
        }
        src::RootDefinition::Struct(ref sd) => {
            dst::RootDefinition::Struct(dst::StructDefinition {
                name: sd.name.clone(),
                members: try!(result_map(untype_struct_member, &sd.members)),
            })
        },
        src::RootDefinition::Function(ref fd) => {
            dst::RootDefinition::Function(dst::FunctionDefinition {
                name: fd.name.clone(),
                returntype: try!(untype_type(&fd.returntype)),
                params: try!(result_map(untype_function_param, &fd.params)),
                body: try!(result_map(untype_statement, &fd.body)),
            })
        }
        src::RootDefinition::Kernel(ref kernel) => {
            dst::RootDefinition::Kernel(dst::Kernel {
                params: try!(result_map(untype_kernel_param, &kernel.params)),
                body: try!(result_map(untype_statement, &kernel.body)),
                group_dimensions: kernel.group_dimensions.clone(),
            })
        }
    })
}

pub fn untype_module(module: &src::Module) -> Result<dst::Module, UntyperError> {
    Ok(dst::Module {
        root_definitions: try!(result_map(untype_root_definition, &module.root_definitions)),
        binds: module.binds.clone(),
    })
}