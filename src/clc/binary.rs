
use super::cir::*;

/// A type to represent the final OpenCL source
/// This is both the code (text) and slot binding metadata
#[derive(Debug)]
pub struct Binary {
    source: String
}

impl Binary {
    pub fn from_cir(cir: &Module) -> Binary {
        let printer = print(cir);
        Binary {
            source: printer.buffer.clone(),
        }
    }
}

impl ToString for Binary {
    fn to_string(&self) -> String {
        self.source.clone()
    }
}

struct Printer {
    indent: u32,
    buffer: String,
}

impl Printer {
    fn new() -> Printer {
        Printer {
            indent: 0,
            buffer: String::new(),
        }
    }

    fn print(&mut self, string: &str) {
        self.buffer.push_str(string);
    }

    fn space(&mut self) {
        self.buffer.push_str(" ");
    }

    fn separator(&mut self) {
        self.buffer.push_str("");
    }

    fn indent(&mut self) {
        self.indent = self.indent + 1;
    }

    fn unindent(&mut self) {
        assert!(self.indent > 0);
        self.indent = self.indent - 1;
    }

    fn line(&mut self) {
        self.buffer.push_str("\n");
        for _ in 0..self.indent {
            self.buffer.push_str("\t");
        }
    }

    pub fn to_string(&self) -> String {
        self.buffer.clone()
    }
}

fn print_dimension(dim: &VectorDimension, printer: &mut Printer) {
    printer.print(match dim {
        &VectorDimension::Two => "2",
        &VectorDimension::Three => "3",
        &VectorDimension::Four => "4",
        &VectorDimension::Eight => "8",
        &VectorDimension::Sixteen => "16",
    });
}

fn print_scalar(scalar: &Scalar, printer: &mut Printer) {
    printer.print(match scalar {
        &Scalar::Bool => "bool",
        &Scalar::Char => "char",
        &Scalar::UChar => "uchar",
        &Scalar::Short => "short",
        &Scalar::UShort => "ushort",
        &Scalar::Int => "int",
        &Scalar::UInt => "uint",
        &Scalar::Long => "long",
        &Scalar::ULong => "ulong",
        &Scalar::Half => "half",
        &Scalar::Float => "float",
        &Scalar::Double => "double",
    });
}

fn print_address_space(address_space: &AddressSpace, printer: &mut Printer) {
    printer.print(match address_space {
        &AddressSpace::Private => "__private",
        &AddressSpace::Local => "__local",
        &AddressSpace::Constant => "__constant",
        &AddressSpace::Global => "__global",
    });
}

fn print_typename(typename: &Type, printer: &mut Printer) {
    match typename {
        &Type::Void => printer.print("void"),
        &Type::Scalar(ref scalar) => print_scalar(scalar, printer),
        &Type::Vector(ref scalar, ref dim) => {
            print_scalar(scalar, printer);
            print_dimension(dim, printer);
        },
        &Type::Pointer(ref address_space, ref pointed_type) => {
            print_address_space(address_space, printer);
            printer.space();
            print_typename(pointed_type, printer);
            printer.print("*");
        },
        _ => unimplemented!(),
    };
}

fn print_binaryoperation(binop: &BinOp, lhs: &Box<Expression>, rhs: &Box<Expression>, last_precedence: u32, printer: &mut Printer) {

    let op_symbol = match binop {
        &BinOp::Add => "+",
        &BinOp::Subtract => "-",
        &BinOp::Multiply => "*",
        &BinOp::Divide => "/",
        &BinOp::Modulus => "%",
        &BinOp::Assignment => "=",
    };

    let op_prec = match binop {
        &BinOp::Add => 4,
        &BinOp::Subtract => 4,
        &BinOp::Multiply => 3,
        &BinOp::Divide => 3,
        &BinOp::Modulus => 3,
        &BinOp::Assignment => 14,
    };

    if last_precedence <= op_prec { printer.print("("); }
    print_expression_inner(lhs, op_prec, printer);
    printer.space();
    printer.print(op_symbol);
    printer.space();
    print_expression_inner(rhs, op_prec, printer);
    if last_precedence <= op_prec { printer.print(")"); }
}

fn print_literal(lit: &Literal, printer: &mut Printer) {
    printer.print(&(match lit {
        &Literal::Int(i) => format!("{}", i),
        &Literal::UInt(i) => format!("{}u", i),
        &Literal::Long(i) => format!("{}L", i),
        &Literal::Half(f) => format!("{}{}h", f, if f == f.floor() { ".0" }  else { "" }),
        &Literal::Float(f) => format!("{}{}f", f, if f == f.floor() { ".0" }  else { "" }),
        &Literal::Double(f) => format!("{}{}", f, if f == f.floor() { ".0" }  else { "" }),
    })[..]);
}

fn print_constructor(cons: &Constructor, printer: &mut Printer) {
    match cons {
        &Constructor::UInt3(ref x, ref y, ref z) => {
            printer.print("(");
            print_typename(&Type::Vector(Scalar::UInt, VectorDimension::Three), printer);
            printer.print(")");
            printer.print("(");
            print_expression_inner(x, 15, printer);
            printer.print(",");
            printer.space();
            print_expression_inner(&*y, 15, printer);
            printer.print(",");
            printer.space();
            print_expression_inner(&*z, 15, printer);
            printer.print(")");
        },
    }
}

fn print_intrinsic(intrinsic: &Intrinsic, printer: &mut Printer) {
    match intrinsic {
        &Intrinsic::GetGlobalId(ref expr) => {
            printer.print("get_global_id");
            printer.print("(");
            print_expression_inner(expr, 15, printer);
            printer.print(")");
        },
    }
}

fn print_expression_inner(expression: &Expression, last_precedence: u32, printer: &mut Printer) {
    match expression {
        &Expression::Literal(ref lit) => print_literal(lit, printer),
        &Expression::Constructor(ref cons) => print_constructor(cons, printer),
        &Expression::Variable(ref name) => printer.print(name),
        &Expression::BinaryOperation(ref binop, ref lhs, ref rhs) => print_binaryoperation(binop, lhs, rhs, last_precedence, printer),
        &Expression::ArraySubscript(ref array, ref sub) => {
            print_expression_inner(array, 1, printer);
            printer.print("[");
            print_expression(sub, printer);
            printer.print("]");
        },
        &Expression::Intrinsic(ref intrinsic) => print_intrinsic(intrinsic, printer),
        _ => unimplemented!(),
    }
}

fn print_expression(expression: &Expression, printer: &mut Printer) {
    // 16 is more than the largest precedence of any operation
    print_expression_inner(expression, 16, printer)
}

fn print_vardef(vardef: &VarDef, printer: &mut Printer) {
    print_typename(&vardef.typename, printer);
    printer.space();
    printer.print(&vardef.name[..]);
    match &vardef.assignment {
        &Some(ref expr) => {
            printer.space();
            printer.print("=");
            printer.space();
            print_expression(expr, printer);
        },
        &None => { },
    };
}

fn print_statement(statement: &Statement, printer: &mut Printer) {
    printer.line();
    match statement {
        &Statement::Expression(ref expr) => print_expression(expr, printer),
        &Statement::Var(ref vd) => print_vardef(vd, printer),
        _ => unimplemented!(),
    }
    printer.print(";");
}

fn print_statements(statements: &[Statement], printer: &mut Printer) {
    for statement in statements {
        print_statement(statement, printer);
    }
}

fn print_rootdefinition_struct(_: &StructDefinition, _: &mut Printer) {

}

fn print_rootdefinition_function(function: &FunctionDefinition, printer: &mut Printer) {
    print_typename(&function.returntype, printer);
    printer.space();
    printer.print(&function.name[..]);
    printer.separator();
    printer.print("(");
    let mut first = true;
    for param in &function.params {
        if !first {
            printer.print(",");
            printer.space();
        } else {
            first = false;
        }
        print_typename(&param.typename, printer);
        printer.space();
        printer.print(&param.name[..]);
    }
    printer.print(")");
    printer.line();
    printer.print("{");
    printer.indent();
    print_statements(&function.body[..], printer);
    printer.unindent();
    printer.line();
    printer.print("}");
    printer.line();
    printer.line();
}

fn print_rootdefinition_kernel(kernel: &Kernel, printer: &mut Printer) {
    printer.print("kernel");
    printer.space();
    printer.print("void");
    printer.space();
    printer.print("MyKernel");
    printer.separator();
    printer.print("(");
    let mut first = true;
    for param in &kernel.params {
        if !first {
            printer.print(",");
            printer.space();
        } else {
            first = false;
        }
        print_typename(&param.typename, printer);
        printer.space();
        printer.print(&param.name[..]);
    }
    printer.print(")");
    printer.line();
    printer.print("{");
    printer.indent();
    print_statements(&kernel.body[..], printer);
    printer.unindent();
    printer.line();
    printer.print("}");
    printer.line();
}

fn print_rootdefinition(rootdef: &RootDefinition, printer: &mut Printer) {
    match rootdef {
        &RootDefinition::Struct(ref sd) => print_rootdefinition_struct(sd, printer),
        &RootDefinition::Function(ref fd) => print_rootdefinition_function(fd, printer),
        &RootDefinition::Kernel(ref kernel) => print_rootdefinition_kernel(kernel, printer),
    }
}

fn print(module: &Module) -> Printer {
    let mut printer = Printer::new();
    printer.line();
    for rootdef in &module.root_definitions {
        print_rootdefinition(&rootdef, &mut printer);
    }
    printer
}