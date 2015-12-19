
use super::cst::*;

/// A type to represent the final OpenCL source
/// This is both the code (text) and slot binding metadata
#[derive(PartialEq, Debug, Clone)]
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

fn print_access_modifier(access_modifier: &AccessModifier, printer: &mut Printer) {
    match *access_modifier {
        AccessModifier::ReadOnly => printer.print("read_only "),
        AccessModifier::WriteOnly => printer.print("write_only "),
        AccessModifier::ReadWrite => printer.print("read_write "),
    };
}

fn print_typename(typename: &Type, printer: &mut Printer) {
    match typename {
        &Type::Void => printer.print("void"),
        &Type::Bool => printer.print("bool"),
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
        &Type::Struct(ref identifier) => {
            printer.print("struct");
            printer.space();
            printer.print(identifier);
        },
        &Type::Array(_, _) => panic!("Array types should not be printed directly"),
        &Type::Image2D(ref access_modifier) => {
            print_access_modifier(access_modifier, printer);
            printer.print("image2d_t");
        },
        _ => unimplemented!(),
    };
}

fn print_unaryoperation(unaryop: &UnaryOp, expr: &Box<Expression>, last_precedence: u32, printer: &mut Printer) {

    let (op_symbol, op_prec, after) = match *unaryop {
        UnaryOp::PrefixIncrement => ("++", 2, false),
        UnaryOp::PrefixDecrement => ("--", 2, false),
        UnaryOp::PostfixIncrement => ("++", 1, true),
        UnaryOp::PostfixDecrement => ("--", 1, true),
        UnaryOp::Plus => ("+", 2, false),
        UnaryOp::Minus => ("-", 2, false),
        UnaryOp::LogicalNot => ("!", 2, false),
        UnaryOp::BitwiseNot => ("~", 2, false),
    };

    if last_precedence <= op_prec { printer.print("("); }
    if !after {
        printer.print(op_symbol);
    }
    print_expression_inner(expr, op_prec, printer);
    if after {
        printer.print(op_symbol);
    }
    if last_precedence <= op_prec { printer.print(")"); }
}

fn print_binaryoperation(binop: &BinOp, lhs: &Box<Expression>, rhs: &Box<Expression>, last_precedence: u32, printer: &mut Printer) {

    let op_symbol = match *binop {
        BinOp::Add => "+",
        BinOp::Subtract => "-",
        BinOp::Multiply => "*",
        BinOp::Divide => "/",
        BinOp::Modulus => "%",
        BinOp::LeftShift => "<<",
        BinOp::RightShift => ">>",
        BinOp::BitwiseAnd => "&",
        BinOp::BitwiseOr => "|",
        BinOp::BitwiseXor => "^",
        BinOp::LogicalAnd => "&&",
        BinOp::LogicalOr => "||",
        BinOp::LessThan => "<",
        BinOp::LessEqual => "<=",
        BinOp::GreaterThan => ">",
        BinOp::GreaterEqual => ">=",
        BinOp::Equality => "==",
        BinOp::Inequality => "!=",
        BinOp::Assignment => "=",
    };

    let op_prec = match *binop {
        BinOp::Add => 4,
        BinOp::Subtract => 4,
        BinOp::Multiply => 3,
        BinOp::Divide => 3,
        BinOp::Modulus => 3,
        BinOp::LeftShift => 5,
        BinOp::RightShift => 5,
        BinOp::BitwiseAnd => 8,
        BinOp::BitwiseOr => 10,
        BinOp::BitwiseXor => 9,
        BinOp::LogicalAnd => 11,
        BinOp::LogicalOr => 12,
        BinOp::LessThan => 6,
        BinOp::LessEqual => 6,
        BinOp::GreaterThan => 6,
        BinOp::GreaterEqual => 6,
        BinOp::Equality => 7,
        BinOp::Inequality => 7,
        BinOp::Assignment => 14,
    };

    if last_precedence <= op_prec { printer.print("("); }
    print_expression_inner(lhs, op_prec + 1, printer);
    printer.space();
    printer.print(op_symbol);
    printer.space();
    print_expression_inner(rhs, op_prec, printer);
    if last_precedence <= op_prec { printer.print(")"); }
}

fn print_literal(lit: &Literal, printer: &mut Printer) {
    printer.print(&(match lit {
        &Literal::Bool(b) => if b { "true".to_string() } else { "false".to_string() },
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
        &Constructor::Float4(ref x, ref y, ref z, ref w) => {
            printer.print("(");
            print_typename(&Type::Vector(Scalar::Float, VectorDimension::Four), printer);
            printer.print(")");
            printer.print("(");
            print_expression_inner(x, 15, printer);
            printer.print(",");
            printer.space();
            print_expression_inner(&*y, 15, printer);
            printer.print(",");
            printer.space();
            print_expression_inner(&*z, 15, printer);
            printer.print(",");
            printer.space();
            print_expression_inner(&*w, 15, printer);
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
        &Expression::UnaryOperation(ref unaryop, ref expr) => print_unaryoperation(unaryop, expr, last_precedence, printer),
        &Expression::BinaryOperation(ref binop, ref lhs, ref rhs) => print_binaryoperation(binop, lhs, rhs, last_precedence, printer),
        &Expression::TernaryConditional(ref cond, ref lhs, ref rhs) => {
            let prec = 13;
            if last_precedence <= prec { printer.print("(") }
            print_expression_inner(cond, prec, printer);
            printer.space();
            printer.print("?");
            printer.space();
            print_expression_inner(lhs, prec, printer);
            printer.space();
            printer.print(":");
            printer.space();
            print_expression_inner(rhs, prec, printer);
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::Swizzle(ref vec, ref swizzle) => {
            let prec = 1;
            if last_precedence <= prec { printer.print("(") }
            print_expression_inner(vec, prec + 1, printer);
            printer.print(".");
            for swizzle_slot in swizzle {
                match *swizzle_slot {
                    SwizzleSlot::X => printer.print("x"),
                    SwizzleSlot::Y => printer.print("y"),
                    SwizzleSlot::Z => printer.print("z"),
                    SwizzleSlot::W => printer.print("w"),
                }
            };
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::ArraySubscript(ref array, ref sub) => {
            let prec = 1;
            if last_precedence <= prec { printer.print("(") }
            print_expression_inner(array, prec + 1, printer);
            printer.print("[");
            print_expression(sub, printer);
            printer.print("]");
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::Member(ref composite, ref member) => {
            let prec = 1;
            if last_precedence <= prec { printer.print("(") }
            print_expression_inner(composite, prec + 1, printer);
            printer.print(".");
            printer.print(member);
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::Deref(ref inner) => {
            let prec = 2;
            if last_precedence <= prec { printer.print("(") }
            printer.print("*");
            printer.separator();
            print_expression_inner(inner, prec + 1, printer);
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::MemberDeref(ref composite, ref member) => {
            let prec = 1;
            if last_precedence <= prec { printer.print("(") }
            print_expression_inner(composite, prec + 1, printer);
            printer.print("->");
            printer.print(member);
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::AddressOf(ref inner) => {
            let prec = 2;
            if last_precedence <= prec { printer.print("(") }
            printer.print("&");
            printer.separator();
            print_expression_inner(inner, prec, printer);
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::Call(ref func, ref params) => {
            print_expression_inner(func, 1, printer);
            printer.print("(");
            for (idx, param) in params.iter().enumerate() {
                print_expression(param, printer);
                if idx < params.len() - 1 {
                    printer.print(",");
                    printer.space();
                }
            }
            printer.print(")");
        },
        &Expression::Cast(ref ty, ref expr) => {
            let prec = 2;
            if last_precedence <= prec { printer.print("(") }
            printer.print("(");
            print_typename(ty, printer);
            printer.print(")");
            print_expression_inner(expr, prec + 1, printer);
            if last_precedence <= prec { printer.print(")") }
        },
        &Expression::Intrinsic(ref intrinsic) => print_intrinsic(intrinsic, printer),
    }
}

fn print_expression(expression: &Expression, printer: &mut Printer) {
    // 16 is more than the largest precedence of any operation
    print_expression_inner(expression, 16, printer)
}

fn print_vardef(vardef: &VarDef, printer: &mut Printer) {
    let array_dim = match vardef.typename {
        Type::Array(ref inner, ref dim) => {
            print_typename(inner, printer);
            Some(dim.clone())
        },
        ref ty => {
            print_typename(ty, printer);
            None
        },
    };
    printer.space();
    printer.print(&vardef.name);
    if let Some(dim) = array_dim {
        printer.print("[");
        print_literal(&Literal::Int(dim), printer);
        printer.print("]");
    };
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

#[allow(dead_code)]
fn print_condition(cond: &Condition, printer: &mut Printer) {
    match *cond {
        Condition::Expr(ref expr) => print_expression(expr, printer),
        Condition::Assignment(ref vd) => print_vardef(vd, printer),
    }
}

fn print_block(block: &[Statement], printer: &mut Printer) {
    printer.print("{");
    printer.indent();
    for statement in block {
        print_statement(statement, printer);
    }
    printer.unindent();
    printer.line();
    printer.print("}");
}

fn print_statement(statement: &Statement, printer: &mut Printer) {
    printer.line();
    match statement {
        &Statement::Empty => {
            printer.print(";");
        },
        &Statement::Expression(ref expr) => {
            print_expression(expr, printer);
            printer.print(";");
        },
        &Statement::Var(ref vd) => {
            print_vardef(vd, printer);
            printer.print(";");
        },
        &Statement::Block(ref statements) => print_block(&statements, printer),
        &Statement::If(ref cond, ref statement) => {
            printer.print("if");
            printer.space();
            printer.print("(");
            print_expression(cond, printer);
            printer.print(")");
            print_statement(statement, printer);
        },
        &Statement::For(ref init, ref cond, ref update, ref statement) => {
            printer.print("for");
            printer.space();
            printer.print("(");
            print_condition(init, printer);
            printer.print(";");
            printer.space();
            print_expression(cond, printer);
            printer.print(";");
            printer.space();
            print_expression(update, printer);
            printer.print(")");
            print_statement(statement, printer);
        },
        &Statement::While(ref cond, ref statement) => {
            printer.print("while");
            printer.space();
            printer.print("(");
            print_expression(cond, printer);
            printer.print(")");
            print_statement(statement, printer);
        },
        &Statement::Return(ref expr) => {
            printer.print("return");
            printer.space();
            print_expression(expr, printer);
            printer.print(";");
        },
    }
}

fn print_statements(statements: &[Statement], printer: &mut Printer) {
    for statement in statements {
        print_statement(statement, printer);
    }
}

fn print_rootdefinition_globalvariable(gv: &GlobalVariable, printer: &mut Printer) {
    print_typename(&gv.ty, printer);
    printer.space();
    print_address_space(&gv.address_space, printer);
    printer.space();
    printer.print(&gv.name);
    match &gv.init {
        &Some(ref expr) => {
            printer.space();
            printer.print("=");
            printer.space();
            print_expression(expr, printer);
        },
        &None => { },
    }
    printer.print(";");
}


fn print_rootdefinition_struct(structdefinition: &StructDefinition, printer: &mut Printer) {
    printer.print("struct");
    printer.space();
    printer.print(&structdefinition.name);
    printer.line();
    printer.print("{");
    printer.indent();
    for member in &structdefinition.members {
        printer.line();
        print_typename(&member.typename, printer);
        printer.space();
        printer.print(&member.name);
        printer.separator();
        printer.print(";");
    }
    printer.unindent();
    printer.line();
    printer.print("}");
    printer.separator();
    printer.print(";");
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
}

fn print_rootdefinition_kernel(kernel: &Kernel, printer: &mut Printer) {

    printer.print("__attribute__");
    printer.print("(");
    printer.print("(");
    printer.print("reqd_work_group_size");
    printer.print("(");
    printer.print(&kernel.group_dimensions.0.to_string());
    printer.print(",");
    printer.space();
    printer.print(&kernel.group_dimensions.1.to_string());
    printer.print(",");
    printer.space();
    printer.print(&kernel.group_dimensions.2.to_string());
    printer.print(")");
    printer.print(")");
    printer.print(")");
    printer.line();

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
}

fn print_rootdefinition(rootdef: &RootDefinition, printer: &mut Printer) {
    printer.line();
    match rootdef {
        &RootDefinition::GlobalVariable(ref gv) => print_rootdefinition_globalvariable(gv, printer),
        &RootDefinition::Struct(ref sd) => print_rootdefinition_struct(sd, printer),
        &RootDefinition::Function(ref fd) => print_rootdefinition_function(fd, printer),
        &RootDefinition::Kernel(ref kernel) => print_rootdefinition_kernel(kernel, printer),
    };
    printer.line();
}

fn print(module: &Module) -> Printer {
    let mut printer = Printer::new();
    for rootdef in &module.root_definitions {
        print_rootdefinition(&rootdef, &mut printer);
    }
    printer
}