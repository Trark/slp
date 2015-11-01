use hlsl::*;
use nom::{IResult,multispace,alphanumeric};
use std::str;
use std::str::FromStr;

named!(wsr<()>, chain!(multispace, || { () }));
named!(wso<()>, chain!(opt!(multispace), || { () }));

named!(digit<u32>, alt!(
    tag!("0") => { |_| { 0u32 } } |
    tag!("1") => { |_| { 1u32 } } |
    tag!("2") => { |_| { 2u32 } } |
    tag!("3") => { |_| { 3u32 } } |
    tag!("4") => { |_| { 4u32 } } |
    tag!("5") => { |_| { 5u32 } } |
    tag!("6") => { |_| { 6u32 } } |
    tag!("7") => { |_| { 7u32 } } |
    tag!("8") => { |_| { 8u32 } } |
    tag!("9") => { |_| { 9u32 } }
));

named!(literal_uint<u32>, map_opt!(
    many1!(digit),
    |digits| {
        let mut value = 0u32;
        for digit in digits {
            value = value * 10;
            value += digit;
        }
        Some(value)
    }
));

fn string_to_var(s: String) -> Result<String, ()> { if s.len() > 0 { Result::Ok(s) } else { Result::Err(()) } }
named!(variable_name<String>, map_res!(map_res!(map_res!(alphanumeric, str::from_utf8), FromStr::from_str), string_to_var));

fn string_to_type(s: String) -> Result<TypeName, ()> { if s.len() > 0 { Result::Ok(TypeName(s)) } else { Result::Err(()) } }
named!(type_name<TypeName>, map_res!(map_res!(map_res!(alphanumeric, str::from_utf8), FromStr::from_str), string_to_type));

named!(expr_variable<Expression>, chain!(acc: variable_name, || { return Expression::Variable(acc) }));

named!(expr_paren<Expression>, alt!(
        delimited!(tag!("("), expr, tag!(")")) |
        expr_variable
    )
);

#[derive(Clone)]
enum Precedence1Postfix {
    Increment,
    Decrement,
    Call(Vec<Expression>),
    ArraySubscript(Expression),
    Member(String),
}

named!(expr_p1_right<Precedence1Postfix>, chain!(
    wso ~
    right: alt!(
        tag!("++") => { |_| Precedence1Postfix::Increment } |
        tag!("--") => { |_| Precedence1Postfix::Decrement } |
        chain!(
            tag!("(") ~
            wso ~
            params: opt!(chain!(
                first: expr ~
                rest: many0!(chain!(tag!(",") ~ wso ~ next: expr, || { next })),
                || {
                    let mut v = Vec::new();
                    v.push(first);
                    for next in rest.iter() {
                        v.push(next.clone())
                    }
                    v
                }
            )) ~
            wso ~
            tag!(")"),
            || { Precedence1Postfix::Call(match params.clone() { Some(v) => v, None => Vec::new() }) }
        ) |
        chain!(
            tag!(".") ~
            wso ~
            member: variable_name,
            || { Precedence1Postfix::Member(member) }
        ) |
        chain!(
            tag!("[") ~
            wso ~
            subscript: expr ~
            wso ~
            tag!("]"),
            || { Precedence1Postfix::ArraySubscript(subscript) }
        )
    ),
    || { right }
));

named!(expr_p1<Expression>, chain!(
    left: expr_paren ~
    rights: many0!(expr_p1_right),
    || {
        let mut final_expression = left;
        for val in rights.iter() {
            match val.clone() {
                Precedence1Postfix::Increment => final_expression = Expression::UnaryOperation(UnaryOp::PostfixIncrement, Box::new(final_expression)),
                Precedence1Postfix::Decrement => final_expression = Expression::UnaryOperation(UnaryOp::PostfixDecrement, Box::new(final_expression)),
                Precedence1Postfix::Call(params) => final_expression = Expression::Call(Box::new(final_expression), params),
                Precedence1Postfix::ArraySubscript(expr) => final_expression = Expression::ArraySubscript(Box::new(final_expression), Box::new(expr)),
                Precedence1Postfix::Member(name) => final_expression = Expression::Member(Box::new(final_expression), name),
            }
        }
        final_expression
    }
));

named!(unaryop_prefix<UnaryOp>, alt!(
    chain!(tag!("+") ~ tag!("+"), || { UnaryOp::PrefixIncrement }) |
    chain!(tag!("-") ~ tag!("-"), || { UnaryOp::PrefixDecrement }) |
    tag!("+") => { |_| UnaryOp::Plus } |
    tag!("-") => { |_| UnaryOp::Minus } |
    tag!("!") => { |_| UnaryOp::LogicalNot } |
    tag!("~") => { |_| UnaryOp::BitwiseNot }
));

named!(expr_p2<Expression>, alt!(
    chain!(unary: unaryop_prefix ~ wso ~ expr: expr_p2, || { Expression::UnaryOperation(unary, Box::new(expr)) }) |
    chain!(tag!("(") ~ wso ~ cast: type_name ~ wso ~ tag!(")") ~ wso ~ expr: expr_p2, || { Expression::Cast(cast, Box::new(expr)) }) |
    expr_p1
));

fn combine_rights(left: Expression, rights: Vec<(BinOp, Expression)>) -> Expression {
    let mut final_expression = left;
    for val in rights.iter() {
        let (ref op, ref exp) = *val;
        final_expression = Expression::BinaryOperation(op.clone(), Box::new(final_expression), Box::new(exp.clone()))
    }
    final_expression
}

named!(binop_p3<BinOp>, alt!(
    tag!("*") => { |_| BinOp::Multiply } |
    tag!("/") => { |_| BinOp::Divide } |
    tag!("%") => { |_| BinOp::Modulus }
));

named!(expr_p3_right<(BinOp, Expression)>, chain!(
    wso ~
    op: binop_p3 ~
    wso ~
    right: expr_p2,
    || { return (op, right) }
));

named!(expr_p3<Expression>, chain!(
    left: expr_p2 ~
    rights: many0!(expr_p3_right),
    || { combine_rights(left, rights) }
));

named!(binop_p4<BinOp>, alt!(
    tag!("+") => { |_| BinOp::Add } |
    tag!("-") => { |_| BinOp::Subtract }
));

named!(expr_p4_right<(BinOp, Expression)>, chain!(
    wso ~
    op: binop_p4 ~
    wso ~
    right: expr_p3,
    || { return (op, right) }
));

named!(expr_p4<Expression>, chain!(
    left: expr_p3 ~
    rights: many0!(expr_p4_right),
    || { combine_rights(left, rights) }
));

named!(expr<Expression>, chain!(wso ~ top: expr_p4 ~ wso, || { top }));

named!(vardef<VarDef>, chain!(
    wso ~
    typename: type_name ~
    wsr ~
    varname: variable_name ~
    assign: opt!(
        chain!(
            wso ~
            tag!("=") ~
            assignment_expr: expr,
            || { assignment_expr }
        )
    ) ~
    wso,
    || { VarDef::new(varname, typename, assign) }
));

named!(condition<Condition>, chain!(
    wso ~
    c: alt!(
        vardef => { |variable_definition| Condition::Assignment(variable_definition) } |
        expr => { |expression| Condition::Expr(expression) }
    ) ~
    wso,
    || { c }
));

named!(statement<Statement>, chain!(
    wso ~
    st: alt!(
        tag!(";") => { |_| { Statement::Empty } } |
        chain!(tag!("if") ~ wso ~ tag!("(") ~ cond: condition ~ tag!(")") ~ inner_statement: statement, || { Statement::If(cond, Box::new(inner_statement)) }) |
        chain!(tag!("{") ~ statements: many0!(statement) ~ tag!("}"), || { Statement::Block(statements) }) |
        chain!(
            tag!("for") ~
            wso ~
            tag!("(") ~
            init: condition ~
            tag!(";") ~
            cond: condition ~
            tag!(";") ~
            inc: condition ~
            tag!(")") ~
            inner: statement,
            || { Statement::For(init, cond, inc, Box::new(inner)) }
        ) |
        chain!(tag!("while") ~ wso ~ tag!("(") ~ cond: condition ~ tag!(")") ~ inner: statement, || { Statement::While(cond, Box::new(inner)) }) |
        chain!(var: vardef ~ tag!(";"), || { Statement::Var(var) }) |
        chain!(expression_statement: expr ~ tag!(";"), || { Statement::Expression(expression_statement) })
    ) ~
    wso,
    || { st }
));

named!(structmember<StructMember>, chain!(
    wso ~
    typename: type_name ~
    wsr ~
    varname: variable_name ~
    wso ~
    tag!(";"),
    || { StructMember { name: varname, typename: typename } }
));

named!(structdefinition<StructDefinition>, chain!(
    tag!("struct") ~
    wso ~
    structname: type_name ~
    wso ~
    tag!("{") ~
    members: many0!(chain!(
        wso ~
        member: structmember,
        || { member }
    )) ~
    wso ~
    tag!("}") ~
    wso ~
    tag!(";"),
    || { StructDefinition { name: structname, members: members } }
));

named!(constantvariable<ConstantVariable>, chain!(
    wso ~
    typename: type_name ~
    wsr ~
    varname: variable_name ~
    wso ~
    tag!(";"),
    || { ConstantVariable { name: varname, typename: typename, offset: None } }
));

named!(cbuffer_register<ConstantSlot>, chain!(
    wso ~
    tag!(":") ~
    wso ~
    tag!("register") ~
    slot_index: delimited!(
        chain!(wso ~ tag!("(") ~ wso ~ tag!("b") ~ wso, || { }),
        literal_uint,
        chain!(wso ~ tag!(")") ~ wso, || { })
    ),
    || { ConstantSlot(slot_index) }
));

named!(cbuffer<ConstantBuffer>, chain!(
    tag!("cbuffer") ~
    wsr ~
    name: variable_name ~
    slot: opt!(cbuffer_register) ~
    members: delimited!(
        chain!(wso ~ tag!("{") ~ wso, || { }),
        many0!(constantvariable),
        chain!(wso ~ tag!("}") ~ wso, || { })
    ) ~
    tag!(";"),
    || { ConstantBuffer { name: name, slot: slot, members: members } }
));

named!(functionparam<FunctionParam>, chain!(
    wso ~
    typename: type_name ~
    wsr ~
    param: variable_name ~
    wso,
    || { FunctionParam { name: param, typename: typename } }
));

named!(functiondefinition<FunctionDefinition>, chain!(
    ret: type_name ~ wsr ~
    func_name: variable_name ~
    params: delimited!(
        chain!(wso ~ tag!("(") ~ wso, || { }),
        separated_list!(tag!(","), functionparam),
        chain!(wso ~ tag!(")") ~ wso, || { })
    ) ~
    body: delimited!(
        chain!(wso ~ tag!("{") ~ wso, || { }),
        many0!(statement),
        chain!(wso ~ tag!("}") ~ wso, || { })
    ),
    || { FunctionDefinition { name: func_name, returntype: ret, params: params, body: body } }
));

named!(rootdefinition<RootDefinition>, chain!(
    wso ~
    def: alt!(
        structdefinition => { |structdef| { RootDefinition::Struct(structdef) } } |
        functiondefinition => { |funcdef| { RootDefinition::Function(funcdef) } } |
        cbuffer => { |cbuffer| { RootDefinition::ConstantBuffer(cbuffer) } }
    ) ~
    wso,
    || { def }
));

type RootDefinitionArr = Vec<RootDefinition>;
named!(module<&[u8], RootDefinitionArr>, many0!(rootdefinition));

pub fn parse(entry_point: String, source: &[u8]) -> Option<Module> {
    let parse_result = module(source);
    match parse_result {
        IResult::Done(rest, hlsl) => if rest.len() == 0 { Some(Module { entry_point: entry_point, root_definitions: hlsl }) } else { None },
        IResult::Error(_) => None,
        IResult::Incomplete(_) => None,
    }
}

#[cfg(test)]
fn name(var_name: &'static str) -> String { String::from_str(var_name).unwrap() }
#[cfg(test)]
fn var(var_name: &'static str) -> Expression { Expression::Variable(name(var_name)) }
#[cfg(test)]
fn varb(var_name: &'static str) -> Box<Expression> { Box::new(var(var_name)) }

#[test]
fn test_expression() {
    assert_eq!(binop_p3(&b"*"[..]), IResult::Done(&b""[..], BinOp::Multiply));
    assert_eq!(binop_p3(&b"/"[..]), IResult::Done(&b""[..], BinOp::Divide));
    assert_eq!(binop_p3(&b"%"[..]), IResult::Done(&b""[..], BinOp::Modulus));

    assert!(binop_p3(&b"+"[..]).is_err());
    assert!(!binop_p3(&b""[..]).is_done());

    assert_eq!(binop_p4(&b"+"[..]), IResult::Done(&b""[..], BinOp::Add));
    assert_eq!(binop_p4(&b"-"[..]), IResult::Done(&b""[..], BinOp::Subtract));

    assert_eq!(expr_variable(&b"a"[..]), IResult::Done(&b""[..], var("a")));
    assert_eq!(expr_paren(&b"a"[..]), IResult::Done(&b""[..], var("a")));
    assert_eq!(expr_p1(&b"a"[..]), IResult::Done(&b""[..], var("a")));
    assert_eq!(expr_p2(&b"a"[..]), IResult::Done(&b""[..], var("a")));
    assert_eq!(expr_p3(&b"a"[..]), IResult::Done(&b""[..], var("a")));
    assert_eq!(expr_p4(&b"a"[..]), IResult::Done(&b""[..], var("a")));

    assert_eq!(expr(&b"a"[..]), IResult::Done(&b""[..], var("a")));

    assert_eq!(expr(&b"a+b"[..]), IResult::Done(&b""[..], Expression::BinaryOperation(BinOp::Add, varb("a"), varb("b"))));
    assert_eq!(expr(&b"a*b"[..]), IResult::Done(&b""[..], Expression::BinaryOperation(BinOp::Multiply, varb("a"), varb("b"))));

    assert_eq!(expr(&b"a-b+c"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Add,
            Box::new(Expression::BinaryOperation(BinOp::Subtract, varb("a"), varb("b"))),
            varb("c")
        )
    ));
    assert_eq!(expr(&b"a-b*c"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Subtract,
            varb("a"),
            Box::new(Expression::BinaryOperation(BinOp::Multiply, varb("b"), varb("c")))
        )
    ));
    assert_eq!(expr_p4(&b"a*b-c"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Subtract,
            Box::new(Expression::BinaryOperation(BinOp::Multiply, varb("a"), varb("b"))),
            varb("c"),
        )
    ));
    assert_eq!(expr_p4(&b"a*(b-c)"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Multiply,
            varb("a"),
            Box::new(Expression::BinaryOperation(BinOp::Subtract, varb("b"), varb("c")))
        )
    ));

    assert_eq!(expr(&b"a*b/c"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Divide,
            Box::new(Expression::BinaryOperation(BinOp::Multiply, varb("a"), varb("b"))),
            varb("c")
        )
    ));
    assert_eq!(expr(&b"(a*b)/c"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Divide,
            Box::new(Expression::BinaryOperation(BinOp::Multiply, varb("a"), varb("b"))),
            varb("c")
        )
    ));
    assert_eq!(expr(&b"a*(b/c)"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Multiply,
            varb("a"),
            Box::new(Expression::BinaryOperation(BinOp::Divide, varb("b"), varb("c")))
        )
    ));

    assert_eq!(expr(&b"a++"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::PostfixIncrement, varb("a"))));
    assert_eq!(expr(&b"a--"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::PostfixDecrement, varb("a"))));
    assert_eq!(expr(&b"++a"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::PrefixIncrement, varb("a"))));
    assert_eq!(expr(&b"--a"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::PrefixDecrement, varb("a"))));
    assert_eq!(expr(&b"+a"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::Plus, varb("a"))));
    assert_eq!(expr(&b"-a"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::Minus, varb("a"))));
    assert_eq!(expr(&b"!a"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::LogicalNot, varb("a"))));
    assert_eq!(expr(&b"~a"[..]), IResult::Done(&b""[..], Expression::UnaryOperation(UnaryOp::BitwiseNot, varb("a"))));

    assert_eq!(expr(&b"a[b]"[..]), IResult::Done(&b""[..],
        Expression::ArraySubscript(varb("a"), varb("b"))
    ));
    assert_eq!(expr(&b"d+a[b+c]"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Add,
            varb("d"),
            Box::new(Expression::ArraySubscript(varb("a"),
                Box::new(Expression::BinaryOperation(BinOp::Add,
                    varb("b"), varb("c")
                ))
            ))
        )
    ));
    assert_eq!(expr(&b" d + a\t[ b\n+ c ]"[..]), IResult::Done(&b""[..],
        Expression::BinaryOperation(BinOp::Add,
            varb("d"),
            Box::new(Expression::ArraySubscript(varb("a"),
                Box::new(Expression::BinaryOperation(BinOp::Add,
                    varb("b"), varb("c")
                ))
            ))
        )
    ));
    assert_eq!(expr(&b"array.Load"[..]), IResult::Done(&b""[..],
        Expression::Member(varb("array"), name("Load"))
    ));
    assert_eq!(expr(&b"array.Load()"[..]), IResult::Done(&b""[..],
        Expression::Call(Box::new(Expression::Member(varb("array"), name("Load"))), vec![])
    ));
    assert_eq!(expr(&b" array . Load ( ) "[..]), IResult::Done(&b""[..],
        Expression::Call(Box::new(Expression::Member(varb("array"), name("Load"))), vec![])
    ));
    assert_eq!(expr(&b"array.Load(a)"[..]), IResult::Done(&b""[..],
        Expression::Call(Box::new(Expression::Member(varb("array"), name("Load"))), vec![var("a")])
    ));
    assert_eq!(expr(&b"array.Load(a,b)"[..]), IResult::Done(&b""[..],
        Expression::Call(Box::new(Expression::Member(varb("array"), name("Load"))), vec![var("a"), var("b")])
    ));
    assert_eq!(expr(&b"array.Load(a, b)"[..]), IResult::Done(&b""[..],
        Expression::Call(Box::new(Expression::Member(varb("array"), name("Load"))), vec![var("a"), var("b")])
    ));

    assert_eq!(expr(&b"(float) b"[..]), IResult::Done(&b""[..],
        Expression::Cast(TypeName(name("float")), varb("b"))
    ));
}

#[test]
fn test_statement() {

    // Empty statement
    assert_eq!(statement(&b";"[..]), IResult::Done(&b""[..], Statement::Empty));

    // Expression statements
    assert_eq!(statement(&b"func();"[..]), IResult::Done(&b""[..],
        Statement::Expression(Expression::Call(varb("func"), vec![]))
    ));
    assert_eq!(statement(&b"func();"[..]), statement(&b" func ( ) ; "[..]));

    // Condition expressions
    assert_eq!(condition(&b"x"[..]), IResult::Done(&b""[..],
        Condition::Expr(var("x"))
    ));
    assert_eq!(vardef(&b"uint x"[..]), IResult::Done(&b""[..],
        VarDef::new(name("x"), TypeName(name("uint")), None)
    ));
    assert_eq!(condition(&b"uint x"[..]), IResult::Done(&b""[..],
        Condition::Assignment(VarDef::new(name("x"), TypeName(name("uint")), None))
    ));
    assert_eq!(condition(&b"uint x = y"[..]), IResult::Done(&b""[..],
        Condition::Assignment(VarDef::new(name("x"), TypeName(name("uint")), Some(var("y"))))
    ));

    // Variable declarations
    assert_eq!(statement(&b"uint x = y;"[..]), IResult::Done(&b""[..],
        Statement::Var(VarDef::new(name("x"), TypeName(name("uint")), Some(var("y"))))
    ));

    // Blocks
    assert_eq!(statement(&b"{one();two();}"[..]), IResult::Done(&b""[..],
        Statement::Block(vec![
            Statement::Expression(Expression::Call(varb("one"), vec![])),
            Statement::Expression(Expression::Call(varb("two"), vec![]))
        ])
    ));
    assert_eq!(statement(&b"{one();two();}"[..]), statement(&b" { one(); two(); } "[..]));

    // If statement
    assert_eq!(statement(&b"if(a)func();"[..]), IResult::Done(&b""[..],
        Statement::If(Condition::Expr(var("a")), Box::new(Statement::Expression(Expression::Call(varb("func"), vec![]))))
    ));
    assert_eq!(statement(&b"if(a)func();"[..]), statement(&b"if (a) func(); "[..]));
    assert_eq!(statement(&b"if (a)\n{\n\tone();\n\ttwo();\n}"[..]), IResult::Done(&b""[..],
        Statement::If(Condition::Expr(var("a")), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(varb("one"), vec![])),
            Statement::Expression(Expression::Call(varb("two"), vec![]))
        ])))
    ));
    assert_eq!(statement(&b"if (uint x = y)\n{\n\tone();\n\ttwo();\n}"[..]), IResult::Done(&b""[..],
        Statement::If(Condition::Assignment(VarDef::new(name("x"), TypeName(name("uint")), Some(var("y")))), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(varb("one"), vec![])),
            Statement::Expression(Expression::Call(varb("two"), vec![]))
        ])))
    ));

    // While loops
    assert_eq!(statement(&b"while (a)\n{\n\tone();\n\ttwo();\n}"[..]), IResult::Done(&b""[..],
        Statement::While(Condition::Expr(var("a")), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(varb("one"), vec![])),
            Statement::Expression(Expression::Call(varb("two"), vec![]))
        ])))
    ));
    assert_eq!(statement(&b"while (int x = y)\n{\n\tone();\n\ttwo();\n}"[..]), IResult::Done(&b""[..],
        Statement::While(Condition::Assignment(VarDef::new(name("x"), TypeName(name("int")), Some(var("y")))), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(varb("one"), vec![])),
            Statement::Expression(Expression::Call(varb("two"), vec![]))
        ])))
    ));

    // For loops
    assert_eq!(statement(&b"for(a;b;c)func();"[..]), IResult::Done(&b""[..],
        Statement::For(Condition::Expr(var("a")), Condition::Expr(var("b")), Condition::Expr(var("c")), Box::new(
            Statement::Expression(Expression::Call(varb("func"), vec![]))
        ))
    ));
    assert!(statement(&b"for (uint i = 0; i; i++) { func(); }"[..]).is_done());
}

#[test]
fn test_rootdefinition() {

    let test_struct_str = &b"struct MyStruct { uint a; float b; };"[..];
    let test_struct_ast = StructDefinition {
        name: TypeName(name("MyStruct")),
        members: vec![
            StructMember { name: name("a"), typename: TypeName(name("uint")) },
            StructMember { name: name("b"), typename: TypeName(name("float")) },
        ]
    };
    assert_eq!(structdefinition(test_struct_str), IResult::Done(&b""[..], test_struct_ast.clone()));
    assert_eq!(rootdefinition(test_struct_str), IResult::Done(&b""[..], RootDefinition::Struct(test_struct_ast.clone())));

    let test_func_str = &b"void func(float x) { }"[..];
    let test_func_ast = FunctionDefinition { 
        name: name("func"),
        returntype: TypeName(name("void")),
        params: vec![FunctionParam { name: name("x"), typename: TypeName(name("float")) }],
        body: vec![],
    };
    assert_eq!(functiondefinition(test_func_str), IResult::Done(&b""[..], test_func_ast.clone()));
    assert_eq!(rootdefinition(test_func_str), IResult::Done(&b""[..], RootDefinition::Function(test_func_ast.clone())));

    assert_eq!(literal_uint(&b"1"[..]), IResult::Done(&b""[..], 1));
    assert_eq!(literal_uint(&b"12"[..]), IResult::Done(&b""[..], 12));

    let test_cbuffervar_str = &b"float4x4 wvp;"[..];
    let test_cbuffervar_ast = ConstantVariable { name: name("wvp"), typename: TypeName(name("float4x4")), offset: None };
    assert_eq!(constantvariable(test_cbuffervar_str), IResult::Done(&b""[..], test_cbuffervar_ast.clone()));

    let test_cbuffer1_str = &b"cbuffer globals { float4x4 wvp; };"[..];
    let test_cbuffer1_ast = ConstantBuffer {
        name: name("globals"),
        slot: None,
        members: vec![
            ConstantVariable { name: name("wvp"), typename: TypeName(name("float4x4")), offset: None },
        ]
    };
    assert_eq!(cbuffer(test_cbuffer1_str), IResult::Done(&b""[..], test_cbuffer1_ast.clone()));
    assert_eq!(rootdefinition(test_cbuffer1_str), IResult::Done(&b""[..], RootDefinition::ConstantBuffer(test_cbuffer1_ast.clone())));

    assert_eq!(cbuffer_register(&b" : register(b12) "[..]), IResult::Done(&b""[..], ConstantSlot(12)));

    let test_cbuffer2_str = &b"cbuffer globals : register(b12) { float4x4 wvp; };"[..];
    let test_cbuffer2_ast = ConstantBuffer {
        name: name("globals"),
        slot: Some(ConstantSlot(12)),
        members: vec![
            ConstantVariable { name: name("wvp"), typename: TypeName(name("float4x4")), offset: None },
        ]
    };
    assert_eq!(cbuffer(test_cbuffer2_str), IResult::Done(&b""[..], test_cbuffer2_ast.clone()));
    assert_eq!(rootdefinition(test_cbuffer2_str), IResult::Done(&b""[..], RootDefinition::ConstantBuffer(test_cbuffer2_ast.clone())));
}
