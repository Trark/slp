use super::tokens::*;
use super::ast::*;
use nom::{IResult,Needed,Err,ErrorKind};

#[derive(PartialEq, Debug, Clone)]
pub enum ParseError {
    WrongToken,
    ExpectingIdentifier,
    WrongSlotType,
}

macro_rules! token (
    ($i:expr, $inp: pat) => (
        {
            let res: IResult<&[Token], Token, ParseError> = if $i.len() == 0 {
                IResult::Incomplete(Needed::Size(1))
            } else {
                match $i[0] {
                    $inp => IResult::Done(&$i[1..], $i[0].clone()),
                    _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseError::WrongToken), $i))
                } 
            };
            res
        }
    );
);

fn parse_variablename(input: &[Token]) -> IResult<&[Token], String, ParseError> {
    map!(input, token!(Token::Id(_)), |tok| { match tok { Token::Id(Identifier(name)) => name.clone(), _ => unreachable!() } })
}

fn parse_typename(input: &[Token]) -> IResult<&[Token], TypeName, ParseError> {
    chain!(input,
        root: token!(Token::Id(_)) ~
        // Todo: Implement proper type arguments
        opt!(delimited!(token!(Token::LeftAngleBracket(_)), parse_typename, token!(Token::RightAngleBracket(_)))),
        || {
            match &root {
                &Token::Id(Identifier(ref name)) => TypeName(name.clone()),
                _ => unreachable!()
            }
        }
    )
}

fn expr_paren(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {
    alt!(input,
        delimited!(token!(Token::LeftParen), expr, token!(Token::RightParen)) |
        parse_variablename => { |name| { Expression::Variable(name) } } |
        token!(Token::LiteralUint(_)) => { |tok: Token| { match tok { Token::LiteralUint(value) => Expression::LiteralUint(value), _ => unreachable!() } } } |
        token!(Token::LiteralInt(_)) => { |tok: Token| { match tok { Token::LiteralInt(value) => Expression::LiteralInt(value), _ => unreachable!() } } } |
        token!(Token::LiteralLong(_)) => { |tok: Token| { match tok { Token::LiteralLong(value) => Expression::LiteralLong(value), _ => unreachable!() } } } |
        token!(Token::LiteralFloat(_)) => { |tok: Token| { match tok { Token::LiteralFloat(value) => Expression::LiteralFloat(value), _ => unreachable!() } } } |
        token!(Token::LiteralDouble(_)) => { |tok: Token| { match tok { Token::LiteralDouble(value) => Expression::LiteralDouble(value), _ => unreachable!() } } }
    )
}

fn expr_p1(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {

    #[derive(Clone)]
    enum Precedence1Postfix {
        Increment,
        Decrement,
        Call(Vec<Expression>),
        ArraySubscript(Expression),
        Member(String),
    }

    fn expr_p1_right(input: &[Token]) -> IResult<&[Token], Precedence1Postfix, ParseError> {
        chain!(input,
            right: alt!(
                chain!(token!(Token::Plus) ~ token!(Token::Plus), || { Precedence1Postfix::Increment }) |
                chain!(token!(Token::Minus) ~ token!(Token::Minus), || { Precedence1Postfix::Decrement }) |
                chain!(
                    token!(Token::LeftParen) ~
                    params: opt!(chain!(
                        first: expr ~
                        rest: many0!(chain!(token!(Token::Comma) ~ next: expr, || { next })),
                        || {
                            let mut v = Vec::new();
                            v.push(first);
                            for next in rest.iter() {
                                v.push(next.clone())
                            }
                            v
                        }
                    )) ~
                    token!(Token::RightParen),
                    || { Precedence1Postfix::Call(match params.clone() { Some(v) => v, None => Vec::new() }) }
                ) |
                chain!(
                    token!(Token::Period) ~
                    member: parse_variablename,
                    || { Precedence1Postfix::Member(member) }
                ) |
                chain!(
                    token!(Token::LeftSquareBracket) ~
                    subscript: expr ~
                    token!(Token::RightSquareBracket),
                    || { Precedence1Postfix::ArraySubscript(subscript) }
                )
            ),
            || { right }
        )
    }

    chain!(input,
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
    )
}

fn unaryop_prefix(input: &[Token]) -> IResult<&[Token], UnaryOp, ParseError> {
    alt!(input,
        chain!(token!(Token::Plus) ~ token!(Token::Plus), || { UnaryOp::PrefixIncrement }) |
        chain!(token!(Token::Minus) ~ token!(Token::Minus), || { UnaryOp::PrefixDecrement }) |
        token!(Token::Plus) => { |_| UnaryOp::Plus } |
        token!(Token::Minus) => { |_| UnaryOp::Minus } |
        token!(Token::ExclamationPoint) => { |_| UnaryOp::LogicalNot } |
        token!(Token::Tilde) => { |_| UnaryOp::BitwiseNot }
    )
}

fn expr_p2(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {
    alt!(input,
        chain!(unary: unaryop_prefix ~ expr: expr_p2, || { Expression::UnaryOperation(unary, Box::new(expr)) }) |
        chain!(token!(Token::LeftParen) ~ cast: parse_typename ~ token!(Token::RightParen) ~ expr: expr_p2, || { Expression::Cast(cast, Box::new(expr)) }) |
        expr_p1
    )
}

fn combine_rights(left: Expression, rights: Vec<(BinOp, Expression)>) -> Expression {
    let mut final_expression = left;
    for val in rights.iter() {
        let (ref op, ref exp) = *val;
        final_expression = Expression::BinaryOperation(op.clone(), Box::new(final_expression), Box::new(exp.clone()))
    }
    final_expression
}

fn expr_p3(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {

    fn binop_p3(input: &[Token]) -> IResult<&[Token], BinOp, ParseError> {
        alt!(input,
            token!(Token::Asterix) => { |_| BinOp::Multiply } |
            token!(Token::ForwardSlash) => { |_| BinOp::Divide } |
            token!(Token::Percent) => { |_| BinOp::Modulus }
        )
    }

    fn expr_p3_right(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseError> {
        chain!(input,
            op: binop_p3 ~
            right: expr_p2,
            || { return (op, right) }
        )
    }

    chain!(input,
        left: expr_p2 ~
        rights: many0!(expr_p3_right),
        || { combine_rights(left, rights) }
    )
}

fn expr_p4(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {

    fn binop_p4(input: &[Token]) -> IResult<&[Token], BinOp, ParseError> {
        alt!(input,
            token!(Token::Plus) => { |_| BinOp::Add } |
            token!(Token::Minus) => { |_| BinOp::Subtract }
        )
    }

    fn expr_p4_right(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseError> {
        chain!(input,
            op: binop_p4 ~
            right: expr_p3,
            || { return (op, right) }
        )
    }

    chain!(input,
        left: expr_p3 ~
        rights: many0!(expr_p4_right),
        || { combine_rights(left, rights) }
    )
}

fn expr_p15(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {
    alt!(input,
        chain!(lhs: expr_p4 ~ token!(Token::Equals) ~ rhs: expr_p15, || { Expression::BinaryOperation(BinOp::Assignment, Box::new(lhs), Box::new(rhs)) }) |
        expr_p4
    )
}

fn expr(input: &[Token]) -> IResult<&[Token], Expression, ParseError> {
    expr_p15(input)
}

fn vardef(input: &[Token]) -> IResult<&[Token], VarDef, ParseError> {
    chain!(input,
        typename: parse_typename ~
        varname: parse_variablename ~
        assign: opt!(
            chain!(
                token!(Token::Equals) ~
                assignment_expr: expr,
                || { assignment_expr }
            )
        ),
        || { VarDef::new(varname, typename, assign) }
    )
}

fn condition(input: &[Token]) -> IResult<&[Token], Condition, ParseError> {
    alt!(input,
        vardef => { |variable_definition| Condition::Assignment(variable_definition) } |
        expr => { |expression| Condition::Expr(expression) }
    )
}

fn statement(input: &[Token]) -> IResult<&[Token], Statement, ParseError> {
    alt!(input,
        token!(Token::Semicolon) => { |_| { Statement::Empty } } |
        chain!(token!(Token::If) ~ token!(Token::LeftParen) ~ cond: condition ~ token!(Token::RightParen) ~ inner_statement: statement, || { Statement::If(cond, Box::new(inner_statement)) }) |
        chain!(token!(Token::LeftBrace) ~ statements: many0!(statement) ~ token!(Token::RightBrace), || { Statement::Block(statements) }) |
        chain!(
            token!(Token::For) ~
            token!(Token::LeftParen) ~
            init: condition ~
            token!(Token::Semicolon) ~
            cond: condition ~
            token!(Token::Semicolon) ~
            inc: condition ~
            token!(Token::RightParen) ~
            inner: statement,
            || { Statement::For(init, cond, inc, Box::new(inner)) }
        ) |
        chain!(token!(Token::While) ~ token!(Token::LeftParen) ~ cond: condition ~ token!(Token::RightParen) ~ inner: statement, || { Statement::While(cond, Box::new(inner)) }) |
        chain!(var: vardef ~ token!(Token::Semicolon), || { Statement::Var(var) }) |
        chain!(expression_statement: expr ~ token!(Token::Semicolon), || { Statement::Expression(expression_statement) })
    )
}

fn structmember(input: &[Token]) -> IResult<&[Token], StructMember, ParseError> {
    chain!(input,
        typename: parse_typename ~
        varname: parse_variablename ~
        token!(Token::Semicolon),
        || { StructMember { name: varname, typename: typename } }
    )
}

fn structdefinition(input: &[Token]) -> IResult<&[Token], StructDefinition, ParseError> {
    chain!(input,
        token!(Token::Struct) ~
        structname: parse_typename ~
        token!(Token::LeftBrace) ~
        members: many0!(chain!(
            member: structmember,
            || { member }
        )) ~
        token!(Token::RightBrace) ~
        token!(Token::Semicolon),
        || { StructDefinition { name: structname, members: members } }
    )
}

fn constantvariable(input: &[Token]) -> IResult<&[Token], ConstantVariable, ParseError> {
    chain!(input,
        typename: parse_typename ~
        varname: parse_variablename ~
        token!(Token::Semicolon),
        || { ConstantVariable { name: varname, typename: typename, offset: None } }
    )
}

fn cbuffer_register(input: &[Token]) -> IResult<&[Token], ConstantSlot, ParseError> {
    map_res!(input, preceded!(token!(Token::Colon), token!(Token::Register(_))),
        |reg| { match reg {
            Token::Register(RegisterSlot::B(slot)) => Ok(ConstantSlot(slot)) as Result<ConstantSlot, ParseError>,
            Token::Register(_) => Err(ParseError::WrongSlotType),
            _ => unreachable!(),
        } }
    )
}

fn cbuffer(input: &[Token]) -> IResult<&[Token], ConstantBuffer, ParseError> {
    chain!(input,
        token!(Token::ConstantBuffer) ~
        name: parse_variablename ~
        slot: opt!(cbuffer_register) ~
        members: delimited!(
            token!(Token::LeftBrace),
            many0!(constantvariable),
            token!(Token::RightBrace)
        ) ~
        token!(Token::Semicolon),
        || { ConstantBuffer { name: name, slot: slot, members: members } }
    )
}

fn globalvariable_register(input: &[Token]) -> IResult<&[Token], GlobalSlot, ParseError> {
    map_res!(input, preceded!(token!(Token::Colon), token!(Token::Register(_))),
        |reg| { match reg {
            Token::Register(RegisterSlot::T(slot)) => Ok(GlobalSlot::ReadSlot(slot)) as Result<GlobalSlot, ParseError>,
            Token::Register(RegisterSlot::U(slot)) => Ok(GlobalSlot::ReadWriteSlot(slot)) as Result<GlobalSlot, ParseError>,
            Token::Register(_) => Err(ParseError::WrongSlotType),
            _ => unreachable!(),
        } }
    )
}

fn globalvariable(input: &[Token]) -> IResult<&[Token], GlobalVariable, ParseError> {
    chain!(input,
        typename: parse_typename ~
        name: parse_variablename ~
        slot: opt!(globalvariable_register) ~
        token!(Token::Semicolon),
        || { GlobalVariable { name: name, typename: typename, slot: slot } }
    )
}

fn functionparam(input: &[Token]) -> IResult<&[Token], FunctionParam, ParseError> {
    chain!(input,
        typename: parse_typename ~
        param: parse_variablename ~
        opt!(chain!(
            token!(Token::Colon) ~
            tok: token!(Token::Id(_)), // Todo: Semantics
            || { tok }
        )),
        || { FunctionParam { name: param, typename: typename } }
    )
}

fn functiondefinition(input: &[Token]) -> IResult<&[Token], FunctionDefinition, ParseError> {
    chain!(input,
        ret: parse_typename ~
        func_name: parse_variablename ~
        params: delimited!(
            token!(Token::LeftParen),
            separated_list!(token!(Token::Comma), functionparam),
            token!(Token::RightParen)
        ) ~
        body: delimited!(
            token!(Token::LeftBrace),
            many0!(statement),
            token!(Token::RightBrace)
        ),
        || { FunctionDefinition { name: func_name, returntype: ret, params: params, body: body } }
    )
}



// Todo: Attribute parsing
fn attribute(input: &[Token]) -> IResult<&[Token], Vec<()>, ParseError> {

    fn not_end(input: &[Token]) -> IResult<&[Token], (), ParseError> {
        if input.len() == 0 {
            IResult::Incomplete(Needed::Size(1))
        } else {
            match input[0] {
                Token::RightSquareBracket => IResult::Error(Err::Position(ErrorKind::Custom(ParseError::WrongToken), input)),
                _ => IResult::Done(&input[1..], ()),
            }
        }
    }

    delimited!(input, token!(Token::LeftSquareBracket), many0!(not_end), token!(Token::RightSquareBracket))
}

fn rootdefinition(input: &[Token]) -> IResult<&[Token], RootDefinition, ParseError> {
    chain!(input, many0!(attribute) ~ def: alt!(
        structdefinition => { |structdef| { RootDefinition::Struct(structdef) } } |
        cbuffer => { |cbuffer| { RootDefinition::ConstantBuffer(cbuffer) } } |
        globalvariable => { |globalvariable| { RootDefinition::GlobalVariable(globalvariable) } } |
        functiondefinition => { |funcdef| { RootDefinition::Function(funcdef) } }
    ), || { def })
}

pub fn module(input: &[Token]) -> IResult<&[Token], Vec<RootDefinition>, ParseError> {
    many0!(input, rootdefinition)
}

pub fn parse(entry_point: String, source: &[Token]) -> Option<Module> {
    let parse_result = module(source);
    match parse_result {
        IResult::Done(rest, hlsl) => if rest.len() == 0 { Some(Module { entry_point: entry_point, root_definitions: hlsl }) } else { None },
        IResult::Error(_) => None,
        IResult::Incomplete(_) => None,
    }
}


#[cfg(test)]
fn exp_var(var_name: &'static str) -> Expression { Expression::Variable(var_name.to_string()) }
#[cfg(test)]
fn bexp_var(var_name: &'static str) -> Box<Expression> { Box::new(exp_var(var_name)) }

#[cfg(test)]
fn parse_from_str<T>(parse_func: Box<Fn(&[Token]) -> IResult<&[Token], T, ParseError>>) -> Box<Fn(&'static str) -> T> where T: 'static {
    Box::new(move |string: &'static str| {
        let input = string.as_bytes();
        let lex_result = super::lexer::token_stream(input);
        match lex_result {
            IResult::Done(rem, TokenStream(mut stream)) => {
                if rem.len() != 0 { panic!("Tokens remaining while lexing `{:?}`: {:?}", string, rem) };
                stream.push(Token::Eof);
                match parse_func(&stream[..]) {
                    IResult::Done(rem, exp) => {
                        if rem == &[Token::Eof] {
                            exp
                        } else {
                            panic!("Tokens remaining while parsing `{:?}`: {:?}", stream, rem)
                        }
                    },
                    IResult::Incomplete(needed) => panic!("Failed to parse `{:?}`: Needed {:?} more", stream, needed),
                    _ => panic!("Failed to parse {:?}", stream)
                }
            }
            _ => panic!("Failed to lex `{:?}`", string)
        }
    })
}

#[test]
fn test_expr() {

    assert_eq!(expr(&[Token::Id(Identifier("a".to_string())), Token::Asterix, Token::Id(Identifier("b".to_string())), Token::Eof][..]), IResult::Done(&[Token::Eof][..],
        Expression::BinaryOperation(
            BinOp::Multiply,
            Box::new(Expression::Variable("a".to_string())),
            Box::new(Expression::Variable("b".to_string()))
        )
    ));

    let expr_str = parse_from_str(Box::new(expr));

    assert_eq!(expr_str("a"), exp_var("a"));
    assert_eq!(expr_str("4"), Expression::LiteralInt(4));
    assert_eq!(expr_str("a+b"), Expression::BinaryOperation(BinOp::Add, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a*b"), Expression::BinaryOperation(BinOp::Multiply, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a + b"), Expression::BinaryOperation(BinOp::Add, bexp_var("a"), bexp_var("b")));

    assert_eq!(expr_str("a-b+c"), Expression::BinaryOperation(
        BinOp::Add,
        Box::new(Expression::BinaryOperation(BinOp::Subtract, bexp_var("a"), bexp_var("b"))),
        bexp_var("c")
    ));
    assert_eq!(expr_str("a-b*c"), Expression::BinaryOperation(
        BinOp::Subtract,
        bexp_var("a"),
        Box::new(Expression::BinaryOperation(BinOp::Multiply, bexp_var("b"), bexp_var("c")))
    ));
    assert_eq!(expr_str("a*b-c"), Expression::BinaryOperation(
        BinOp::Subtract,
        Box::new(Expression::BinaryOperation(BinOp::Multiply, bexp_var("a"), bexp_var("b"))),
        bexp_var("c")
    ));
    assert_eq!(expr_str("a-b*c"), Expression::BinaryOperation(
        BinOp::Subtract,
        bexp_var("a"),
        Box::new(Expression::BinaryOperation(BinOp::Multiply, bexp_var("b"), bexp_var("c")))
    ));
    assert_eq!(expr_str("a*b-c"), Expression::BinaryOperation(
        BinOp::Subtract,
        Box::new(Expression::BinaryOperation(BinOp::Multiply, bexp_var("a"), bexp_var("b"))),
        bexp_var("c")
    ));
    assert_eq!(expr_str("a*(b-c)"), Expression::BinaryOperation(
        BinOp::Multiply,
        bexp_var("a"),
        Box::new(Expression::BinaryOperation(BinOp::Subtract, bexp_var("b"), bexp_var("c")))
    ));
    assert_eq!(expr_str("a*b/c"), Expression::BinaryOperation(
        BinOp::Divide,
        Box::new(Expression::BinaryOperation(BinOp::Multiply, bexp_var("a"), bexp_var("b"))),
        bexp_var("c")
    ));
    assert_eq!(expr_str("(a*b)/c"), Expression::BinaryOperation(
        BinOp::Divide,
        Box::new(Expression::BinaryOperation(BinOp::Multiply, bexp_var("a"), bexp_var("b"))),
        bexp_var("c")
    ));
    assert_eq!(expr_str("a*(b/c)"), Expression::BinaryOperation(
        BinOp::Multiply,
        bexp_var("a"),
        Box::new(Expression::BinaryOperation(BinOp::Divide, bexp_var("b"), bexp_var("c")))
    ));

    assert_eq!(expr_str("a++"), Expression::UnaryOperation(UnaryOp::PostfixIncrement, bexp_var("a")));
    assert_eq!(expr_str("a--"), Expression::UnaryOperation(UnaryOp::PostfixDecrement, bexp_var("a")));
    assert_eq!(expr_str("++a"), Expression::UnaryOperation(UnaryOp::PrefixIncrement, bexp_var("a")));
    assert_eq!(expr_str("--a"), Expression::UnaryOperation(UnaryOp::PrefixDecrement, bexp_var("a")));
    assert_eq!(expr_str("+a"), Expression::UnaryOperation(UnaryOp::Plus, bexp_var("a")));
    assert_eq!(expr_str("-a"), Expression::UnaryOperation(UnaryOp::Minus, bexp_var("a")));
    assert_eq!(expr_str("!a"), Expression::UnaryOperation(UnaryOp::LogicalNot, bexp_var("a")));
    assert_eq!(expr_str("~a"), Expression::UnaryOperation(UnaryOp::BitwiseNot, bexp_var("a")));

    assert_eq!(expr_str("a[b]"),
        Expression::ArraySubscript(bexp_var("a"), bexp_var("b"))
    );
    assert_eq!(expr_str("d+a[b+c]"),
        Expression::BinaryOperation(BinOp::Add,
            bexp_var("d"),
            Box::new(Expression::ArraySubscript(bexp_var("a"),
                Box::new(Expression::BinaryOperation(BinOp::Add,
                    bexp_var("b"), bexp_var("c")
                ))
            ))
        )
    );
    assert_eq!(expr_str(" d + a\t[ b\n+ c ]"),
        Expression::BinaryOperation(BinOp::Add,
            bexp_var("d"),
            Box::new(Expression::ArraySubscript(bexp_var("a"),
                Box::new(Expression::BinaryOperation(BinOp::Add,
                    bexp_var("b"), bexp_var("c")
                ))
            ))
        )
    );

    assert_eq!(expr_str("array.Load"),
        Expression::Member(bexp_var("array"), "Load".to_string())
    );
    assert_eq!(expr_str("array.Load()"),
        Expression::Call(Box::new(Expression::Member(bexp_var("array"), "Load".to_string())), vec![])
    );
    assert_eq!(expr_str(" array . Load ( ) "),
        Expression::Call(Box::new(Expression::Member(bexp_var("array"), "Load".to_string())), vec![])
    );
    assert_eq!(expr_str("array.Load(a)"),
        Expression::Call(Box::new(Expression::Member(bexp_var("array"), "Load".to_string())), vec![exp_var("a")])
    );
    assert_eq!(expr_str("array.Load(a,b)"),
        Expression::Call(Box::new(Expression::Member(bexp_var("array"), "Load".to_string())), vec![exp_var("a"), exp_var("b")])
    );
    assert_eq!(expr_str("array.Load(a, b)"),
        Expression::Call(Box::new(Expression::Member(bexp_var("array"), "Load".to_string())), vec![exp_var("a"), exp_var("b")])
    );

    assert_eq!(expr_str("(float) b"),
        Expression::Cast(TypeName("float".to_string()), bexp_var("b"))
    );

    assert_eq!(expr_str("a = b"), Expression::BinaryOperation(BinOp::Assignment, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a = b = c"), Expression::BinaryOperation(
        BinOp::Assignment,
        bexp_var("a"),
        Box::new(Expression::BinaryOperation(
            BinOp::Assignment,
            bexp_var("b"),
            bexp_var("c")
        ))
    ));
}

#[test]
fn test_statement() {

    let statement_str = parse_from_str(Box::new(statement));

    // Empty statement
    assert_eq!(statement_str(";"), Statement::Empty);

    // Expression statements
    assert_eq!(statement_str("func();"),
        Statement::Expression(Expression::Call(bexp_var("func"), vec![]))
    );
    assert_eq!(statement_str("func();"), statement_str(" func ( ) ; "));

    // Condition expressions
    let condition_str = parse_from_str(Box::new(condition));
    let vardef_str = parse_from_str(Box::new(vardef));

    assert_eq!(condition_str("x"),
        Condition::Expr(exp_var("x"))
    );
    assert_eq!(vardef_str("uint x"),
        VarDef::new("x".to_string(), TypeName("uint".to_string()), None)
    );
    assert_eq!(condition_str("uint x"),
        Condition::Assignment(VarDef::new("x".to_string(), TypeName("uint".to_string()), None))
    );
    assert_eq!(condition_str("uint x = y"),
        Condition::Assignment(VarDef::new("x".to_string(), TypeName("uint".to_string()), Some(exp_var("y"))))
    );

    // Variable declarations
    assert_eq!(statement_str("uint x = y;"),
        Statement::Var(VarDef::new("x".to_string(), TypeName("uint".to_string()), Some(exp_var("y"))))
    );

    // Blocks
    assert_eq!(statement_str("{one();two();}"),
        Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])
    );
    assert_eq!(statement_str("{one();two();}"), statement_str(" { one(); two(); } "));

    // If statement
    assert_eq!(statement_str("if(a)func();"),
        Statement::If(Condition::Expr(exp_var("a")), Box::new(Statement::Expression(Expression::Call(bexp_var("func"), vec![]))))
    );
    assert_eq!(statement_str("if(a)func();"), statement_str("if (a) func(); "));
    assert_eq!(statement_str("if (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::If(Condition::Expr(exp_var("a")), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])))
    );
    assert_eq!(statement_str("if (uint x = y)\n{\n\tone();\n\ttwo();\n}"),
        Statement::If(Condition::Assignment(VarDef::new("x".to_string(), TypeName("uint".to_string()), Some(exp_var("y")))), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])))
    );

    // While loops
    assert_eq!(statement_str("while (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::While(Condition::Expr(exp_var("a")), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])))
    );
    assert_eq!(statement_str("while (int x = y)\n{\n\tone();\n\ttwo();\n}"),
        Statement::While(Condition::Assignment(VarDef::new("x".to_string(), TypeName("int".to_string()), Some(exp_var("y")))), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])))
    );

    // For loops
    assert_eq!(statement_str("for(a;b;c)func();"),
        Statement::For(Condition::Expr(exp_var("a")), Condition::Expr(exp_var("b")), Condition::Expr(exp_var("c")), Box::new(
            Statement::Expression(Expression::Call(bexp_var("func"), vec![]))
        ))
    );
    assert_eq!(statement_str("for (uint i = 0; i; i++) { func(); }"),
        Statement::For(
            Condition::Assignment(VarDef::new("i".to_string(), TypeName("uint".to_string()), Some(Expression::LiteralInt(0)))),
            Condition::Expr(exp_var("i")),
            Condition::Expr(Expression::UnaryOperation(UnaryOp::PostfixIncrement, bexp_var("i"))),
            Box::new(Statement::Block(vec![Statement::Expression(Expression::Call(bexp_var("func"), vec![]))]))
        )
    );
}

#[test]
fn test_rootdefinition() {

    let rootdefinition_str = parse_from_str(Box::new(rootdefinition));

    let structdefinition_str = parse_from_str(Box::new(structdefinition));

    let test_struct_str = "struct MyStruct { uint a; float b; };";
    let test_struct_ast = StructDefinition {
        name: TypeName("MyStruct".to_string()),
        members: vec![
            StructMember { name: "a".to_string(), typename: TypeName("uint".to_string()) },
            StructMember { name: "b".to_string(), typename: TypeName("float".to_string()) },
        ]
    };
    assert_eq!(structdefinition_str(test_struct_str), test_struct_ast.clone());
    assert_eq!(rootdefinition_str(test_struct_str), RootDefinition::Struct(test_struct_ast.clone()));

    let functiondefinition_str = parse_from_str(Box::new(functiondefinition));

    let test_func_str = "void func(float x) { }";
    let test_func_ast = FunctionDefinition { 
        name: "func".to_string(),
        returntype: TypeName("void".to_string()),
        params: vec![FunctionParam { name: "x".to_string(), typename: TypeName("float".to_string()) }],
        body: vec![],
    };
    assert_eq!(functiondefinition_str(test_func_str), test_func_ast.clone());
    assert_eq!(rootdefinition_str(test_func_str), RootDefinition::Function(test_func_ast.clone()));
    assert_eq!(rootdefinition_str("[numthreads(1, 1, 1)] void func(float x) { }"), RootDefinition::Function(FunctionDefinition {
        name: "func".to_string(),
        returntype: TypeName("void".to_string()),
        params: vec![FunctionParam { name: "x".to_string(), typename: TypeName("float".to_string()) }],
        body: vec![],
    }));

    let constantvariable_str = parse_from_str(Box::new(constantvariable));

    let test_cbuffervar_str = "float4x4 wvp;";
    let test_cbuffervar_ast = ConstantVariable { name: "wvp".to_string(), typename: TypeName("float4x4".to_string()), offset: None };
    assert_eq!(constantvariable_str(test_cbuffervar_str), test_cbuffervar_ast.clone());

    let cbuffer_str = parse_from_str(Box::new(cbuffer));

    let test_cbuffer1_str = "cbuffer globals { float4x4 wvp; };";
    let test_cbuffer1_ast = ConstantBuffer {
        name: "globals".to_string(),
        slot: None,
        members: vec![
            ConstantVariable { name: "wvp".to_string(), typename: TypeName("float4x4".to_string()), offset: None },
        ]
    };
    assert_eq!(cbuffer_str(test_cbuffer1_str), test_cbuffer1_ast.clone());
    assert_eq!(rootdefinition_str(test_cbuffer1_str), RootDefinition::ConstantBuffer(test_cbuffer1_ast.clone()));

    let cbuffer_register_str = parse_from_str(Box::new(cbuffer_register));
    assert_eq!(cbuffer_register_str(" : register(b12) "), ConstantSlot(12));

    let test_cbuffer2_str = "cbuffer globals : register(b12) { float4x4 wvp; };";
    let test_cbuffer2_ast = ConstantBuffer {
        name: "globals".to_string(),
        slot: Some(ConstantSlot(12)),
        members: vec![
            ConstantVariable { name: "wvp".to_string(), typename: TypeName("float4x4".to_string()), offset: None },
        ]
    };
    assert_eq!(cbuffer_str(test_cbuffer2_str), test_cbuffer2_ast.clone());
    assert_eq!(rootdefinition_str(test_cbuffer2_str), RootDefinition::ConstantBuffer(test_cbuffer2_ast.clone()));

    let globalvariable_str = parse_from_str(Box::new(globalvariable));

    let test_buffersrv_str = "Buffer g_myBuffer : register(t1);";
    let test_buffersrv_ast = GlobalVariable {
        name: "g_myBuffer".to_string(),
        typename: TypeName("Buffer".to_string()),
        slot: Some(GlobalSlot::ReadSlot(1)),
    };
    assert_eq!(globalvariable_str(test_buffersrv_str), test_buffersrv_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv_str), RootDefinition::GlobalVariable(test_buffersrv_ast.clone()));
}

