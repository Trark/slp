use std::error;
use std::fmt;
use super::tokens::*;
use super::ast::*;
use nom::{IResult,Needed,Err,ErrorKind};

#[derive(PartialEq, Debug, Clone)]
pub struct ParseError(pub ParseErrorReason, pub Option<Vec<Token>>, pub Option<Box<ParseError>>);

#[derive(PartialEq, Debug, Clone)]
pub enum ParseErrorReason {
    Unknown,
    UnexpectedEndOfStream,
    FailedToParse,
    WrongToken,
    ExpectingIdentifier,
    WrongSlotType,
    UnknownType,
}

impl error::Error for ParseError {
    fn description(&self) -> &str {
        "parser error"
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

macro_rules! token (
    ($i:expr, $inp: pat) => (
        {
            let res: IResult<&[Token], Token, ParseErrorReason> = if $i.len() == 0 {
                IResult::Incomplete(Needed::Size(1))
            } else {
                match $i[0] {
                    $inp => IResult::Done(&$i[1..], $i[0].clone()),
                    _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken), $i))
                } 
            };
            res
        }
    );
    ($i:expr, $inp: pat => $res: expr) => (
        {
            if $i.len() == 0 {
                IResult::Incomplete(Needed::Size(1))
            } else {
                match $i[0] {
                    $inp => IResult::Done(&$i[1..], $res),
                    _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken), $i))
                }
            }
        }
    );
);

fn parse_variablename(input: &[Token]) -> IResult<&[Token], String, ParseErrorReason> {
    map!(input, token!(Token::Id(_)), |tok| { match tok { Token::Id(Identifier(name)) => name.clone(), _ => unreachable!() } })
}

fn parse_datalayout(input: &[Token]) -> IResult<&[Token], DataLayout, ParseErrorReason> {

    // Parse a vector dimension as a token
    fn parse_digit(input: &[Token]) -> IResult<&[Token], u32, ParseErrorReason> {
        token!(input, Token::LiteralInt(i) => i as u32)
    }

    // Parse scalar type as part of a string
    fn parse_scalartype_str(input: &[u8]) -> IResult<&[u8], ScalarType> {
        alt!(input,
            complete!(tag!("bool")) => { |_| ScalarType::Bool } |
            complete!(tag!("int")) => { |_| ScalarType::Int } |
            complete!(tag!("uint")) => { |_| ScalarType::UInt } |
            complete!(tag!("dword")) => { |_| ScalarType::UInt } |
            complete!(tag!("float")) => { |_| ScalarType::Float } |
            complete!(tag!("double")) => { |_| ScalarType::Double }
        )
    }

    // Parse scalar type as a full token
    fn parse_scalartype(input: &[Token]) -> IResult<&[Token], ScalarType, ParseErrorReason> {
        if input.len() == 0 {
            IResult::Incomplete(Needed::Size(1))
        } else {
            match &input[0] {
                &Token::Id(Identifier(ref name)) => {
                    match parse_scalartype_str(&name[..].as_bytes()) {
                        IResult::Done(rest, ty) => if rest.len() == 0 {
                            IResult::Done(&input[1..], ty)
                        } else {
                            IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input))
                        },
                        IResult::Incomplete(rem) => IResult::Incomplete(rem),
                        IResult::Error(_) => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input)),
                    }
                },
                _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken), input))
            }
        }
    }


    fn parse_datatype_str(typename: &str) -> Option<DataLayout> {

        fn digit(input: &[u8]) -> IResult<&[u8], u32> {
            alt!(input,
                tag!("1") => { |_| { 1 } } |
                tag!("2") => { |_| { 2 } } |
                tag!("3") => { |_| { 3 } } |
                tag!("4") => { |_| { 4 } }
            )
        }

        fn parse_str(input: &[u8]) -> IResult<&[u8], DataLayout> {
            match parse_scalartype_str(input) {
                IResult::Incomplete(rem) => IResult::Incomplete(rem),
                IResult::Error(err) => IResult::Error(err),
                IResult::Done(rest, ty) => {
                    if rest.len() == 0 {
                        IResult::Done(&[], DataLayout::Scalar(ty))
                    } else {
                        match digit(rest) {
                            IResult::Incomplete(rem) => IResult::Incomplete(rem),
                            IResult::Error(err) => IResult::Error(err),
                            IResult::Done(rest, x) => {
                                if rest.len() == 0 {
                                    IResult::Done(&[], DataLayout::Vector(ty, x))
                                } else {
                                    match preceded!(rest, tag!("x"), digit) {
                                        IResult::Incomplete(rem) => IResult::Incomplete(rem),
                                        IResult::Error(err) => IResult::Error(err),
                                        IResult::Done(rest, y) => if rest.len() == 0 {
                                            IResult::Done(&[], DataLayout::Matrix(ty, x, y))
                                        } else {
                                            IResult::Error(Err::Position(ErrorKind::Custom(0), input))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        match parse_str(&typename[..].as_bytes()) {
            IResult::Done(rest, ty) => { assert_eq!(rest.len(), 0); Some(ty) },
            IResult::Incomplete(_) | IResult::Error(_) => None,
        }
    }

    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        match &input[0] {
            &Token::Id(Identifier(ref name)) => {
                match &name[..] {
                    "vector" => {
                        chain!(&input[1..],
                            token!(Token::LeftAngleBracket(_)) ~
                            scalar: parse_scalartype ~
                            token!(Token::Comma) ~
                            x: parse_digit ~
                            token!(Token::RightAngleBracket(_)),
                            || { DataLayout::Vector(scalar, x) }
                        )
                    },
                    "matrix" => {
                        chain!(&input[1..],
                            token!(Token::LeftAngleBracket(_)) ~
                            scalar: parse_scalartype ~
                            token!(Token::Comma) ~
                            x: parse_digit ~ 
                            token!(Token::Comma) ~
                            y: parse_digit ~
                            token!(Token::RightAngleBracket(_)),
                            || { DataLayout::Matrix(scalar, x, y) }
                        )
                    },
                    _ => match parse_datatype_str(&name[..]) {
                        Some(ty) => IResult::Done(&input[1..], ty),
                        None => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input)),
                    }
                }
            },
            _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken), input))
        }
    }
}

fn parse_datatype(input: &[Token]) -> IResult<&[Token], DataType, ParseErrorReason> {
    // Todo: Modifiers
    match parse_datalayout(input) {
        IResult::Done(rest, layout) => IResult::Done(rest, DataType(layout, Default::default())),
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_structuredlayout(input: &[Token]) -> IResult<&[Token], StructuredLayout, ParseErrorReason> {
    alt!(input,
        parse_datalayout => { |ty| { match ty {
                DataLayout::Scalar(scalar) => StructuredLayout::Scalar(scalar),
                DataLayout::Vector(scalar, x) => StructuredLayout::Vector(scalar, x),
                DataLayout::Matrix(scalar, x, y) => StructuredLayout::Matrix(scalar, x, y),
            }
        } } |
        token!(Token::Id(Identifier(ref name)) => StructuredLayout::Custom(name.clone()))
    )
}

fn parse_structuredtype(input: &[Token]) -> IResult<&[Token], StructuredType, ParseErrorReason> {
    // Todo: Modifiers
    match parse_structuredlayout(input) {
        IResult::Done(rest, layout) => IResult::Done(rest, StructuredType(layout, Default::default())),
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_objecttype(input: &[Token]) -> IResult<&[Token], ObjectType, ParseErrorReason> {
    if input.len() == 0 {
        return IResult::Incomplete(Needed::Size(1))
    }

    enum ParseType {
        Buffer,
        RWBuffer,

        ByteAddressBuffer,
        RWByteAddressBuffer,

        StructuredBuffer,
        RWStructuredBuffer,
        AppendStructuredBuffer,
        ConsumeStructuredBuffer,

        Texture1D,
        Texture1DArray,
        Texture2D,
        Texture2DArray,
        Texture2DMS,
        Texture2DMSArray,
        Texture3D,
        TextureCube,
        TextureCubeArray,
        RWTexture1D,
        RWTexture1DArray,
        RWTexture2D,
        RWTexture2DArray,
        RWTexture3D,

        InputPatch,
        OutputPatch,
    }

    let object_type = match &input[0] {
        &Token::Id(Identifier(ref name)) => {
            match &name[..] {
                "Buffer" => ParseType::Buffer,
                "RWBuffer" => ParseType::RWBuffer,

                "ByteAddressBuffer" => ParseType::ByteAddressBuffer,
                "RWByteAddressBuffer" => ParseType::RWByteAddressBuffer,

                "StructuredBuffer" => ParseType::StructuredBuffer,
                "RWStructuredBuffer" => ParseType::RWStructuredBuffer,
                "AppendStructuredBuffer" => ParseType::AppendStructuredBuffer,
                "ConsumeStructuredBuffer" => ParseType::ConsumeStructuredBuffer,

                "Texture1D" => ParseType::Texture1D,
                "Texture1DArray" => ParseType::Texture1DArray,
                "Texture2D" => ParseType::Texture2D,
                "Texture2DArray" => ParseType::Texture2DArray,
                "Texture2DMS" => ParseType::Texture2DMS,
                "Texture2DMSArray" => ParseType::Texture2DMSArray,
                "Texture3D" => ParseType::Texture3D,
                "TextureCube" => ParseType::TextureCube,
                "TextureCubeArray" => ParseType::TextureCubeArray,
                "RWTexture1D" => ParseType::RWTexture1D,
                "RWTexture1DArray" => ParseType::RWTexture1DArray,
                "RWTexture2D" => ParseType::RWTexture2D,
                "RWTexture2DArray" => ParseType::RWTexture2DArray,
                "RWTexture3D" => ParseType::RWTexture3D,

                "InputPatch" => ParseType::InputPatch,
                "OutputPatch" => ParseType::OutputPatch,

                _ => return IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input))
            }
        },
        _ => return IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input))
    };

    let rest = &input[1..];

    match object_type {

        ParseType::ByteAddressBuffer => IResult::Done(rest, ObjectType::ByteAddressBuffer),
        ParseType::RWByteAddressBuffer => IResult::Done(rest, ObjectType::RWByteAddressBuffer),

        ParseType::Buffer |
        ParseType::RWBuffer |
        ParseType::Texture1D |
        ParseType::Texture1DArray |
        ParseType::Texture2D |
        ParseType::Texture2DArray |
        ParseType::Texture2DMS |
        ParseType::Texture2DMSArray |
        ParseType::Texture3D |
        ParseType::TextureCube |
        ParseType::TextureCubeArray |
        ParseType::RWTexture1D |
        ParseType::RWTexture1DArray |
        ParseType::RWTexture2D |
        ParseType::RWTexture2DArray |
        ParseType::RWTexture3D => {

            let (buffer_arg, rest) = match delimited!(rest, token!(Token::LeftAngleBracket(_)), parse_datatype, token!(Token::RightAngleBracket(_))) {
                IResult::Done(rest, ty) => (ty, rest),
                IResult::Incomplete(rem) => return IResult::Incomplete(rem),
                IResult::Error(_) => (DataType(DataLayout::Vector(ScalarType::Float, 4), TypeModifier::default()), rest)
            };

            IResult::Done(rest, match object_type {
                ParseType::Buffer => ObjectType::Buffer(buffer_arg),
                ParseType::RWBuffer => ObjectType::RWBuffer(buffer_arg),
                ParseType::Texture1D => ObjectType::Texture1D(buffer_arg),
                ParseType::Texture1DArray => ObjectType::Texture1DArray(buffer_arg),
                ParseType::Texture2D => ObjectType::Texture2D(buffer_arg),
                ParseType::Texture2DArray => ObjectType::Texture2DArray(buffer_arg),
                ParseType::Texture2DMS => ObjectType::Texture2DMS(buffer_arg),
                ParseType::Texture2DMSArray => ObjectType::Texture2DMSArray(buffer_arg),
                ParseType::Texture3D => ObjectType::Texture3D(buffer_arg),
                ParseType::TextureCube => ObjectType::TextureCube(buffer_arg),
                ParseType::TextureCubeArray => ObjectType::TextureCubeArray(buffer_arg),
                ParseType::RWTexture1D => ObjectType::RWTexture1D(buffer_arg),
                ParseType::RWTexture1DArray => ObjectType::RWTexture1DArray(buffer_arg),
                ParseType::RWTexture2D => ObjectType::RWTexture2D(buffer_arg),
                ParseType::RWTexture2DArray => ObjectType::RWTexture2DArray(buffer_arg),
                ParseType::RWTexture3D => ObjectType::RWTexture3D(buffer_arg),
                _ => unreachable!(),
            })
        },

        ParseType::StructuredBuffer |
        ParseType::RWStructuredBuffer |
        ParseType::AppendStructuredBuffer |
        ParseType::ConsumeStructuredBuffer => {

            let (buffer_arg, rest) = match delimited!(rest, token!(Token::LeftAngleBracket(_)), parse_structuredtype, token!(Token::RightAngleBracket(_))) {
                IResult::Done(rest, ty) => (ty, rest),
                IResult::Incomplete(rem) => return IResult::Incomplete(rem),
                IResult::Error(_) => (StructuredType(StructuredLayout::Vector(ScalarType::Float, 4), TypeModifier::default()), rest)
            };

            IResult::Done(rest, match object_type {
                ParseType::StructuredBuffer => ObjectType::StructuredBuffer(buffer_arg),
                ParseType::RWStructuredBuffer => ObjectType::RWStructuredBuffer(buffer_arg),
                ParseType::AppendStructuredBuffer => ObjectType::AppendStructuredBuffer(buffer_arg),
                ParseType::ConsumeStructuredBuffer => ObjectType::ConsumeStructuredBuffer(buffer_arg),
                _ => unreachable!(),
            })
        },

        ParseType::InputPatch => IResult::Done(rest, ObjectType::InputPatch),
        ParseType::OutputPatch => IResult::Done(rest, ObjectType::OutputPatch),
    }

}

fn parse_voidtype(input: &[Token]) -> IResult<&[Token], TypeLayout, ParseErrorReason> {
    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        match &input[0] {
            &Token::Id(Identifier(ref name)) => if name == "void" {
                IResult::Done(&input[1..], TypeLayout::Void)
            } else {
                IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input))
            },
            _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken), input)),
        }
    }
}

fn parse_typelayout(input: &[Token]) -> IResult<&[Token], TypeLayout, ParseErrorReason> {
    alt!(input,
        parse_objecttype => { |ty| { TypeLayout::Object(ty) } } |
        parse_voidtype |
        token!(Token::SamplerState => TypeLayout::SamplerState) |
        // Structured types eat everything as user defined types so must come last
        parse_structuredlayout => { |ty| { TypeLayout::from(ty) } }
    )
}

fn parse_typename(input: &[Token]) -> IResult<&[Token], Type, ParseErrorReason> {
    // Todo: Modifiers
    match parse_typelayout(input) {
        IResult::Done(rest, layout) => IResult::Done(rest, Type(layout, Default::default())),
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_paramtype(input: &[Token]) -> IResult<&[Token], ParamType, ParseErrorReason> {
    // Todo: input modifiers
    match parse_typename(input) {
        IResult::Done(rest, ty) => IResult::Done(rest, ParamType(ty, InputModifier::default(), None)),
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn expr_paren(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {
    alt!(input,
        delimited!(token!(Token::LeftParen), expr, token!(Token::RightParen)) |
        parse_variablename => { |name| { Expression::Variable(name) } } |
        token!(Token::LiteralInt(i) => Expression::Literal(Literal::UntypedInt(i))) |
        token!(Token::LiteralUint(i) => Expression::Literal(Literal::UInt(i))) |
        token!(Token::LiteralLong(i) => Expression::Literal(Literal::Long(i))) |
        token!(Token::LiteralHalf(i) => Expression::Literal(Literal::Half(i))) |
        token!(Token::LiteralFloat(i) => Expression::Literal(Literal::Float(i))) |
        token!(Token::LiteralDouble(i) => Expression::Literal(Literal::Double(i))) |
        token!(Token::True => Expression::Literal(Literal::Bool(true))) |
        token!(Token::False => Expression::Literal(Literal::Bool(false)))
    )
}

fn expr_p1(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {

    #[derive(Clone)]
    enum Precedence1Postfix {
        Increment,
        Decrement,
        Call(Vec<Expression>),
        ArraySubscript(Expression),
        Member(String),
    }

    fn expr_p1_right(input: &[Token]) -> IResult<&[Token], Precedence1Postfix, ParseErrorReason> {
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

fn unaryop_prefix(input: &[Token]) -> IResult<&[Token], UnaryOp, ParseErrorReason> {
    alt!(input,
        chain!(token!(Token::Plus) ~ token!(Token::Plus), || { UnaryOp::PrefixIncrement }) |
        chain!(token!(Token::Minus) ~ token!(Token::Minus), || { UnaryOp::PrefixDecrement }) |
        token!(Token::Plus) => { |_| UnaryOp::Plus } |
        token!(Token::Minus) => { |_| UnaryOp::Minus } |
        token!(Token::ExclamationPoint) => { |_| UnaryOp::LogicalNot } |
        token!(Token::Tilde) => { |_| UnaryOp::BitwiseNot }
    )
}

fn expr_p2(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {
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

fn expr_p3(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {

    fn binop_p3(input: &[Token]) -> IResult<&[Token], BinOp, ParseErrorReason> {
        alt!(input,
            token!(Token::Asterix) => { |_| BinOp::Multiply } |
            token!(Token::ForwardSlash) => { |_| BinOp::Divide } |
            token!(Token::Percent) => { |_| BinOp::Modulus }
        )
    }

    fn expr_p3_right(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseErrorReason> {
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

fn expr_p4(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {

    fn binop_p4(input: &[Token]) -> IResult<&[Token], BinOp, ParseErrorReason> {
        alt!(input,
            token!(Token::Plus) => { |_| BinOp::Add } |
            token!(Token::Minus) => { |_| BinOp::Subtract }
        )
    }

    fn expr_p4_right(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseErrorReason> {
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

fn expr_p5(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {

    fn parse_op(input: &[Token]) -> IResult<&[Token], BinOp, ParseErrorReason> {
        alt!(input,
            chain!(token!(Token::LeftAngleBracket(FollowedBy::Token)) ~ token!(Token::LeftAngleBracket(_)), || { BinOp::LeftShift }) |
            chain!(token!(Token::RightAngleBracket(FollowedBy::Token)) ~ token!(Token::RightAngleBracket(_)), || { BinOp::RightShift })
        )
    }

    fn parse_rights(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseErrorReason> {
        chain!(input,
            op: parse_op ~
            right: expr_p4,
            || { return (op, right) }
        )
    }

    chain!(input,
        left: expr_p4 ~
        rights: many0!(parse_rights),
        || { combine_rights(left, rights) }
    )
}

fn expr_p6(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {

    fn parse_op(input: &[Token]) -> IResult<&[Token], BinOp, ParseErrorReason> {
        alt!(input,
            chain!(token!(Token::LeftAngleBracket(FollowedBy::Token)) ~ token!(Token::Equals), || { BinOp::LessEqual }) |
            chain!(token!(Token::RightAngleBracket(FollowedBy::Token)) ~ token!(Token::Equals), || { BinOp::GreaterEqual }) |
            token!(Token::LeftAngleBracket(_)) => { |_| BinOp::LessThan } |
            token!(Token::RightAngleBracket(_)) => { |_| BinOp::GreaterThan }
        )
    }

    fn parse_rights(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseErrorReason> {
        chain!(input,
            op: parse_op ~
            right: expr_p5,
            || { return (op, right) }
        )
    }

    chain!(input,
        left: expr_p5 ~
        rights: many0!(parse_rights),
        || { combine_rights(left, rights) }
    )
}

fn expr_p7(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {

    fn parse_op(input: &[Token]) -> IResult<&[Token], BinOp, ParseErrorReason> {
        alt!(input,
            token!(Token::DoubleEquals) => { |_| BinOp::Equality } |
            token!(Token::ExclamationEquals) => { |_| BinOp::Inequality }
        )
    }

    fn parse_rights(input: &[Token]) -> IResult<&[Token], (BinOp, Expression), ParseErrorReason> {
        chain!(input,
            op: parse_op ~
            right: expr_p6,
            || { return (op, right) }
        )
    }

    chain!(input,
        left: expr_p6 ~
        rights: many0!(parse_rights),
        || { combine_rights(left, rights) }
    )
}

fn expr_p15(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {
    alt!(input,
        chain!(lhs: expr_p7 ~ token!(Token::Equals) ~ rhs: expr_p15, || { Expression::BinaryOperation(BinOp::Assignment, Box::new(lhs), Box::new(rhs)) }) |
        expr_p7
    )
}

fn expr(input: &[Token]) -> IResult<&[Token], Expression, ParseErrorReason> {
    expr_p15(input)
}

fn vardef(input: &[Token]) -> IResult<&[Token], VarDef, ParseErrorReason> {
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

fn condition(input: &[Token]) -> IResult<&[Token], Condition, ParseErrorReason> {
    alt!(input,
        vardef => { |variable_definition| Condition::Assignment(variable_definition) } |
        expr => { |expression| Condition::Expr(expression) }
    )
}

fn statement(input: &[Token]) -> IResult<&[Token], Statement, ParseErrorReason> {
    alt!(input,
        token!(Token::Semicolon) => { |_| { Statement::Empty } } |
        chain!(token!(Token::If) ~ token!(Token::LeftParen) ~ cond: expr ~ token!(Token::RightParen) ~ inner_statement: statement, || { Statement::If(cond, Box::new(inner_statement)) }) |
        chain!(token!(Token::LeftBrace) ~ statements: many0!(statement) ~ token!(Token::RightBrace), || { Statement::Block(statements) }) |
        chain!(
            token!(Token::For) ~
            token!(Token::LeftParen) ~
            init: condition ~
            token!(Token::Semicolon) ~
            cond: expr ~
            token!(Token::Semicolon) ~
            inc: expr ~
            token!(Token::RightParen) ~
            inner: statement,
            || { Statement::For(init, cond, inc, Box::new(inner)) }
        ) |
        chain!(token!(Token::While) ~ token!(Token::LeftParen) ~ cond: expr ~ token!(Token::RightParen) ~ inner: statement, || { Statement::While(cond, Box::new(inner)) }) |
        chain!(var: vardef ~ token!(Token::Semicolon), || { Statement::Var(var) }) |
        chain!(token!(Token::Return) ~ expression_statement: expr ~ token!(Token::Semicolon), || { Statement::Return(expression_statement) }) |
        chain!(expression_statement: expr ~ token!(Token::Semicolon), || { Statement::Expression(expression_statement) })
    )
}

fn structmember(input: &[Token]) -> IResult<&[Token], StructMember, ParseErrorReason> {
    chain!(input,
        typename: parse_typename ~
        varname: parse_variablename ~
        token!(Token::Semicolon),
        || { StructMember { name: varname, typename: typename } }
    )
}

fn structdefinition(input: &[Token]) -> IResult<&[Token], StructDefinition, ParseErrorReason> {
    chain!(input,
        token!(Token::Struct) ~
        structname: parse_variablename ~
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

fn constantvariable(input: &[Token]) -> IResult<&[Token], ConstantVariable, ParseErrorReason> {
    chain!(input,
        typename: parse_typename ~
        varname: parse_variablename ~
        token!(Token::Semicolon),
        || { ConstantVariable { name: varname, typename: typename, offset: None } }
    )
}

fn cbuffer_register(input: &[Token]) -> IResult<&[Token], ConstantSlot, ParseErrorReason> {
    map_res!(input, preceded!(token!(Token::Colon), token!(Token::Register(_))),
        |reg| { match reg {
            Token::Register(RegisterSlot::B(slot)) => Ok(ConstantSlot(slot)) as Result<ConstantSlot, ParseErrorReason>,
            Token::Register(_) => Err(ParseErrorReason::WrongSlotType),
            _ => unreachable!(),
        } }
    )
}

fn cbuffer(input: &[Token]) -> IResult<&[Token], ConstantBuffer, ParseErrorReason> {
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

fn globalvariable_register(input: &[Token]) -> IResult<&[Token], GlobalSlot, ParseErrorReason> {
    map_res!(input, preceded!(token!(Token::Colon), token!(Token::Register(_))),
        |reg| { match reg {
            Token::Register(RegisterSlot::T(slot)) => Ok(GlobalSlot::ReadSlot(slot)) as Result<GlobalSlot, ParseErrorReason>,
            Token::Register(RegisterSlot::U(slot)) => Ok(GlobalSlot::ReadWriteSlot(slot)) as Result<GlobalSlot, ParseErrorReason>,
            Token::Register(_) => Err(ParseErrorReason::WrongSlotType),
            _ => unreachable!(),
        } }
    )
}

fn globalvariable(input: &[Token]) -> IResult<&[Token], GlobalVariable, ParseErrorReason> {
    chain!(input,
        typename: parse_typename ~
        name: parse_variablename ~
        slot: opt!(globalvariable_register) ~
        token!(Token::Semicolon),
        || { GlobalVariable { name: name, typename: typename, slot: slot } }
    )
}

fn functionattribute(input: &[Token]) -> IResult<&[Token], FunctionAttribute, ParseErrorReason> {
    chain!(input,
        token!(Token::LeftSquareBracket) ~
        attr: alt!(
            chain!(
                map_res!(token!(Token::Id(_)), |tok| { if let Token::Id(Identifier(name)) = tok { match &name[..] { "numthreads" => Ok(name.clone()), _ => Err(()) } } else { Err(()) } }) ~
                token!(Token::LeftParen) ~
                x: token!(Token::LiteralInt(x) => x) ~
                token!(Token::Comma) ~
                y: token!(Token::LiteralInt(y) => y) ~
                token!(Token::Comma) ~
                z: token!(Token::LiteralInt(z) => z) ~
                token!(Token::RightParen),
                || { FunctionAttribute::NumThreads(x, y, z) }
            )
        ) ~
        token!(Token::RightSquareBracket),
        || { attr }
    )
}

fn functionparam(input: &[Token]) -> IResult<&[Token], FunctionParam, ParseErrorReason> {
    chain!(input,
        ty: parse_paramtype ~
        param: parse_variablename ~
        semantic: opt!(chain!(
            token!(Token::Colon) ~
            tok: map_res!(token!(Token::Id(_)), |tok| {
                if let Token::Id(Identifier(name)) = tok {
                    match &name[..] {
                        "SV_DispatchThreadID" => Ok(Semantic::DispatchThreadId),
                        _ => Err(())
                    }
                } else {
                    Err(())
                }
            }),
            || { tok }
        )),
        || { FunctionParam { name: param, param_type: ty, semantic: semantic } }
    )
}

fn functiondefinition(input: &[Token]) -> IResult<&[Token], FunctionDefinition, ParseErrorReason> {
    chain!(input,
        attributes: many0!(functionattribute) ~
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
        || { FunctionDefinition { name: func_name, returntype: ret, params: params, body: body, attributes: attributes } }
    )
}

fn rootdefinition(input: &[Token]) -> IResult<&[Token], RootDefinition, ParseErrorReason> {
    alt!(input,
        structdefinition => { |structdef| { RootDefinition::Struct(structdef) } } |
        cbuffer => { |cbuffer| { RootDefinition::ConstantBuffer(cbuffer) } } |
        globalvariable => { |globalvariable| { RootDefinition::GlobalVariable(globalvariable) } } |
        functiondefinition => { |funcdef| { RootDefinition::Function(funcdef) } }
    )
}

pub fn module(input: &[Token]) -> IResult<&[Token], Vec<RootDefinition>, ParseErrorReason> {
    chain!(input, roots: many0!(rootdefinition) ~ token!(Token::Eof), || { roots })
}

fn errorkind_to_reason(errkind: ErrorKind<ParseErrorReason>) -> ParseErrorReason {
    match errkind {
        ErrorKind::Custom(reason) => reason,
        _ => ParseErrorReason::Unknown,
    }
}

fn iresult_to_error(err: Err<&[Token], ParseErrorReason>) -> ParseError {
    match err {
        Err::Code(error) => ParseError(errorkind_to_reason(error), None, None),
        Err::Node(error, inner_err) => ParseError(errorkind_to_reason(error), None, Some(Box::new(iresult_to_error(*inner_err)))),
        Err::Position(error, position) => ParseError(errorkind_to_reason(error), Some(position.to_vec()), None),
        Err::NodePosition(error, position, inner_err) =>  ParseError(errorkind_to_reason(error), Some(position.to_vec()), Some(Box::new(iresult_to_error(*inner_err)))),
    }
}

pub fn parse(entry_point: String, source: &[Token]) -> Result<Module, ParseError> {
    let parse_result = module(source);
    match parse_result {
        IResult::Done(rest, _) if rest.len() != 0 => Err(ParseError(ParseErrorReason::FailedToParse, Some(rest.to_vec()), None)),
        IResult::Done(_, hlsl) => Ok(Module { entry_point: entry_point, root_definitions: hlsl }),
        IResult::Error(err) => Err(iresult_to_error(err)),
        IResult::Incomplete(_) => Err(ParseError(ParseErrorReason::UnexpectedEndOfStream, None, None)),
    }
}


#[cfg(test)]
fn exp_var(var_name: &'static str) -> Expression { Expression::Variable(var_name.to_string()) }
#[cfg(test)]
fn bexp_var(var_name: &'static str) -> Box<Expression> { Box::new(exp_var(var_name)) }

#[cfg(test)]
fn parse_result_from_str<T>(parse_func: Box<Fn(&[Token]) -> IResult<&[Token], T, ParseErrorReason>>) -> Box<Fn(&'static str) -> Result<T, ParseErrorReason>> where T: 'static {
    Box::new(move |string: &'static str| {
        let modified_string = string.to_string() + "\n";
        let input = &modified_string[..].as_bytes();
        let lex_result = super::lexer::lex(input);
        match lex_result {
            Ok(TokenStream(stream)) => {
                match parse_func(&stream[..]) {
                    IResult::Done(rem, exp) => {
                        if rem == &[Token::Eof] {
                            Ok(exp)
                        } else {
                            Err(ParseErrorReason::FailedToParse)
                        }
                    },
                    IResult::Incomplete(_) => Err(ParseErrorReason::UnexpectedEndOfStream),
                    _ => Err(ParseErrorReason::FailedToParse),
                }
            }
            Err(error) => panic!("Failed to lex `{:?}`", error)
        }
    })
}

#[cfg(test)]
fn parse_from_str<T>(parse_func: Box<Fn(&[Token]) -> IResult<&[Token], T, ParseErrorReason>>) -> Box<Fn(&'static str) -> T> where T: 'static {
    Box::new(move |string: &'static str| {
        let modified_string = string.to_string() + "\n";
        let input = &modified_string[..].as_bytes();
        let lex_result = super::lexer::lex(input);
        match lex_result {
            Ok(TokenStream(stream)) => {
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
            Err(error) => panic!("Failed to lex `{:?}`", error)
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
    let expr_str_fail = parse_result_from_str(Box::new(expr));

    assert_eq!(expr_str("a"), exp_var("a"));
    assert_eq!(expr_str("4"), Expression::Literal(Literal::UntypedInt(4)));
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

    assert_eq!(expr_str("a << b"), Expression::BinaryOperation(BinOp::LeftShift, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a >> b"), Expression::BinaryOperation(BinOp::RightShift, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a < b"), Expression::BinaryOperation(BinOp::LessThan, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a <= b"), Expression::BinaryOperation(BinOp::LessEqual, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a > b"), Expression::BinaryOperation(BinOp::GreaterThan, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a >= b"), Expression::BinaryOperation(BinOp::GreaterEqual, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a == b"), Expression::BinaryOperation(BinOp::Equality, bexp_var("a"), bexp_var("b")));
    assert_eq!(expr_str("a != b"), Expression::BinaryOperation(BinOp::Inequality, bexp_var("a"), bexp_var("b")));

    assert_eq!(expr_str_fail("a < < b"), Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a > > b"), Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a < = b"), Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a > = b"), Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a = = b"), Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a ! = b"), Err(ParseErrorReason::FailedToParse));

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
        Expression::Cast(Type::float(), bexp_var("b"))
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
        VarDef::new("x".to_string(), Type::uint(), None)
    );
    assert_eq!(condition_str("uint x"),
        Condition::Assignment(VarDef::new("x".to_string(), Type::uint(), None))
    );
    assert_eq!(condition_str("uint x = y"),
        Condition::Assignment(VarDef::new("x".to_string(), Type::uint(), Some(exp_var("y"))))
    );

    // Variable declarations
    assert_eq!(statement_str("uint x = y;"),
        Statement::Var(VarDef::new("x".to_string(), Type::uint(), Some(exp_var("y"))))
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
        Statement::If(exp_var("a"), Box::new(Statement::Expression(Expression::Call(bexp_var("func"), vec![]))))
    );
    assert_eq!(statement_str("if(a)func();"), statement_str("if (a) func(); "));
    assert_eq!(statement_str("if (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::If(exp_var("a"), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])))
    );

    // While loops
    assert_eq!(statement_str("while (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::While(exp_var("a"), Box::new(Statement::Block(vec![
            Statement::Expression(Expression::Call(bexp_var("one"), vec![])),
            Statement::Expression(Expression::Call(bexp_var("two"), vec![]))
        ])))
    );

    // For loops
    assert_eq!(statement_str("for(a;b;c)func();"),
        Statement::For(Condition::Expr(exp_var("a")), exp_var("b"), exp_var("c"), Box::new(
            Statement::Expression(Expression::Call(bexp_var("func"), vec![]))
        ))
    );
    assert_eq!(statement_str("for (uint i = 0; i; i++) { func(); }"),
        Statement::For(
            Condition::Assignment(VarDef::new("i".to_string(), Type::uint(), Some(Expression::Literal(Literal::UntypedInt(0))))),
            exp_var("i"),
            Expression::UnaryOperation(UnaryOp::PostfixIncrement, bexp_var("i")),
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
        name: "MyStruct".to_string(),
        members: vec![
            StructMember { name: "a".to_string(), typename: Type::uint() },
            StructMember { name: "b".to_string(), typename: Type::float() },
        ]
    };
    assert_eq!(structdefinition_str(test_struct_str), test_struct_ast.clone());
    assert_eq!(rootdefinition_str(test_struct_str), RootDefinition::Struct(test_struct_ast.clone()));

    let functiondefinition_str = parse_from_str(Box::new(functiondefinition));

    let test_func_str = "void func(float x) { }";
    let test_func_ast = FunctionDefinition { 
        name: "func".to_string(),
        returntype: Type::void(),
        params: vec![FunctionParam { name: "x".to_string(), param_type: Type::float().into(), semantic: None }],
        body: vec![],
        attributes: vec![],
    };
    assert_eq!(functiondefinition_str(test_func_str), test_func_ast.clone());
    assert_eq!(rootdefinition_str(test_func_str), RootDefinition::Function(test_func_ast.clone()));
    assert_eq!(rootdefinition_str("[numthreads(16, 16, 1)] void func(float x) { }"), RootDefinition::Function(FunctionDefinition {
        name: "func".to_string(),
        returntype: Type::void(),
        params: vec![FunctionParam { name: "x".to_string(), param_type: Type::float().into(), semantic: None }],
        body: vec![],
        attributes: vec![FunctionAttribute::NumThreads(16, 16, 1)],
    }));

    let constantvariable_str = parse_from_str(Box::new(constantvariable));

    let test_cbuffervar_str = "float4x4 wvp;";
    let test_cbuffervar_ast = ConstantVariable { name: "wvp".to_string(), typename: Type::float4x4(), offset: None };
    assert_eq!(constantvariable_str(test_cbuffervar_str), test_cbuffervar_ast.clone());

    let cbuffer_str = parse_from_str(Box::new(cbuffer));

    let test_cbuffer1_str = "cbuffer globals { float4x4 wvp; };";
    let test_cbuffer1_ast = ConstantBuffer {
        name: "globals".to_string(),
        slot: None,
        members: vec![
            ConstantVariable { name: "wvp".to_string(), typename: Type::float4x4(), offset: None },
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
            ConstantVariable { name: "wvp".to_string(), typename: Type::float4x4(), offset: None },
        ]
    };
    assert_eq!(cbuffer_str(test_cbuffer2_str), test_cbuffer2_ast.clone());
    assert_eq!(rootdefinition_str(test_cbuffer2_str), RootDefinition::ConstantBuffer(test_cbuffer2_ast.clone()));

    let globalvariable_str = parse_from_str(Box::new(globalvariable));

    let test_buffersrv_str = "Buffer g_myBuffer : register(t1);";
    let test_buffersrv_ast = GlobalVariable {
        name: "g_myBuffer".to_string(),
        typename: Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::Float, 4), TypeModifier::default()))),
        slot: Some(GlobalSlot::ReadSlot(1)),
    };
    assert_eq!(globalvariable_str(test_buffersrv_str), test_buffersrv_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv_str), RootDefinition::GlobalVariable(test_buffersrv_ast.clone()));

    let test_buffersrv2_str = "Buffer<uint4> g_myBuffer : register(t1);";
    let test_buffersrv2_ast = GlobalVariable {
        name: "g_myBuffer".to_string(),
        typename: Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::UInt, 4), TypeModifier::default()))),
        slot: Some(GlobalSlot::ReadSlot(1)),
    };
    assert_eq!(globalvariable_str(test_buffersrv2_str), test_buffersrv2_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv2_str), RootDefinition::GlobalVariable(test_buffersrv2_ast.clone()));

    let test_buffersrv3_str = "Buffer<vector<int, 4>> g_myBuffer : register(t1);";
    let test_buffersrv3_ast = GlobalVariable {
        name: "g_myBuffer".to_string(),
        typename: Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::Int, 4), TypeModifier::default()))),
        slot: Some(GlobalSlot::ReadSlot(1)),
    };
    assert_eq!(globalvariable_str(test_buffersrv3_str), test_buffersrv3_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv3_str), RootDefinition::GlobalVariable(test_buffersrv3_ast.clone()));

    let test_buffersrv4_str = "StructuredBuffer<CustomType> g_myBuffer : register(t1);";
    let test_buffersrv4_ast = GlobalVariable {
        name: "g_myBuffer".to_string(),
        typename: Type::from_object(ObjectType::StructuredBuffer(StructuredType(StructuredLayout::Custom("CustomType".to_string()), TypeModifier::default()))),
        slot: Some(GlobalSlot::ReadSlot(1)),
    };
    assert_eq!(globalvariable_str(test_buffersrv4_str), test_buffersrv4_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv4_str), RootDefinition::GlobalVariable(test_buffersrv4_ast.clone()));
}

