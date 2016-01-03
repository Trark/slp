use std::error;
use std::fmt;
use slp_shared::*;
use slp_lang_htk::*;
use slp_lang_hst::*;
use nom::{IResult, Needed, Err, ErrorKind};

#[derive(PartialEq, Debug, Clone)]
pub struct ParseError(pub ParseErrorReason, pub Option<Vec<LexToken>>, pub Option<Box<ParseError>>);

#[derive(PartialEq, Debug, Clone)]
pub enum ParseErrorReason {
    Unknown,
    UnexpectedEndOfStream,
    FailedToParse,
    WrongToken,
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
            let _: &[LexToken] = $i;
            let res: IResult<&[LexToken], LexToken, ParseErrorReason> = if $i.len() == 0 {
                IResult::Incomplete(Needed::Size(1))
            } else {
                match $i[0] {
                    LexToken($inp, _) => IResult::Done(&$i[1..], $i[0].clone()),
                    _ => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken), $i))
                } 
            };
            res
        }
    );
    ($i:expr, $inp: pat => $res: expr) => (
        {
            let _: &[LexToken] = $i;
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

fn parse_variablename(input: &[LexToken]) -> IResult<&[LexToken], Located<String>, ParseErrorReason> {
    map!(input, token!(Token::Id(_)), |tok| {
        match tok {
            LexToken(Token::Id(Identifier(name)), loc) => Located::new(name.clone(), loc),
            _ => unreachable!(),
        }
    })
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

// Parse data type as part of a string
fn parse_datalayout_str(typename: &str) -> Option<DataLayout> {

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
                                    IResult::Done(rest, y) => {
                                        if rest.len() == 0 {
                                            IResult::Done(&[], DataLayout::Matrix(ty, x, y))
                                        } else {
                                            IResult::Error(Err::Position(ErrorKind::Custom(0),
                                                                         input))
                                        }
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
        IResult::Done(rest, ty) => {
            assert_eq!(rest.len(), 0);
            Some(ty)
        }
        IResult::Incomplete(_) | IResult::Error(_) => None,
    }
}

fn parse_datalayout(input: &[LexToken]) -> IResult<&[LexToken], DataLayout, ParseErrorReason> {

    // Parse a vector dimension as a token
    fn parse_digit(input: &[LexToken]) -> IResult<&[LexToken], u32, ParseErrorReason> {
        token!(input, LexToken(Token::LiteralInt(i), _) => i as u32)
    }

    // Parse scalar type as a full token
    fn parse_scalartype(input: &[LexToken]) -> IResult<&[LexToken], ScalarType, ParseErrorReason> {
        if input.len() == 0 {
            IResult::Incomplete(Needed::Size(1))
        } else {
            match &input[0] {
                &LexToken(Token::Id(Identifier(ref name)), _) => {
                    match parse_scalartype_str(&name[..].as_bytes()) {
                        IResult::Done(rest, ty) => {
                            if rest.len() == 0 {
                                IResult::Done(&input[1..], ty)
                            } else {
                                IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input))
                            }
                        }
                        IResult::Incomplete(rem) => IResult::Incomplete(rem),
                        IResult::Error(_) => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input)),
                    }
                }
                _ => {
                    IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken),
                                                 input))
                }
            }
        }
    }

    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        match &input[0] {
            &LexToken(Token::Id(Identifier(ref name)), _) => {
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
                    }
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
                    }
                    _ => {
                        match parse_datalayout_str(&name[..]) {
                            Some(ty) => IResult::Done(&input[1..], ty),
                            None => IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input)),
                        }
                    }
                }
            }
            _ => {
                IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken),
                                             input))
            }
        }
    }
}

fn parse_datatype(input: &[LexToken]) -> IResult<&[LexToken], DataType, ParseErrorReason> {
    // Todo: Modifiers
    match parse_datalayout(input) {
        IResult::Done(rest, layout) => IResult::Done(rest, DataType(layout, Default::default())),
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_structuredlayout(input: &[LexToken])
                          -> IResult<&[LexToken], StructuredLayout, ParseErrorReason> {
    alt!(input,
        parse_datalayout => { |ty| { match ty {
                DataLayout::Scalar(scalar) => StructuredLayout::Scalar(scalar),
                DataLayout::Vector(scalar, x) => StructuredLayout::Vector(scalar, x),
                DataLayout::Matrix(scalar, x, y) => StructuredLayout::Matrix(scalar, x, y),
            }
        } } |
        token!(LexToken(Token::Id(Identifier(ref name)), _) => StructuredLayout::Custom(name.clone()))
    )
}

fn parse_structuredtype(input: &[LexToken]) -> IResult<&[LexToken], StructuredType, ParseErrorReason> {
    // Todo: Modifiers
    match parse_structuredlayout(input) {
        IResult::Done(rest, layout) => {
            IResult::Done(rest, StructuredType(layout, Default::default()))
        }
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_objecttype(input: &[LexToken]) -> IResult<&[LexToken], ObjectType, ParseErrorReason> {
    if input.len() == 0 {
        return IResult::Incomplete(Needed::Size(1));
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
        &LexToken(Token::Id(Identifier(ref name)), _) => {
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

                _ => return IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType), input)),
            }
        }
        _ => {
            return IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType),
                                                input))
        }
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

            let (buffer_arg, rest) = match delimited!(rest,
                                                      token!(Token::LeftAngleBracket(_)),
                                                      parse_datatype,
                                                      token!(Token::RightAngleBracket(_))) {
                IResult::Done(rest, ty) => (ty, rest),
                IResult::Incomplete(rem) => return IResult::Incomplete(rem),
                IResult::Error(_) => {
                    (DataType(DataLayout::Vector(ScalarType::Float, 4),
                              TypeModifier::default()),
                     rest)
                }
            };

            IResult::Done(rest,
                          match object_type {
                              ParseType::Buffer => ObjectType::Buffer(buffer_arg),
                              ParseType::RWBuffer => ObjectType::RWBuffer(buffer_arg),
                              ParseType::Texture1D => ObjectType::Texture1D(buffer_arg),
                              ParseType::Texture1DArray => ObjectType::Texture1DArray(buffer_arg),
                              ParseType::Texture2D => ObjectType::Texture2D(buffer_arg),
                              ParseType::Texture2DArray => ObjectType::Texture2DArray(buffer_arg),
                              ParseType::Texture2DMS => ObjectType::Texture2DMS(buffer_arg),
                              ParseType::Texture2DMSArray => {
                                  ObjectType::Texture2DMSArray(buffer_arg)
                              }
                              ParseType::Texture3D => ObjectType::Texture3D(buffer_arg),
                              ParseType::TextureCube => ObjectType::TextureCube(buffer_arg),
                              ParseType::TextureCubeArray => {
                                  ObjectType::TextureCubeArray(buffer_arg)
                              }
                              ParseType::RWTexture1D => ObjectType::RWTexture1D(buffer_arg),
                              ParseType::RWTexture1DArray => {
                                  ObjectType::RWTexture1DArray(buffer_arg)
                              }
                              ParseType::RWTexture2D => ObjectType::RWTexture2D(buffer_arg),
                              ParseType::RWTexture2DArray => {
                                  ObjectType::RWTexture2DArray(buffer_arg)
                              }
                              ParseType::RWTexture3D => ObjectType::RWTexture3D(buffer_arg),
                              _ => unreachable!(),
                          })
        }

        ParseType::StructuredBuffer |
        ParseType::RWStructuredBuffer |
        ParseType::AppendStructuredBuffer |
        ParseType::ConsumeStructuredBuffer => {

            let (buffer_arg, rest) = match delimited!(rest,
                                                      token!(Token::LeftAngleBracket(_)),
                                                      parse_structuredtype,
                                                      token!(Token::RightAngleBracket(_))) {
                IResult::Done(rest, ty) => (ty, rest),
                IResult::Incomplete(rem) => return IResult::Incomplete(rem),
                IResult::Error(_) => {
                    (StructuredType(StructuredLayout::Vector(ScalarType::Float, 4),
                                    TypeModifier::default()),
                     rest)
                }
            };

            IResult::Done(rest,
                          match object_type {
                              ParseType::StructuredBuffer => {
                                  ObjectType::StructuredBuffer(buffer_arg)
                              }
                              ParseType::RWStructuredBuffer => {
                                  ObjectType::RWStructuredBuffer(buffer_arg)
                              }
                              ParseType::AppendStructuredBuffer => {
                                  ObjectType::AppendStructuredBuffer(buffer_arg)
                              }
                              ParseType::ConsumeStructuredBuffer => {
                                  ObjectType::ConsumeStructuredBuffer(buffer_arg)
                              }
                              _ => unreachable!(),
                          })
        }

        ParseType::InputPatch => IResult::Done(rest, ObjectType::InputPatch),
        ParseType::OutputPatch => IResult::Done(rest, ObjectType::OutputPatch),
    }

}

fn parse_voidtype(input: &[LexToken]) -> IResult<&[LexToken], TypeLayout, ParseErrorReason> {
    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        match &input[0] {
            &LexToken(Token::Id(Identifier(ref name)), _) => {
                if name == "void" {
                    IResult::Done(&input[1..], TypeLayout::Void)
                } else {
                    IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::UnknownType),
                                                 input))
                }
            }
            _ => {
                IResult::Error(Err::Position(ErrorKind::Custom(ParseErrorReason::WrongToken),
                                             input))
            }
        }
    }
}

fn parse_typelayout(input: &[LexToken]) -> IResult<&[LexToken], TypeLayout, ParseErrorReason> {
    alt!(input,
        parse_objecttype => { |ty| { TypeLayout::Object(ty) } } |
        parse_voidtype |
        token!(LexToken(Token::SamplerState, _) => TypeLayout::SamplerState) |
        // Structured types eat everything as user defined types so must come last
        parse_structuredlayout => { |ty| { TypeLayout::from(ty) } }
    )
}

fn parse_typename(input: &[LexToken]) -> IResult<&[LexToken], Type, ParseErrorReason> {
    // Todo: modifiers that aren't const
    chain!(input,
        is_const: opt!(token!(Token::Const)) ~
        tl: parse_typelayout,
        || {
            Type(tl, TypeModifier { is_const: !is_const.is_none(), .. TypeModifier::default() })
        }
    )
}

fn parse_globaltype(input: &[LexToken]) -> IResult<&[LexToken], GlobalType, ParseErrorReason> {
    // Interpolation modifiers unimplemented
    // Non-standard combinations of storage classes unimplemented
    chain!(input,
        gs: opt!(alt!(
            token!(LexToken(Token::Static, _) => GlobalStorage::Static) |
            token!(LexToken(Token::GroupShared, _) => GlobalStorage::GroupShared) |
            token!(LexToken(Token::Extern, _) => GlobalStorage::Extern)
        )) ~
        ty: parse_typename,
        || {
            GlobalType(ty, gs.unwrap_or(GlobalStorage::default()), None)
        }
    )
}

fn parse_inputmodifier(input: &[LexToken]) -> IResult<&[LexToken], InputModifier, ParseErrorReason> {
    alt!(input,
        token!(Token::In) => { |_| InputModifier::In } |
        token!(Token::Out) => { |_| InputModifier::Out } |
        token!(Token::InOut) => { |_| InputModifier::InOut }
    )
}

fn parse_paramtype(input: &[LexToken]) -> IResult<&[LexToken], ParamType, ParseErrorReason> {
    let (input, it) = match parse_inputmodifier(input) {
        IResult::Done(rest, it) => (rest, it),
        IResult::Incomplete(i) => return IResult::Incomplete(i),
        IResult::Error(_) => (input, InputModifier::default()),
    };
    // Todo: interpolation modifiers
    match parse_typename(input) {
        IResult::Done(rest, ty) => IResult::Done(rest, ParamType(ty, it, None)),
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_localtype(input: &[LexToken]) -> IResult<&[LexToken], LocalType, ParseErrorReason> {
    // Todo: input modifiers
    match parse_typename(input) {
        IResult::Done(rest, ty) => {
            IResult::Done(rest, LocalType(ty, LocalStorage::default(), None))
        }
        IResult::Incomplete(i) => IResult::Incomplete(i),
        IResult::Error(err) => IResult::Error(err),
    }
}

fn parse_arraydim(input: &[LexToken])
                  -> IResult<&[LexToken], Option<Located<Expression>>, ParseErrorReason> {
    chain!(input,
        token!(Token::LeftSquareBracket) ~
        constant_expression: opt!(expr) ~
        token!(Token::RightSquareBracket),
        || { constant_expression }
    )
}

fn expr_paren(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    alt!(input,
        chain!(start: token!(Token::LeftParen) ~ expr: expr ~ token!(Token::RightParen), || { Located::new(expr.to_node(), start.to_loc()) }) |
        parse_variablename => { |name: Located<String>| { Located::new(Expression::Variable(name.node), name.location) } } |
        token!(LexToken(Token::LiteralInt(i), ref loc) => Located::new(Expression::Literal(Literal::UntypedInt(i)), loc.clone())) |
        token!(LexToken(Token::LiteralUInt(i), ref loc) => Located::new(Expression::Literal(Literal::UInt(i)), loc.clone())) |
        token!(LexToken(Token::LiteralLong(i), ref loc) => Located::new(Expression::Literal(Literal::Long(i)), loc.clone())) |
        token!(LexToken(Token::LiteralHalf(i), ref loc) => Located::new(Expression::Literal(Literal::Half(i)), loc.clone())) |
        token!(LexToken(Token::LiteralFloat(i), ref loc) => Located::new(Expression::Literal(Literal::Float(i)), loc.clone())) |
        token!(LexToken(Token::LiteralDouble(i), ref loc) => Located::new(Expression::Literal(Literal::Double(i)), loc.clone())) |
        token!(LexToken(Token::True, ref loc) => Located::new(Expression::Literal(Literal::Bool(true)), loc.clone())) |
        token!(LexToken(Token::False, ref loc) => Located::new(Expression::Literal(Literal::Bool(false)), loc.clone()))
    )
}

fn expr_p1(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    #[derive(Clone)]
    enum Precedence1Postfix {
        Increment,
        Decrement,
        Call(Vec<Located<Expression>>),
        ArraySubscript(Located<Expression>),
        Member(String),
    }

    fn expr_p1_right(input: &[LexToken])
                     -> IResult<&[LexToken], Located<Precedence1Postfix>, ParseErrorReason> {
        chain!(input,
            right: alt!(
                chain!(start: token!(Token::Plus) ~ token!(Token::Plus), || { Located::new(Precedence1Postfix::Increment, start.to_loc()) }) |
                chain!(start: token!(Token::Minus) ~ token!(Token::Minus), || { Located::new(Precedence1Postfix::Decrement, start.to_loc()) }) |
                chain!(
                    start: token!(Token::LeftParen) ~
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
                    || { Located::new(Precedence1Postfix::Call(match params.clone() { Some(v) => v, None => Vec::new() }), start.to_loc()) }
                ) |
                chain!(
                    token!(Token::Period) ~
                    member: parse_variablename,
                    || { Located::new(Precedence1Postfix::Member(member.node.clone()), member.location.clone()) }
                ) |
                chain!(
                    start: token!(Token::LeftSquareBracket) ~
                    subscript: expr ~
                    token!(Token::RightSquareBracket),
                    || { Located::new(Precedence1Postfix::ArraySubscript(subscript), start.to_loc()) }
                )
            ),
            || { right }
        )
    }

    chain!(input,
        left: expr_paren ~
        rights: many0!(expr_p1_right),
        || {
            let loc = left.location.clone();
            let mut final_expression = left;
            for val in rights.iter() {
                final_expression = Located::new(match val.node.clone() {
                    Precedence1Postfix::Increment => Expression::UnaryOperation(UnaryOp::PostfixIncrement, Box::new(final_expression)),
                    Precedence1Postfix::Decrement => Expression::UnaryOperation(UnaryOp::PostfixDecrement, Box::new(final_expression)),
                    Precedence1Postfix::Call(params) => {
                        let ty_opt = match *final_expression {
                            Expression::Variable(ref name) => parse_datalayout_str(name),
                            _ => None,
                        };
                        match ty_opt {
                            Some(dty) => Expression::NumericConstructor(dty, params),
                            None => Expression::Call(Box::new(final_expression), params),
                        }
                    },
                    Precedence1Postfix::ArraySubscript(expr) => Expression::ArraySubscript(Box::new(final_expression), Box::new(expr)),
                    Precedence1Postfix::Member(name) => Expression::Member(Box::new(final_expression), name),
                }, loc.clone())
            }
            final_expression
        }
    )
}

fn unaryop_prefix(input: &[LexToken]) -> IResult<&[LexToken], Located<UnaryOp>, ParseErrorReason> {
    alt!(input,
        chain!(start: token!(Token::Plus) ~ token!(Token::Plus), || { Located::new(UnaryOp::PrefixIncrement, start.to_loc()) }) |
        chain!(start: token!(Token::Minus) ~ token!(Token::Minus), || { Located::new(UnaryOp::PrefixDecrement, start.to_loc()) }) |
        token!(Token::Plus) => { |s: LexToken| Located::new(UnaryOp::Plus, s.to_loc()) } |
        token!(Token::Minus) => { |s: LexToken| Located::new(UnaryOp::Minus, s.to_loc()) } |
        token!(Token::ExclamationPoint) => { |s: LexToken| Located::new(UnaryOp::LogicalNot, s.to_loc()) } |
        token!(Token::Tilde) => { |s: LexToken| Located::new(UnaryOp::BitwiseNot, s.to_loc()) }
    )
}

fn expr_p2(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    alt!(input,
        chain!(unary: unaryop_prefix ~ expr: expr_p2, || { Located::new(Expression::UnaryOperation(unary.node.clone(), Box::new(expr)), unary.location.clone()) }) |
        chain!(start: token!(Token::LeftParen) ~ cast: parse_typename ~ token!(Token::RightParen) ~ expr: expr_p2, || { Located::new(Expression::Cast(cast, Box::new(expr)), start.to_loc()) }) |
        expr_p1
    )
}

fn combine_rights(left: Located<Expression>,
                  rights: Vec<(BinOp, Located<Expression>)>)
                  -> Located<Expression> {
    let loc = left.location.clone();
    let mut final_expression = left;
    for val in rights.iter() {
        let (ref op, ref exp) = *val;
        final_expression = Located::new(Expression::BinaryOperation(op.clone(),
                                                                    Box::new(final_expression),
                                                                    Box::new(exp.clone())),
                                        loc.clone())
    }
    final_expression
}

fn expr_p3(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    fn binop_p3(input: &[LexToken]) -> IResult<&[LexToken], BinOp, ParseErrorReason> {
        alt!(input,
            token!(Token::Asterix) => { |_| BinOp::Multiply } |
            token!(Token::ForwardSlash) => { |_| BinOp::Divide } |
            token!(Token::Percent) => { |_| BinOp::Modulus }
        )
    }

    fn expr_p3_right(input: &[LexToken])
                     -> IResult<&[LexToken], (BinOp, Located<Expression>), ParseErrorReason> {
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

fn expr_p4(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    fn binop_p4(input: &[LexToken]) -> IResult<&[LexToken], BinOp, ParseErrorReason> {
        alt!(input,
            token!(Token::Plus) => { |_| BinOp::Add } |
            token!(Token::Minus) => { |_| BinOp::Subtract }
        )
    }

    fn expr_p4_right(input: &[LexToken])
                     -> IResult<&[LexToken], (BinOp, Located<Expression>), ParseErrorReason> {
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

fn expr_p5(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    fn parse_op(input: &[LexToken]) -> IResult<&[LexToken], BinOp, ParseErrorReason> {
        alt!(input,
            chain!(token!(Token::LeftAngleBracket(FollowedBy::Token)) ~ token!(Token::LeftAngleBracket(_)), || { BinOp::LeftShift }) |
            chain!(token!(Token::RightAngleBracket(FollowedBy::Token)) ~ token!(Token::RightAngleBracket(_)), || { BinOp::RightShift })
        )
    }

    fn parse_rights(input: &[LexToken])
                    -> IResult<&[LexToken], (BinOp, Located<Expression>), ParseErrorReason> {
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

fn expr_p6(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    fn parse_op(input: &[LexToken]) -> IResult<&[LexToken], BinOp, ParseErrorReason> {
        alt!(input,
            chain!(token!(Token::LeftAngleBracket(FollowedBy::Token)) ~ token!(Token::Equals), || { BinOp::LessEqual }) |
            chain!(token!(Token::RightAngleBracket(FollowedBy::Token)) ~ token!(Token::Equals), || { BinOp::GreaterEqual }) |
            token!(Token::LeftAngleBracket(_)) => { |_| BinOp::LessThan } |
            token!(Token::RightAngleBracket(_)) => { |_| BinOp::GreaterThan }
        )
    }

    fn parse_rights(input: &[LexToken])
                    -> IResult<&[LexToken], (BinOp, Located<Expression>), ParseErrorReason> {
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

fn expr_p7(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    fn parse_op(input: &[LexToken]) -> IResult<&[LexToken], BinOp, ParseErrorReason> {
        alt!(input,
            token!(Token::DoubleEquals) => { |_| BinOp::Equality } |
            token!(Token::ExclamationEquals) => { |_| BinOp::Inequality }
        )
    }

    fn parse_rights(input: &[LexToken])
                    -> IResult<&[LexToken], (BinOp, Located<Expression>), ParseErrorReason> {
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

fn expr_p8(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    chain!(input,
        left: expr_p7 ~
        rights: many0!(chain!(
            op: token!(LexToken(Token::Ampersand(_), _) => BinOp::BitwiseAnd) ~
            right: expr_p7,
            || (op, right)
        )),
        || { combine_rights(left, rights) }
    )
}

fn expr_p9(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    chain!(input,
        left: expr_p8 ~
        rights: many0!(chain!(
            op: token!(LexToken(Token::Hat, _) => BinOp::BitwiseXor) ~
            right: expr_p8,
            || (op, right)
        )),
        || { combine_rights(left, rights) }
    )
}

fn expr_p10(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    chain!(input,
        left: expr_p9 ~
        rights: many0!(chain!(
            op: token!(LexToken(Token::VerticalBar(_), _) => BinOp::BitwiseOr) ~
            right: expr_p9,
            || (op, right)
        )),
        || { combine_rights(left, rights) }
    )
}

fn expr_p11(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    chain!(input,
        left: expr_p10 ~
        rights: many0!(chain!(
            op: chain!(token!(Token::Ampersand(FollowedBy::Token)) ~ token!(Token::Ampersand(_)), || BinOp::BooleanAnd) ~
            right: expr_p10,
            || (op, right)
        )),
        || { combine_rights(left, rights) }
    )
}

fn expr_p12(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    chain!(input,
        left: expr_p11 ~
        rights: many0!(chain!(
            op: chain!(token!(Token::VerticalBar(FollowedBy::Token)) ~ token!(Token::VerticalBar(_)), || BinOp::BooleanOr) ~
            right: expr_p11,
            || (op, right)
        )),
        || { combine_rights(left, rights) }
    )
}

fn expr_p13(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    chain!(input,
        main: expr_p12 ~
        opt: opt!(chain!(
            token!(Token::QuestionMark) ~
            left: expr_p13 ~
            token!(Token::Colon) ~
            right: expr_p13,
            || (left, right)
        )),
        || {
            match opt.clone() {
                Some((left, right)) => {
                    let loc = main.location.clone();
                    Located::new(Expression::TernaryConditional(Box::new(main), Box::new(left), Box::new(right)), loc)
                }
                None => main,
            }
        }
    )
}

fn expr_p14(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {

    fn op_p14(input: &[LexToken]) -> IResult<&[LexToken], BinOp, ParseErrorReason> {
        alt!(input,
             token!(LexToken(Token::Equals, _) => BinOp::Assignment) |
             chain!(token!(Token::Plus) ~ token!(Token::Equals), || BinOp::SumAssignment) |
             chain!(token!(Token::Minus) ~ token!(Token::Equals), || BinOp::DifferenceAssignment) |
             chain!(token!(Token::Asterix) ~ token!(Token::Equals), || BinOp::ProductAssignment) |
             chain!(token!(Token::ForwardSlash) ~ token!(Token::Equals), || BinOp::QuotientAssignment) |
             chain!(token!(Token::Percent) ~ token!(Token::Equals), || BinOp::RemainderAssignment)
        )
    }

    chain!(input,
        lhs: expr_p13 ~
        opt: opt!(chain!(op: op_p14 ~ rhs: expr_p14, || (op, rhs))),
        || {
            match opt.clone() {
                Some((op, rhs)) => {
                    let loc = lhs.location.clone();
                    Located::new(Expression::BinaryOperation(op, Box::new(lhs), Box::new(rhs)), loc)
                }
                None => lhs,
            }
        }
    )
}

fn expr(input: &[LexToken]) -> IResult<&[LexToken], Located<Expression>, ParseErrorReason> {
    expr_p14(input)
}

fn initializer(input: &[LexToken]) -> IResult<&[LexToken], Option<Initializer>, ParseErrorReason> {

    fn init_expr(input: &[LexToken]) -> IResult<&[LexToken], Initializer, ParseErrorReason> {
        map!(input, expr, |expr| Initializer::Expression(expr))
    }

    fn init_aggregate(input: &[LexToken]) -> IResult<&[LexToken], Initializer, ParseErrorReason> {
        map!(input,
             delimited!(token!(Token::LeftBrace),
                        separated_nonempty_list!(token!(Token::Comma), init_any),
                        token!(Token::RightBrace)),
             |exprs| Initializer::Aggregate(exprs))
    }

    fn init_any(input: &[LexToken]) -> IResult<&[LexToken], Initializer, ParseErrorReason> {
        alt!(input, init_expr | init_aggregate)
    }

    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        match input[0].0 {
            Token::Equals => map!(&input[1..], init_any, |init| Some(init)),
            _ => IResult::Done(input, None),
        }
    }
}

#[test]
fn test_initializer() {
    assert_eq!(initializer(&[]), IResult::Incomplete(Needed::Size(1)));

    // Semicolon to trigger parsing to end
    let semicolon = LexToken::with_no_loc(Token::Semicolon);
    let done_toks = &[semicolon.clone()][..];

    // No initializer tests
    assert_eq!(initializer(&[semicolon.clone()]),
               IResult::Done(done_toks, None));

    // Expression initialization tests
    // = [expr]
    let expr_lit = [LexToken::with_no_loc(Token::Equals),
                    LexToken::with_no_loc(Token::LiteralInt(4)),
                    semicolon.clone()];
    let hst_lit = Located::none(Expression::Literal(Literal::UntypedInt(4)));
    assert_eq!(initializer(&expr_lit),
               IResult::Done(done_toks, Some(Initializer::Expression(hst_lit))));

    // Aggregate initialization tests
    // = { [expr], [expr], [expr] }
    fn loc_lit(i: u64) -> Initializer {
        Initializer::Expression(Located::none(Expression::Literal(Literal::UntypedInt(i))))
    }

    let aggr_1 = [LexToken::with_no_loc(Token::Equals),
                  LexToken::with_no_loc(Token::LeftBrace),
                  LexToken::with_no_loc(Token::LiteralInt(4)),
                  LexToken::with_no_loc(Token::RightBrace),
                  semicolon.clone()];
    let aggr_1_lit = loc_lit(4);
    assert_eq!(initializer(&aggr_1),
               IResult::Done(done_toks, Some(Initializer::Aggregate(vec![aggr_1_lit]))));

    let aggr_3 = [LexToken::with_no_loc(Token::Equals),
                  LexToken::with_no_loc(Token::LeftBrace),
                  LexToken::with_no_loc(Token::LiteralInt(4)),
                  LexToken::with_no_loc(Token::Comma),
                  LexToken::with_no_loc(Token::LiteralInt(2)),
                  LexToken::with_no_loc(Token::Comma),
                  LexToken::with_no_loc(Token::LiteralInt(1)),
                  LexToken::with_no_loc(Token::RightBrace),
                  semicolon.clone()];
    let aggr_3_lits = vec![loc_lit(4), loc_lit(2), loc_lit(1)];
    assert_eq!(initializer(&aggr_3),
               IResult::Done(done_toks, Some(Initializer::Aggregate(aggr_3_lits))));

    // = {} should fail
    let aggr_0 = [LexToken::with_no_loc(Token::Equals),
                  LexToken::with_no_loc(Token::LeftBrace),
                  LexToken::with_no_loc(Token::RightBrace),
                  semicolon.clone()];
    assert!(match initializer(&aggr_0) {
        IResult::Error(_) => true,
        _ => false,
    });
}

fn vardef(input: &[LexToken]) -> IResult<&[LexToken], VarDef, ParseErrorReason> {
    chain!(input,
        typename: parse_localtype ~
        defs: map_res!(separated_list!(
            token!(Token::Comma),
            chain!(
                varname: parse_variablename ~
                array_dim: opt!(parse_arraydim) ~
                init: initializer,
                || LocalVariableName {
                    name: varname.to_node(),
                    bind: match array_dim { Some(ref expr) => VariableBind::Array(expr.clone()), None => VariableBind::Normal },
                    init: init
                }
            )
        ), |res: Vec<_>| if res.len() > 0 { Ok(res) } else { Err(()) }),
        || {
            VarDef {
                local_type: typename,
                defs: defs,
            }
        }
    )
}

fn init_statement(input: &[LexToken]) -> IResult<&[LexToken], InitStatement, ParseErrorReason> {
    alt!(input,
        vardef => { |variable_definition| InitStatement::Declaration(variable_definition) } |
        expr => { |expression| InitStatement::Expression(expression) }
    )
}

fn statement_attribute(input: &[LexToken]) -> IResult<&[LexToken], (), ParseErrorReason> {
    chain!(input,
        token!(Token::LeftSquareBracket) ~
        token!(Token::Id(_)) ~
        opt!(chain!(
            token!(Token::LeftParen) ~
            separated_nonempty_list!(token!(Token::Comma), token!(Token::Id(_))) ~
            token!(Token::RightParen),
            || ()
        )) ~
        token!(Token::RightSquareBracket),
        || ()
    )
}

#[test]
fn test_statement_attribute() {
    let fastopt = &[LexToken::with_no_loc(Token::LeftSquareBracket),
                    LexToken::with_no_loc(Token::Id(Identifier("fastopt".to_string()))),
                    LexToken::with_no_loc(Token::RightSquareBracket)];
    assert_eq!(statement_attribute(fastopt), IResult::Done(&[][..], ()));
}

fn statement(input: &[LexToken]) -> IResult<&[LexToken], Statement, ParseErrorReason> {
    // Parse and ignore attributes before a statement
    let input = match many0!(input, statement_attribute) {
        IResult::Done(rest, _) => rest,
        IResult::Incomplete(rem) => return IResult::Incomplete(rem),
        IResult::Error(err) => return IResult::Error(err),
    };
    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        let (head, tail) = (input[0].clone(), &input[1..]);
        match head {
            LexToken(Token::Semicolon, _) => IResult::Done(tail, Statement::Empty),
            LexToken(Token::If, _) => {
                let if_part = chain!(tail,
                    token!(Token::LeftParen) ~
                    cond: expr ~
                    token!(Token::RightParen) ~
                    inner_statement: statement,
                    || Statement::If(cond, Box::new(inner_statement))
                );
                match if_part {
                    IResult::Incomplete(rem) => IResult::Incomplete(rem),
                    IResult::Error(err) => IResult::Error(err),
                    IResult::Done(rest, Statement::If(cond, first)) => {
                        if input.len() == 0 {
                            IResult::Incomplete(Needed::Size(1))
                        } else {
                            let (head, tail) = (rest[0].clone(), &rest[1..]);
                            match head {
                                LexToken(Token::Else, _) => {
                                    match statement(tail) {
                                        IResult::Incomplete(rem) => IResult::Incomplete(rem),
                                        IResult::Error(err) => IResult::Error(err),
                                        IResult::Done(rest, else_part) => {
                                            let s = Statement::IfElse(cond,
                                                                      first,
                                                                      Box::new(else_part));
                                            IResult::Done(rest, s)
                                        }
                                    }
                                }
                                _ => IResult::Done(rest, Statement::If(cond, first)),
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            LexToken(Token::For, _) => {
                chain!(tail,
                    token!(Token::LeftParen) ~
                    init: init_statement ~
                    token!(Token::Semicolon) ~
                    cond: expr ~
                    token!(Token::Semicolon) ~
                    inc: expr ~
                    token!(Token::RightParen) ~
                    inner: statement,
                    || Statement::For(init, cond, inc, Box::new(inner))
                )
            }
            LexToken(Token::While, _) => {
                chain!(tail,
                    token!(Token::LeftParen) ~
                    cond: expr ~
                    token!(Token::RightParen) ~
                    inner: statement,
                    || Statement::While(cond, Box::new(inner))
                )
            }
            LexToken(Token::Return, _) => {
                chain!(tail,
                    expression_statement: expr ~
                    token!(Token::Semicolon),
                    || Statement::Return(expression_statement)
                )
            }
            LexToken(Token::LeftBrace, _) => map!(input, statement_block, |s| Statement::Block(s)),
            _ => {
                // Try parsing a variable definition
                let err = match chain!(input, var: vardef ~ token!(Token::Semicolon), || Statement::Var(var)) {
                    IResult::Done(rest, statement) => return IResult::Done(rest, statement),
                    IResult::Incomplete(rem) => return IResult::Incomplete(rem),
                    IResult::Error(e) => e,
                };
                // Try parsing an expression statement
                let err = match chain!(input, expression_statement: expr ~ token!(Token::Semicolon), || Statement::Expression(expression_statement)) {
                    IResult::Done(rest, statement) => return IResult::Done(rest, statement),
                    IResult::Incomplete(rem) => return IResult::Incomplete(rem),
                    IResult::Error(e) => get_most_relevant_error(err, e),
                };
                // Return the most likely error
                IResult::Error(err)
            }
        }
    }
}

fn statement_block(input: &[LexToken]) -> IResult<&[LexToken], Vec<Statement>, ParseErrorReason> {
    let mut statements = Vec::new();
    let mut rest = match token!(input, Token::LeftBrace) {
        IResult::Done(rest, _) => rest,
        IResult::Incomplete(rem) => return IResult::Incomplete(rem),
        IResult::Error(err) => return IResult::Error(err),
    };
    loop {
        let last_def = statement(rest);
        if let IResult::Done(remaining, root) = last_def {
            statements.push(root);
            rest = remaining;
        } else {
            return match token!(rest, Token::RightBrace) {
                IResult::Done(rest, _) => IResult::Done(rest, statements),
                IResult::Incomplete(rem) => IResult::Incomplete(rem),
                IResult::Error(_) => {
                    match last_def {
                        IResult::Done(_, _) => unreachable!(),
                        IResult::Incomplete(rem) => IResult::Incomplete(rem),
                        IResult::Error(err) => IResult::Error(err),
                    }
                }
            };
        }
    }
}

fn structmembername(input: &[LexToken]) -> IResult<&[LexToken], StructMemberName, ParseErrorReason> {
    chain!(input,
        name: parse_variablename ~
        array_dim: opt!(parse_arraydim),
        || StructMemberName {
            name: name.to_node(),
            bind: match array_dim { Some(ref expr) => VariableBind::Array(expr.clone()), None => VariableBind::Normal },
        }
    )
}

fn structmember(input: &[LexToken]) -> IResult<&[LexToken], StructMember, ParseErrorReason> {
    chain!(input,
        typename: parse_typename ~
        defs: separated_nonempty_list!(token!(Token::Comma), structmembername) ~
        token!(Token::Semicolon),
        || StructMember { ty: typename, defs: defs }
    )
}

fn structdefinition(input: &[LexToken]) -> IResult<&[LexToken], StructDefinition, ParseErrorReason> {
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
        || { StructDefinition { name: structname.to_node(), members: members } }
    )
}

fn constantvariablename(input: &[LexToken])
                        -> IResult<&[LexToken], ConstantVariableName, ParseErrorReason> {
    chain!(input,
        name: parse_variablename ~
        array_dim: opt!(parse_arraydim),
        || { ConstantVariableName {
            name: name.to_node(),
            bind: match array_dim { Some(ref expr) => VariableBind::Array(expr.clone()), None => VariableBind::Normal },
            offset: None,
        } }
    )
}

fn constantvariable(input: &[LexToken]) -> IResult<&[LexToken], ConstantVariable, ParseErrorReason> {
    chain!(input,
        typename: parse_typename ~
        defs: separated_nonempty_list!(token!(Token::Comma), constantvariablename) ~
        token!(Token::Semicolon),
        || { ConstantVariable { ty: typename, defs: defs } }
    )
}

fn cbuffer_register(input: &[LexToken]) -> IResult<&[LexToken], ConstantSlot, ParseErrorReason> {
    map_res!(input,
             preceded!(token!(Token::Colon), token!(Token::Register(_))),
             |reg| {
                 match reg {
                     LexToken(Token::Register(RegisterSlot::B(slot)), _) => {
                         Ok(ConstantSlot(slot)) as Result<ConstantSlot, ParseErrorReason>
                     }
                     LexToken(Token::Register(_), _) => Err(ParseErrorReason::WrongSlotType),
                     _ => unreachable!(),
                 }
             })
}

fn cbuffer(input: &[LexToken]) -> IResult<&[LexToken], ConstantBuffer, ParseErrorReason> {
    chain!(input,
        token!(Token::ConstantBuffer) ~
        name: parse_variablename ~
        slot: opt!(cbuffer_register) ~
        members: delimited!(
            token!(Token::LeftBrace),
            many0!(constantvariable),
            token!(Token::RightBrace)
        ),
        || { ConstantBuffer { name: name.to_node(), slot: slot, members: members } }
    )
}

fn globalvariable_register(input: &[LexToken]) -> IResult<&[LexToken], GlobalSlot, ParseErrorReason> {
    map_res!(input,
             preceded!(token!(Token::Colon), token!(Token::Register(_))),
             |reg| {
                 match reg {
                     LexToken(Token::Register(RegisterSlot::T(slot)), _) => {
                         Ok(GlobalSlot::ReadSlot(slot)) as Result<GlobalSlot, ParseErrorReason>
                     }
                     LexToken(Token::Register(RegisterSlot::U(slot)), _) => {
                         Ok(GlobalSlot::ReadWriteSlot(slot)) as Result<GlobalSlot, ParseErrorReason>
                     }
                     LexToken(Token::Register(_), _) => Err(ParseErrorReason::WrongSlotType),
                     _ => unreachable!(),
                 }
             })
}

fn globalvariablename(input: &[LexToken])
                      -> IResult<&[LexToken], GlobalVariableName, ParseErrorReason> {
    chain!(input,
        name: parse_variablename ~
        array_dim: opt!(parse_arraydim) ~
        slot: opt!(globalvariable_register) ~
        init: initializer,
        || { GlobalVariableName {
            name: name.to_node(),
            bind: match array_dim { Some(ref expr) => VariableBind::Array(expr.clone()), None => VariableBind::Normal },
            slot: slot,
            init: init
        } }
    )
}

fn globalvariable(input: &[LexToken]) -> IResult<&[LexToken], GlobalVariable, ParseErrorReason> {
    chain!(input,
        typename: parse_globaltype ~
        defs: separated_nonempty_list!(token!(Token::Comma), globalvariablename) ~
        token!(Token::Semicolon),
        || { GlobalVariable { global_type: typename, defs: defs } }
    )
}

fn functionattribute(input: &[LexToken]) -> IResult<&[LexToken], FunctionAttribute, ParseErrorReason> {
    chain!(input,
        token!(Token::LeftSquareBracket) ~
        attr: alt!(
            chain!(
                map_res!(token!(Token::Id(_)), |tok| {
                    if let LexToken(Token::Id(Identifier(name)), _) = tok {
                        match &name[..] {
                            "numthreads" => Ok(name.clone()),
                            _ => Err(())
                        }
                    } else {
                        Err(())
                    }
                }) ~
                token!(Token::LeftParen) ~
                x: expr ~
                token!(Token::Comma) ~
                y: expr ~
                token!(Token::Comma) ~
                z: expr ~
                token!(Token::RightParen),
                || { FunctionAttribute::NumThreads(x, y, z) }
            )
        ) ~
        token!(Token::RightSquareBracket),
        || { attr }
    )
}

fn functionparam(input: &[LexToken]) -> IResult<&[LexToken], FunctionParam, ParseErrorReason> {
    chain!(input,
        ty: parse_paramtype ~
        param: parse_variablename ~
        semantic: opt!(chain!(
            token!(Token::Colon) ~
            tok: map_res!(token!(Token::Id(_)), |tok| {
                if let LexToken(Token::Id(Identifier(name)), _) = tok {
                    match &name[..] {
                        "SV_DispatchThreadID" => Ok(Semantic::DispatchThreadId),
                        "SV_GroupID" => Ok(Semantic::GroupId),
                        "SV_GroupIndex" => Ok(Semantic::GroupIndex),
                        "SV_GroupThreadID" => Ok(Semantic::GroupThreadId),
                        _ => Err(())
                    }
                } else {
                    Err(())
                }
            }),
            || { tok }
        )),
        || { FunctionParam { name: param.to_node(), param_type: ty, semantic: semantic } }
    )
}

fn functiondefinition(input: &[LexToken])
                      -> IResult<&[LexToken], FunctionDefinition, ParseErrorReason> {
    chain!(input,
        attributes: many0!(functionattribute) ~
        ret: parse_typename ~
        func_name: parse_variablename ~
        params: delimited!(
            token!(Token::LeftParen),
            separated_list!(token!(Token::Comma), functionparam),
            token!(Token::RightParen)
        ) ~
        body: statement_block,
        || { FunctionDefinition { name: func_name.to_node(), returntype: ret, params: params, body: body, attributes: attributes } }
    )
}

fn rootdefinition(input: &[LexToken]) -> IResult<&[LexToken], RootDefinition, ParseErrorReason> {

    let err = match structdefinition(input) {
        IResult::Done(rest, structdef) => {
            return IResult::Done(rest, RootDefinition::Struct(structdef))
        }
        IResult::Incomplete(rem) => return IResult::Incomplete(rem),
        IResult::Error(e) => e,
    };

    let err = match cbuffer(input) {
        IResult::Done(rest, cbuffer) => {
            return IResult::Done(rest, RootDefinition::ConstantBuffer(cbuffer))
        }
        IResult::Incomplete(rem) => return IResult::Incomplete(rem),
        IResult::Error(e) => get_most_relevant_error(err, e),
    };

    let err = match globalvariable(input) {
        IResult::Done(rest, globalvariable) => {
            return IResult::Done(rest, RootDefinition::GlobalVariable(globalvariable))
        }
        IResult::Incomplete(rem) => return IResult::Incomplete(rem),
        IResult::Error(e) => get_most_relevant_error(err, e),
    };

    let err = match functiondefinition(input) {
        IResult::Done(rest, funcdef) => {
            return IResult::Done(rest, RootDefinition::Function(funcdef))
        }
        IResult::Incomplete(rem) => return IResult::Incomplete(rem),
        IResult::Error(e) => get_most_relevant_error(err, e),
    };

    IResult::Error(err)
}

fn rootdefinition_with_semicolon(input: &[LexToken])
                                 -> IResult<&[LexToken], RootDefinition, ParseErrorReason> {
    terminated!(input, rootdefinition, many0!(token!(Token::Semicolon)))
}

// Find the error with the longest tokens used
fn get_most_relevant_error<'a: 'c, 'b: 'c, 'c>(lhs: Err<&'a [LexToken], ParseErrorReason>,
                                               rhs: Err<&'b [LexToken], ParseErrorReason>)
                                               -> Err<&'c [LexToken], ParseErrorReason> {
    let lhs_len = match lhs {
        Err::Code(_) => panic!("expected error position"),
        Err::Node(_, _) => panic!("expected error position"),
        Err::Position(_, ref p) => p.len(),
        Err::NodePosition(_, ref p, _) => p.len(),
    };
    let rhs_len = match lhs {
        Err::Code(_) => panic!("expected error position"),
        Err::Node(_, _) => panic!("expected error position"),
        Err::Position(_, ref p) => p.len(),
        Err::NodePosition(_, ref p, _) => p.len(),
    };
    if rhs_len < lhs_len {
        lhs as Err<&'c [LexToken], ParseErrorReason>
    } else {
        rhs as Err<&'c [LexToken], ParseErrorReason>
    }
}

pub fn module(input: &[LexToken]) -> IResult<&[LexToken], Vec<RootDefinition>, ParseErrorReason> {
    let mut roots = Vec::new();
    let mut rest = input;
    loop {
        let last_def = rootdefinition_with_semicolon(rest);
        if let IResult::Done(remaining, root) = last_def {
            roots.push(root);
            rest = remaining;
        } else {
            return match rest {
                a if a.len() == 1 && a[0].0 == Token::Eof => IResult::Done(&[], roots),
                _ => {
                    match last_def {
                        IResult::Done(_, _) => unreachable!(),
                        IResult::Incomplete(rem) => IResult::Incomplete(rem),
                        IResult::Error(err) => IResult::Error(err),
                    }
                }
            };
        }
    }
}

fn errorkind_to_reason(errkind: ErrorKind<ParseErrorReason>) -> ParseErrorReason {
    match errkind {
        ErrorKind::Custom(reason) => reason,
        _ => ParseErrorReason::Unknown,
    }
}

fn iresult_to_error(err: Err<&[LexToken], ParseErrorReason>) -> ParseError {
    match err {
        Err::Code(error) => ParseError(errorkind_to_reason(error), None, None),
        Err::Node(error, inner_err) => {
            ParseError(errorkind_to_reason(error),
                       None,
                       Some(Box::new(iresult_to_error(*inner_err))))
        }
        Err::Position(error, position) => {
            ParseError(errorkind_to_reason(error), Some(position.to_vec()), None)
        }
        Err::NodePosition(error, position, inner_err) => {
            ParseError(errorkind_to_reason(error),
                       Some(position.to_vec()),
                       Some(Box::new(iresult_to_error(*inner_err))))
        }
    }
}

pub fn parse(entry_point: String, source: &[LexToken]) -> Result<Module, ParseError> {
    let parse_result = module(source);

    match parse_result {
        IResult::Done(rest, _) if rest.len() != 0 => {
            Err(ParseError(ParseErrorReason::FailedToParse, Some(rest.to_vec()), None))
        }
        IResult::Done(_, hlsl) => {
            Ok(Module {
                entry_point: entry_point,
                root_definitions: hlsl,
            })
        }
        IResult::Error(err) => Err(iresult_to_error(err)),
        IResult::Incomplete(_) => {
            Err(ParseError(ParseErrorReason::UnexpectedEndOfStream, None, None))
        }
    }
}


#[cfg(test)]
fn exp_var(var_name: &'static str, line: u64, column: u64) -> Located<Expression> {
    Located::loc(line, column, Expression::Variable(var_name.to_string()))
}
#[cfg(test)]
fn bexp_var(var_name: &'static str, line: u64, column: u64) -> Box<Located<Expression>> {
    Box::new(Located::loc(line, column, Expression::Variable(var_name.to_string())))
}

#[cfg(test)]
fn parse_result_from_str<T>(parse_func: Box<Fn(&[LexToken])
                                               -> IResult<&[LexToken], T, ParseErrorReason>>)
                            -> Box<Fn(&'static str) -> Result<T, ParseErrorReason>>
    where T: 'static
{
    use slp_transform_preprocess::preprocess_single;
    use slp_transform_lexer::lex;
    Box::new(move |string: &'static str| {
        let modified_string = string.to_string() + "\n";
        let preprocessed_text = preprocess_single(&modified_string).expect("preprocess failed");
        let lex_result = lex(&preprocessed_text);
        match lex_result {
            Ok(tokens) => {
                let stream = &tokens.stream;
                match parse_func(stream) {
                    IResult::Done(rem, exp) => {
                        if rem.len() == 1 && rem[0].0 == Token::Eof {
                            Ok(exp)
                        } else {
                            Err(ParseErrorReason::FailedToParse)
                        }
                    }
                    IResult::Incomplete(_) => Err(ParseErrorReason::UnexpectedEndOfStream),
                    _ => Err(ParseErrorReason::FailedToParse),
                }
            }
            Err(error) => panic!("Failed to lex `{:?}`", error),
        }
    })
}

#[cfg(test)]
fn parse_from_str<T>(parse_func: Box<Fn(&[LexToken]) -> IResult<&[LexToken], T, ParseErrorReason>>)
                     -> Box<Fn(&'static str) -> T>
    where T: 'static
{
    use slp_transform_preprocess::preprocess_single;
    use slp_transform_lexer::lex;
    Box::new(move |string: &'static str| {
        let modified_string = string.to_string() + "\n";
        let preprocessed_text = preprocess_single(&modified_string).expect("preprocess failed");
        let lex_result = lex(&preprocessed_text);
        match lex_result {
            Ok(tokens) => {
                let stream = &tokens.stream;
                match parse_func(stream) {
                    IResult::Done(rem, exp) => {
                        if rem.len() == 1 && rem[0].0 == Token::Eof {
                            exp
                        } else {
                            panic!("Tokens remaining while parsing `{:?}`: {:?}", stream, rem)
                        }
                    }
                    IResult::Incomplete(needed) => {
                        panic!("Failed to parse `{:?}`: Needed {:?} more", stream, needed)
                    }
                    IResult::Error(err) => {
                        panic!("Failed to parse `{:?}`: Error: {:?}", err, stream)
                    }
                }
            }
            Err(error) => panic!("Failed to lex `{:?}`", error),
        }
    })
}

#[test]
fn test_expr() {

    use slp_shared::{FileLocation, File, Line, Column};

    assert_eq!(expr(&[
            LexToken(Token::Id(Identifier("a".to_string())), FileLocation(File::Unknown, Line(1), Column(1))),
            LexToken(Token::Asterix, FileLocation(File::Unknown, Line(1), Column(2))),
            LexToken(Token::Id(Identifier("b".to_string())), FileLocation(File::Unknown, Line(1), Column(3))),
            LexToken(Token::Eof, FileLocation(File::Unknown, Line(1), Column(4)))
        ][..]), IResult::Done(&[LexToken(Token::Eof, FileLocation(File::Unknown, Line(1), Column(4)))][..],
        Located::loc(1, 1, Expression::BinaryOperation(
            BinOp::Multiply,
            bexp_var("a", 1, 1),
            bexp_var("b", 1, 3)
        ))
    ));

    let expr_str = parse_from_str(Box::new(expr));
    let expr_str_fail = parse_result_from_str(Box::new(expr));

    assert_eq!(expr_str("a"), exp_var("a", 1, 1));
    assert_eq!(expr_str("4"),
               Located::loc(1, 1, Expression::Literal(Literal::UntypedInt(4))));
    assert_eq!(expr_str("a+b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::Add,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 3))));
    assert_eq!(expr_str("a*b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::Multiply,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 3))));
    assert_eq!(expr_str("a + b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::Add,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));

    assert_eq!(expr_str("a-b+c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Add,
        Box::new(Located::loc(1, 1, Expression::BinaryOperation(BinOp::Subtract, bexp_var("a", 1, 1), bexp_var("b", 1, 3)))),
        bexp_var("c", 1, 5)
    )));
    assert_eq!(expr_str("a-b*c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Subtract,
        bexp_var("a", 1, 1),
        Box::new(Located::loc(1, 3, Expression::BinaryOperation(BinOp::Multiply, bexp_var("b", 1, 3), bexp_var("c", 1, 5))))
    )));
    assert_eq!(expr_str("a*b-c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Subtract,
        Box::new(Located::loc(1, 1, Expression::BinaryOperation(BinOp::Multiply, bexp_var("a", 1, 1), bexp_var("b", 1, 3)))),
        bexp_var("c", 1, 5)
    )));
    assert_eq!(expr_str("a-b*c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Subtract,
        bexp_var("a", 1, 1),
        Box::new(Located::loc(1, 3, Expression::BinaryOperation(BinOp::Multiply, bexp_var("b", 1, 3), bexp_var("c", 1, 5))))
    )));
    assert_eq!(expr_str("a*b-c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Subtract,
        Box::new(Located::loc(1, 1, Expression::BinaryOperation(BinOp::Multiply, bexp_var("a", 1, 1), bexp_var("b", 1, 3)))),
        bexp_var("c", 1, 5)
    )));
    assert_eq!(expr_str("a*(b-c)"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Multiply,
        bexp_var("a", 1, 1),
        Box::new(Located::loc(1, 3, Expression::BinaryOperation(BinOp::Subtract, bexp_var("b", 1, 4), bexp_var("c", 1, 6))))
    )));
    assert_eq!(expr_str("a*b/c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Divide,
        Box::new(Located::loc(1, 1, Expression::BinaryOperation(BinOp::Multiply, bexp_var("a", 1, 1), bexp_var("b", 1, 3)))),
        bexp_var("c", 1, 5)
    )));
    assert_eq!(expr_str("(a*b)/c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Divide,
        Box::new(Located::loc(1, 1, Expression::BinaryOperation(BinOp::Multiply, bexp_var("a", 1, 2), bexp_var("b", 1, 4)))),
        bexp_var("c", 1, 7)
    )));
    assert_eq!(expr_str("a*(b/c)"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Multiply,
        bexp_var("a", 1, 1),
        Box::new(Located::loc(1, 3, Expression::BinaryOperation(BinOp::Divide, bexp_var("b", 1, 4), bexp_var("c", 1, 6))))
    )));

    assert_eq!(expr_str("a++"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::PostfixIncrement,
                                                       bexp_var("a", 1, 1))));
    assert_eq!(expr_str("a--"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::PostfixDecrement,
                                                       bexp_var("a", 1, 1))));
    assert_eq!(expr_str("++a"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::PrefixIncrement,
                                                       bexp_var("a", 1, 3))));
    assert_eq!(expr_str("--a"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::PrefixDecrement,
                                                       bexp_var("a", 1, 3))));
    assert_eq!(expr_str("+a"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::Plus, bexp_var("a", 1, 2))));
    assert_eq!(expr_str("-a"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::Minus, bexp_var("a", 1, 2))));
    assert_eq!(expr_str("!a"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::LogicalNot, bexp_var("a", 1, 2))));
    assert_eq!(expr_str("~a"),
               Located::loc(1,
                            1,
                            Expression::UnaryOperation(UnaryOp::BitwiseNot, bexp_var("a", 1, 2))));

    assert_eq!(expr_str("a << b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::LeftShift,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a >> b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::RightShift,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a < b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::LessThan,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));
    assert_eq!(expr_str("a <= b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::LessEqual,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a > b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::GreaterThan,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));
    assert_eq!(expr_str("a >= b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::GreaterEqual,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a == b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::Equality,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a != b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::Inequality,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a & b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::BitwiseAnd,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));
    assert_eq!(expr_str("a | b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::BitwiseOr,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));
    assert_eq!(expr_str("a ^ b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::BitwiseXor,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));
    assert_eq!(expr_str("a && b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::BooleanAnd,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a || b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::BooleanOr,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));

    assert_eq!(expr_str_fail("a < < b"),
               Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a > > b"),
               Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a < = b"),
               Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a > = b"),
               Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a = = b"),
               Err(ParseErrorReason::FailedToParse));
    assert_eq!(expr_str_fail("a ! = b"),
               Err(ParseErrorReason::FailedToParse));

    assert_eq!(expr_str("a[b]"),
               Located::loc(1,
                            1,
                            Expression::ArraySubscript(bexp_var("a", 1, 1), bexp_var("b", 1, 3))));
    assert_eq!(expr_str("d+a[b+c]"), Located::loc(1, 1,
        Expression::BinaryOperation(BinOp::Add,
            bexp_var("d", 1, 1),
            Box::new(Located::loc(1, 3, Expression::ArraySubscript(bexp_var("a", 1, 3),
                Box::new(Located::loc(1, 5, Expression::BinaryOperation(BinOp::Add,
                    bexp_var("b", 1, 5), bexp_var("c", 1, 7)
                )))
            )))
        )
    ));
    assert_eq!(expr_str(" d + a\t[ b\n+ c ]"), Located::loc(1, 2,
        Expression::BinaryOperation(BinOp::Add,
            bexp_var("d", 1, 2),
            Box::new(Located::loc(1, 6, Expression::ArraySubscript(bexp_var("a", 1, 6),
                Box::new(Located::loc(1, 10, Expression::BinaryOperation(BinOp::Add,
                    bexp_var("b", 1, 10), bexp_var("c", 2, 3)
                )))
            )))
        )
    ));

    assert_eq!(expr_str("array.Load"),
               Located::loc(1,
                            1,
                            Expression::Member(bexp_var("array", 1, 1), "Load".to_string())));
    assert_eq!(expr_str("array.Load()"), Located::loc(1, 1,
        Expression::Call(Box::new(Located::loc(1, 1, Expression::Member(bexp_var("array", 1, 1), "Load".to_string()))), vec![])
    ));
    assert_eq!(expr_str(" array . Load ( ) "), Located::loc(1, 2,
        Expression::Call(Box::new(Located::loc(1, 2, Expression::Member(bexp_var("array", 1, 2), "Load".to_string()))), vec![])
    ));
    assert_eq!(expr_str("array.Load(a)"), Located::loc(1, 1,
        Expression::Call(Box::new(Located::loc(1, 1, Expression::Member(bexp_var("array", 1, 1), "Load".to_string()))), vec![exp_var("a", 1, 12)])
    ));
    assert_eq!(expr_str("array.Load(a,b)"), Located::loc(1, 1,
        Expression::Call(Box::new(Located::loc(1, 1, Expression::Member(bexp_var("array", 1, 1), "Load".to_string()))), vec![exp_var("a", 1, 12), exp_var("b", 1, 14)])
    ));
    assert_eq!(expr_str("array.Load(a, b)"), Located::loc(1, 1,
        Expression::Call(Box::new(Located::loc(1, 1, Expression::Member(bexp_var("array", 1, 1), "Load".to_string()))), vec![exp_var("a", 1, 12), exp_var("b", 1, 15)])
    ));

    assert_eq!(expr_str("(float) b"),
               Located::loc(1, 1, Expression::Cast(Type::float(), bexp_var("b", 1, 9))));

    assert_eq!(expr_str("float2(b)"),
               Located::loc(1,
                            1,
                            Expression::NumericConstructor(DataLayout::Vector(ScalarType::Float,
                                                                              2),
                                                           vec![exp_var("b", 1, 8)])));

    assert_eq!(expr_str("a = b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::Assignment,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 5))));
    assert_eq!(expr_str("a = b = c"), Located::loc(1, 1, Expression::BinaryOperation(
        BinOp::Assignment,
        bexp_var("a", 1, 1),
        Box::new(Located::loc(1, 5, Expression::BinaryOperation(
            BinOp::Assignment,
            bexp_var("b", 1, 5),
            bexp_var("c", 1, 9)
        )))
    )));

    assert_eq!(expr_str("a += b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::SumAssignment,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));

    assert_eq!(expr_str("a -= b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::DifferenceAssignment,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a *= b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::ProductAssignment,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a /= b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::QuotientAssignment,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));
    assert_eq!(expr_str("a %= b"),
               Located::loc(1,
                            1,
                            Expression::BinaryOperation(BinOp::RemainderAssignment,
                                                        bexp_var("a", 1, 1),
                                                        bexp_var("b", 1, 6))));

    assert_eq!(expr_str("a ? b : c"),
               Located::loc(1,
                            1,
                            Expression::TernaryConditional(bexp_var("a", 1, 1),
                                                           bexp_var("b", 1, 5),
                                                           bexp_var("c", 1, 9))));
    assert_eq!(expr_str("a ? b ? c : d : e"), Located::loc(1, 1,
        Expression::TernaryConditional(
            bexp_var("a", 1, 1),
            Box::new(Located::loc(1, 5, Expression::TernaryConditional(
                bexp_var("b", 1, 5),
                bexp_var("c", 1, 9),
                bexp_var("d", 1, 13)
            ))),
            bexp_var("e", 1, 17)
        )
    ));
    assert_eq!(expr_str("a ? b : c ? d : e"), Located::loc(1, 1,
        Expression::TernaryConditional(
            bexp_var("a", 1, 1),
            bexp_var("b", 1, 5),
            Box::new(Located::loc(1, 9, Expression::TernaryConditional(
                bexp_var("c", 1, 9),
                bexp_var("d", 1, 13),
                bexp_var("e", 1, 17)
            )))
        )
    ));
    assert_eq!(expr_str("a ? b ? c : d : e ? f : g"), Located::loc(1, 1,
        Expression::TernaryConditional(
            bexp_var("a", 1, 1),
            Box::new(Located::loc(1, 5, Expression::TernaryConditional(
                bexp_var("b", 1, 5),
                bexp_var("c", 1, 9),
                bexp_var("d", 1, 13)
            ))),
            Box::new(Located::loc(1, 17, Expression::TernaryConditional(
                bexp_var("e", 1, 17),
                bexp_var("f", 1, 21),
                bexp_var("g", 1, 25)
            )))
        )
    ));
}

#[test]
fn test_statement() {

    let statement_str = parse_from_str(Box::new(statement));

    // Empty statement
    assert_eq!(statement_str(";"), Statement::Empty);

    // Expression statements
    assert_eq!(statement_str("func();"),
               Statement::Expression(Located::loc(1,
                                                  1,
                                                  Expression::Call(bexp_var("func", 1, 1),
                                                                   vec![]))));
    assert_eq!(statement_str(" func ( ) ; "),
               Statement::Expression(Located::loc(1,
                                                  2,
                                                  Expression::Call(bexp_var("func", 1, 2),
                                                                   vec![]))));

    // For loop init statement
    let init_statement_str = parse_from_str(Box::new(init_statement));
    let vardef_str = parse_from_str(Box::new(vardef));

    assert_eq!(init_statement_str("x"),
               InitStatement::Expression(exp_var("x", 1, 1)));
    assert_eq!(vardef_str("uint x"), VarDef::one("x", Type::uint().into()));
    assert_eq!(init_statement_str("uint x"),
               InitStatement::Declaration(VarDef::one("x", Type::uint().into())));
    assert_eq!(init_statement_str("uint x = y"),
               InitStatement::Declaration(VarDef::one_with_expr("x",
                                                                Type::uint().into(),
                                                                exp_var("y", 1, 10))));

    // Variable declarations
    assert_eq!(statement_str("uint x = y;"),
               Statement::Var(VarDef::one_with_expr("x",
                                                    Type::uint().into(),
                                                    exp_var("y", 1, 10))));
    assert_eq!(statement_str("float x[3];"),
        Statement::Var(VarDef {
            local_type: Type::from_layout(TypeLayout::float()).into(),
            defs: vec![LocalVariableName {
                name: "x".to_string(),
                bind: VariableBind::Array(Some(Located::loc(1, 9, Expression::Literal(Literal::UntypedInt(3))))),
                init: None,
            }]
        })
    );

    // Blocks
    assert_eq!(statement_str("{one();two();}"),
        Statement::Block(vec![
            Statement::Expression(Located::loc(1, 2, Expression::Call(bexp_var("one", 1, 2), vec![]))),
            Statement::Expression(Located::loc(1, 8, Expression::Call(bexp_var("two", 1, 8), vec![])))
        ])
    );
    assert_eq!(statement_str(" { one(); two(); } "),
        Statement::Block(vec![
            Statement::Expression(Located::loc(1, 4, Expression::Call(bexp_var("one", 1, 4), vec![]))),
            Statement::Expression(Located::loc(1, 11, Expression::Call(bexp_var("two", 1, 11), vec![])))
        ])
    );

    // If statement
    assert_eq!(statement_str("if(a)func();"),
        Statement::If(exp_var("a", 1, 4), Box::new(Statement::Expression(Located::loc(1, 6, Expression::Call(bexp_var("func", 1, 6), vec![])))))
    );
    assert_eq!(statement_str("if (a) func(); "),
        Statement::If(exp_var("a", 1, 5), Box::new(Statement::Expression(Located::loc(1, 8, Expression::Call(bexp_var("func", 1, 8), vec![])))))
    );
    assert_eq!(statement_str("if (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::If(exp_var("a", 1, 5), Box::new(Statement::Block(vec![
            Statement::Expression(Located::loc(3, 2, Expression::Call(bexp_var("one", 3, 2), vec![]))),
            Statement::Expression(Located::loc(4, 2, Expression::Call(bexp_var("two", 4, 2), vec![])))
        ])))
    );

    // If-else statement
    assert_eq!(statement_str("if (a) one(); else two();"),
        Statement::IfElse(exp_var("a", 1, 5),
            Box::new(Statement::Expression(Located::loc(1, 8, Expression::Call(bexp_var("one", 1, 8), vec![])))),
            Box::new(Statement::Expression(Located::loc(1, 20, Expression::Call(bexp_var("two", 1, 20), vec![]))))
        )
    );

    // While loops
    assert_eq!(statement_str("while (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::While(exp_var("a", 1, 8), Box::new(Statement::Block(vec![
            Statement::Expression(Located::loc(3, 2, Expression::Call(bexp_var("one", 3, 2), vec![]))),
            Statement::Expression(Located::loc(4, 2, Expression::Call(bexp_var("two", 4, 2), vec![])))
        ])))
    );

    // For loops
    assert_eq!(statement_str("for(a;b;c)func();"),
        Statement::For(InitStatement::Expression(exp_var("a", 1, 5)), exp_var("b", 1, 7), exp_var("c", 1, 9), Box::new(
            Statement::Expression(Located::loc(1, 11, Expression::Call(bexp_var("func", 1, 11), vec![])))
        ))
    );
    assert_eq!(statement_str("for (uint i = 0; i; i++) { func(); }"),
        Statement::For(
            InitStatement::Declaration(VarDef::one_with_expr("i", Type::uint().into(), Located::loc(1, 15, Expression::Literal(Literal::UntypedInt(0))))),
            exp_var("i", 1, 18),
            Located::loc(1, 21, Expression::UnaryOperation(UnaryOp::PostfixIncrement, bexp_var("i", 1, 21))),
            Box::new(Statement::Block(vec![Statement::Expression(Located::loc(1, 28, Expression::Call(bexp_var("func", 1, 28), vec![])))]))
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
            StructMember { ty: Type::uint(), defs: vec![StructMemberName { name: "a".to_string(), bind: VariableBind::Normal }] },
            StructMember { ty: Type::float(), defs: vec![StructMemberName { name: "b".to_string(), bind: VariableBind::Normal }] },
        ],
    };
    assert_eq!(structdefinition_str(test_struct_str),
               test_struct_ast.clone());
    assert_eq!(rootdefinition_str(test_struct_str),
               RootDefinition::Struct(test_struct_ast.clone()));

    let functiondefinition_str = parse_from_str(Box::new(functiondefinition));

    let test_func_str = "void func(float x) { }";
    let test_func_ast = FunctionDefinition {
        name: "func".to_string(),
        returntype: Type::void(),
        params: vec![FunctionParam {
                         name: "x".to_string(),
                         param_type: Type::float().into(),
                         semantic: None,
                     }],
        body: vec![],
        attributes: vec![],
    };
    assert_eq!(functiondefinition_str(test_func_str), test_func_ast.clone());
    assert_eq!(rootdefinition_str(test_func_str),
               RootDefinition::Function(test_func_ast.clone()));
    let untyped_int = Literal::UntypedInt;
    let elit = Expression::Literal;
    let loc = Located::loc;
    let numthreads = FunctionAttribute::NumThreads(loc(1, 13, elit(untyped_int(16))),
                                                   loc(1, 17, elit(untyped_int(16))),
                                                   loc(1, 21, elit(untyped_int(1))));
    assert_eq!(rootdefinition_str("[numthreads(16, 16, 1)] void func(float x) { }"),
               RootDefinition::Function(FunctionDefinition {
                   name: "func".to_string(),
                   returntype: Type::void(),
                   params: vec![FunctionParam {
                                    name: "x".to_string(),
                                    param_type: Type::float().into(),
                                    semantic: None,
                                }],
                   body: vec![],
                   attributes: vec![numthreads],
               }));

    let constantvariable_str = parse_from_str(Box::new(constantvariable));

    let test_cbuffervar_str = "float4x4 wvp;";
    let test_cbuffervar_ast = ConstantVariable {
        ty: Type::float4x4(),
        defs: vec![ConstantVariableName {
                       name: "wvp".to_string(),
                       bind: VariableBind::Normal,
                       offset: None,
                   }],
    };
    assert_eq!(constantvariable_str(test_cbuffervar_str),
               test_cbuffervar_ast.clone());

    let cbuffer_str = parse_from_str(Box::new(cbuffer));

    let test_cbuffer1_str = "cbuffer globals { float4x4 wvp; }";
    let test_cbuffer1_ast = ConstantBuffer {
        name: "globals".to_string(),
        slot: None,
        members: vec![ConstantVariable {
                          ty: Type::float4x4(),
                          defs: vec![ConstantVariableName {
                                         name: "wvp".to_string(),
                                         bind: VariableBind::Normal,
                                         offset: None,
                                     }],
                      }],
    };
    assert_eq!(cbuffer_str(test_cbuffer1_str), test_cbuffer1_ast.clone());
    assert_eq!(rootdefinition_str(test_cbuffer1_str),
               RootDefinition::ConstantBuffer(test_cbuffer1_ast.clone()));

    let cbuffer_register_str = parse_from_str(Box::new(cbuffer_register));
    assert_eq!(cbuffer_register_str(" : register(b12) "), ConstantSlot(12));

    let test_cbuffer2_str = "cbuffer globals : register(b12) { float4x4 wvp; float x, y[2]; }";
    let test_cbuffer2_ast_wvp = ConstantVariable {
        ty: Type::float4x4(),
        defs: vec![ConstantVariableName {
                       name: "wvp".to_string(),
                       bind: VariableBind::Normal,
                       offset: None,
                   }],
    };
    let test_cbuffer2_ast_xy = ConstantVariable {
        ty: Type::float(),
        defs: vec![ConstantVariableName {
                       name: "x".to_string(),
                       bind: VariableBind::Normal,
                       offset: None,
                   },
                   ConstantVariableName {
                       name: "y".to_string(),
                       bind: VariableBind::Array(Some(Located::loc(1, 60, Expression::Literal(Literal::UntypedInt(2))))),
                       offset: None,
                   }],
    };
    let test_cbuffer2_ast = ConstantBuffer {
        name: "globals".to_string(),
        slot: Some(ConstantSlot(12)),
        members: vec![test_cbuffer2_ast_wvp, test_cbuffer2_ast_xy],
    };
    assert_eq!(cbuffer_str(test_cbuffer2_str), test_cbuffer2_ast.clone());
    assert_eq!(rootdefinition_str(test_cbuffer2_str),
               RootDefinition::ConstantBuffer(test_cbuffer2_ast.clone()));

    let globalvariable_str = parse_from_str(Box::new(globalvariable));

    let test_buffersrv_str = "Buffer g_myBuffer : register(t1);";
    let test_buffersrv_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::Float, 4), TypeModifier::default()))).into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv_str),
               test_buffersrv_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv_str),
               RootDefinition::GlobalVariable(test_buffersrv_ast.clone()));

    let test_buffersrv2_str = "Buffer<uint4> g_myBuffer : register(t1);";
    let test_buffersrv2_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::UInt, 4), TypeModifier::default()))).into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv2_str),
               test_buffersrv2_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv2_str),
               RootDefinition::GlobalVariable(test_buffersrv2_ast.clone()));

    let test_buffersrv3_str = "Buffer<vector<int, 4>> g_myBuffer : register(t1);";
    let test_buffersrv3_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::Int, 4), TypeModifier::default()))).into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv3_str),
               test_buffersrv3_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv3_str),
               RootDefinition::GlobalVariable(test_buffersrv3_ast.clone()));

    let test_buffersrv4_str = "StructuredBuffer<CustomType> g_myBuffer : register(t1);";
    let test_buffersrv4_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::StructuredBuffer(StructuredType(StructuredLayout::Custom("CustomType".to_string()), TypeModifier::default()))).into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv4_str),
               test_buffersrv4_ast.clone());
    assert_eq!(rootdefinition_str(test_buffersrv4_str),
               RootDefinition::GlobalVariable(test_buffersrv4_ast.clone()));

    let test_static_const_str = "static const int c_numElements = 4;";
    let test_static_const_ast = GlobalVariable {
        global_type: GlobalType(Type(TypeLayout::int(),
                                     TypeModifier { is_const: true, ..TypeModifier::default() }),
                                GlobalStorage::Static,
                                None),
        defs: vec![GlobalVariableName {
                       name: "c_numElements".to_string(),
                       bind: VariableBind::Normal,
                       slot: None,
                       init: Some(Initializer::Expression(Located::loc(1,
                                                     34,
                                                     Expression::Literal(Literal::UntypedInt(4))))),
                   }],
    };
    assert_eq!(globalvariable_str(test_static_const_str),
               test_static_const_ast.clone());
    assert_eq!(rootdefinition_str(test_static_const_str),
               RootDefinition::GlobalVariable(test_static_const_ast.clone()));

    let test_const_arr_str = "static const int data[4] = { 0, 1, 2, 3 };";
    let test_const_arr_ast_lits = vec![
        Initializer::Expression(Located::loc(1, 30, Expression::Literal(Literal::UntypedInt(0)))),
        Initializer::Expression(Located::loc(1, 33, Expression::Literal(Literal::UntypedInt(1)))),
        Initializer::Expression(Located::loc(1, 36, Expression::Literal(Literal::UntypedInt(2)))),
        Initializer::Expression(Located::loc(1, 39, Expression::Literal(Literal::UntypedInt(3)))),
    ];
    let test_const_arr_ast = GlobalVariable {
        global_type: GlobalType(Type(TypeLayout::int(),
                                     TypeModifier { is_const: true, ..TypeModifier::default() }),
                                GlobalStorage::Static,
                                None),
        defs: vec![GlobalVariableName {
                       name: "data".to_string(),
                       bind: VariableBind::Array(Some(Located::loc(1, 23, Expression::Literal(Literal::UntypedInt(4))))),
                       slot: None,
                       init: Some(Initializer::Aggregate(test_const_arr_ast_lits)),
                   }],
    };
    assert_eq!(globalvariable_str(test_const_arr_str),
               test_const_arr_ast.clone());
    assert_eq!(rootdefinition_str(test_const_arr_str),
               RootDefinition::GlobalVariable(test_const_arr_ast.clone()));

    let test_groupshared_str = "groupshared float4 local_data[32];";
    let test_groupshared_ast = GlobalVariable {
        global_type: GlobalType(Type::floatn(4), GlobalStorage::GroupShared, None),
        defs: vec![GlobalVariableName {
                       name: "local_data".to_string(),
                       bind: VariableBind::Array(Some(Located::loc(1, 31, Expression::Literal(Literal::UntypedInt(32))))),
                       slot: None,
                       init: None,
                   }],
    };
    assert_eq!(globalvariable_str(test_groupshared_str),
               test_groupshared_ast.clone());
    assert_eq!(rootdefinition_str(test_groupshared_str),
               RootDefinition::GlobalVariable(test_groupshared_ast.clone()));
}
