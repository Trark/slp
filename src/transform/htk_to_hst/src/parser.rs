use slp_lang_hst::*;
use slp_lang_htk::*;
use slp_shared::*;
use std::collections::HashMap;
use std::fmt;

#[derive(PartialEq, Clone)]
pub struct ParseError(
    pub ParseErrorReason,
    pub Option<Vec<LexToken>>,
    pub Option<Box<ParseError>>,
);

impl fmt::Debug for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tokens = match self.1 {
            Some(ref vec) => {
                let mut slice = &vec[..];
                let mut line = None;
                for i in 0..vec.len() {
                    match vec[i].1 {
                        FileLocation::Known(_, cur_line, _) => match line {
                            Some(last_line) => {
                                if cur_line != last_line {
                                    break;
                                }
                            }
                            None => line = Some(cur_line),
                        },
                        FileLocation::Unknown => break,
                    }
                    slice = &vec[..(i + 1)]
                }
                Some(slice)
            }
            None => None,
        };
        write!(f, "ParseError({:?}, {:?}, {:?})", self.0, tokens, self.2)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum ParseErrorReason {
    Unknown,
    UnexpectedEndOfStream,
    FailedToParse,
    WrongToken,
    WrongSlotType,
    UnknownType,
    DuplicateStructSymbol,
    SymbolIsNotAStruct,
    UnexpectedAttribute(String),
    ErrorKind(nom::error::ErrorKind),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parser error")
    }
}

enum SymbolType {
    Struct,
}

struct SymbolTable(HashMap<String, SymbolType>);

impl SymbolTable {
    fn empty() -> SymbolTable {
        SymbolTable(HashMap::new())
    }
}

type ParseResult<'t, T> = nom::IResult<&'t [LexToken], T, ParseErrorContext<'t>>;

#[derive(PartialEq, Debug, Clone)]
struct ParseErrorContext<'a>(&'a [LexToken], ParseErrorReason);

impl<'a> nom::error::ParseError<&'a [LexToken]> for ParseErrorContext<'a> {
    fn from_error_kind(input: &'a [LexToken], kind: nom::error::ErrorKind) -> Self {
        ParseErrorContext(input, ParseErrorReason::ErrorKind(kind))
    }

    fn append(_: &[LexToken], _: nom::error::ErrorKind, other: Self) -> Self {
        other
    }
}

trait Parse: Sized {
    type Output;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self::Output>;
}

/// Create parser for a type that implements Parse
fn parse_typed<'t, 's, T: Parse>(
    st: &'s SymbolTable,
) -> impl Fn(&'t [LexToken]) -> ParseResult<T::Output> + 's {
    move |input: &'t [LexToken]| T::parse(input, st)
}

/// Parse an exact token from the start of the stream
fn parse_token<'t>(token: Token) -> impl Fn(&'t [LexToken]) -> ParseResult<LexToken> {
    move |input: &'t [LexToken]| {
        if input.is_empty() {
            return Err(nom::Err::Incomplete(nom::Needed::new(1)));
        }
        match input[0] {
            LexToken(ref t, _) if *t == token => Ok((&input[1..], input[0].clone())),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }
}

/// Match a single identifier token
fn match_identifier(input: &[LexToken]) -> ParseResult<&Identifier> {
    match input {
        [LexToken(Token::Id(ref id), _), rest @ ..] => Ok((rest, id)),
        _ => Err(nom::Err::Error(ParseErrorContext(
            input,
            ParseErrorReason::WrongToken,
        ))),
    }
}

/// Match a single < that may or may not be followed by whitespace
fn match_left_angle_bracket(input: &[LexToken]) -> ParseResult<LexToken> {
    match input {
        [first @ LexToken(Token::LeftAngleBracket(_), _), rest @ ..] => Ok((rest, first.clone())),
        _ => Err(nom::Err::Error(ParseErrorContext(
            input,
            ParseErrorReason::WrongToken,
        ))),
    }
}

/// Match a single > that may or may not be followed by whitespace
fn match_right_angle_bracket(input: &[LexToken]) -> ParseResult<LexToken> {
    match input {
        [first @ LexToken(Token::RightAngleBracket(_), _), rest @ ..] => Ok((rest, first.clone())),
        _ => Err(nom::Err::Error(ParseErrorContext(
            input,
            ParseErrorReason::WrongToken,
        ))),
    }
}

struct VariableName;

impl Parse for VariableName {
    type Output = Located<String>;
    fn parse<'t>(input: &'t [LexToken], _: &SymbolTable) -> ParseResult<'t, Self::Output> {
        if input.is_empty() {
            return Err(nom::Err::Incomplete(nom::Needed::new(1)));
        }

        match &input[0] {
            LexToken(Token::Id(Identifier(name)), loc) => {
                Ok((&input[1..], Located::new(name.clone(), loc.clone())))
            }
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }
}

// Parse scalar type as part of a string
fn parse_scalartype_str(input: &[u8]) -> nom::IResult<&[u8], ScalarType> {
    use nom::bytes::complete::tag;
    use nom::combinator::map;
    nom::branch::alt((
        map(tag("bool"), |_| ScalarType::Bool),
        map(tag("int"), |_| ScalarType::Int),
        map(tag("uint"), |_| ScalarType::UInt),
        map(tag("dword"), |_| ScalarType::UInt),
        map(tag("half"), |_| ScalarType::Half),
        map(tag("float"), |_| ScalarType::Float),
        map(tag("double"), |_| ScalarType::Double),
    ))(input)
}

// Parse data type as part of a string
fn parse_datalayout_str(typename: &str) -> Option<DataLayout> {
    fn digit(input: &[u8]) -> nom::IResult<&[u8], u32> {
        // Handle end of stream
        if input.is_empty() {
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            )));
        };

        // Match on the next character
        let n = match input[0] {
            b'1' => 1,
            b'2' => 2,
            b'3' => 3,
            b'4' => 4,
            _ => {
                // Not a digit
                return Err(nom::Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )));
            }
        };

        // Success
        Ok((&input[1..], n))
    }

    fn parse_str(input: &[u8]) -> nom::IResult<&[u8], DataLayout> {
        let (rest, ty) = parse_scalartype_str(input)?;
        if rest.is_empty() {
            return Ok((&[], DataLayout::Scalar(ty)));
        }

        let (rest, x) = digit(rest)?;
        if rest.is_empty() {
            return Ok((&[], DataLayout::Vector(ty, x)));
        }

        let rest = match rest.first() {
            Some(b'x') => &rest[1..],
            _ => {
                return Err(nom::Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        };

        let (rest, y) = digit(rest)?;
        if rest.is_empty() {
            return Ok((&[], DataLayout::Matrix(ty, x, y)));
        }

        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )))
    }

    match parse_str(typename[..].as_bytes()) {
        Ok((rest, ty)) => {
            assert_eq!(rest.len(), 0);
            Some(ty)
        }
        Err(_) => None,
    }
}

#[test]
fn test_parse_datalayout_str() {
    assert_eq!(
        parse_datalayout_str("float"),
        Some(DataLayout::Scalar(ScalarType::Float))
    );
    assert_eq!(
        parse_datalayout_str("uint3"),
        Some(DataLayout::Vector(ScalarType::UInt, 3))
    );
    assert_eq!(
        parse_datalayout_str("bool2x3"),
        Some(DataLayout::Matrix(ScalarType::Bool, 2, 3))
    );

    assert_eq!(parse_datalayout_str(""), None);
    assert_eq!(parse_datalayout_str("float5"), None);
    assert_eq!(parse_datalayout_str("float2x"), None);
}

impl Parse for DataLayout {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], _: &SymbolTable) -> ParseResult<'t, Self> {
        // Parse a vector dimension as a token
        fn parse_digit(input: &[LexToken]) -> ParseResult<u32> {
            if input.is_empty() {
                return Err(nom::Err::Incomplete(nom::Needed::new(1)));
            }

            match input[0] {
                LexToken(Token::LiteralInt(i), _) => Ok((&input[1..], i as u32)),
                _ => Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::WrongToken,
                ))),
            }
        }

        // Parse scalar type as a full token
        fn parse_scalartype(input: &[LexToken]) -> ParseResult<ScalarType> {
            if input.is_empty() {
                return Err(nom::Err::Incomplete(nom::Needed::new(1)));
            }

            match input[0] {
                LexToken(Token::Id(Identifier(ref name)), _) => {
                    match parse_scalartype_str(name[..].as_bytes()) {
                        Ok((rest, ty)) => {
                            if rest.is_empty() {
                                Ok((&input[1..], ty))
                            } else {
                                Err(nom::Err::Error(ParseErrorContext(
                                    input,
                                    ParseErrorReason::UnknownType,
                                )))
                            }
                        }
                        Err(nom::Err::Incomplete(rem)) => Err(nom::Err::Incomplete(rem)),
                        Err(_) => Err(nom::Err::Error(ParseErrorContext(
                            input,
                            ParseErrorReason::UnknownType,
                        ))),
                    }
                }
                _ => Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::WrongToken,
                ))),
            }
        }

        if input.is_empty() {
            Err(nom::Err::Incomplete(nom::Needed::new(1)))
        } else {
            match &input[0] {
                &LexToken(Token::Id(Identifier(ref name)), _) => match &name[..] {
                    "vector" => {
                        let (input, _) = match_left_angle_bracket(&input[1..])?;
                        let (input, scalar) = parse_scalartype(input)?;
                        let (input, _) = parse_token(Token::Comma)(input)?;
                        let (input, x) = parse_digit(input)?;
                        let (input, _) = match_right_angle_bracket(input)?;
                        Ok((input, DataLayout::Vector(scalar, x)))
                    }
                    "matrix" => {
                        let (input, _) = match_left_angle_bracket(&input[1..])?;
                        let (input, scalar) = parse_scalartype(input)?;
                        let (input, _) = parse_token(Token::Comma)(input)?;
                        let (input, x) = parse_digit(input)?;
                        let (input, _) = parse_token(Token::Comma)(input)?;
                        let (input, y) = parse_digit(input)?;
                        let (input, _) = match_right_angle_bracket(input)?;
                        Ok((input, DataLayout::Matrix(scalar, x, y)))
                    }
                    _ => match parse_datalayout_str(&name[..]) {
                        Some(ty) => Ok((&input[1..], ty)),
                        None => Err(nom::Err::Error(ParseErrorContext(
                            input,
                            ParseErrorReason::UnknownType,
                        ))),
                    },
                },
                _ => Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::WrongToken,
                ))),
            }
        }
    }
}

impl Parse for DataType {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        // Todo: Modifiers
        match DataLayout::parse(input, st) {
            Ok((rest, layout)) => Ok((rest, DataType(layout, Default::default()))),
            Err(err) => Err(err),
        }
    }
}

impl Parse for StructuredLayout {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        // Attempt to parse a primitive structured type
        if let Ok((input, ty)) = DataLayout::parse(input, st) {
            let sl = match ty {
                DataLayout::Scalar(scalar) => StructuredLayout::Scalar(scalar),
                DataLayout::Vector(scalar, x) => StructuredLayout::Vector(scalar, x),
                DataLayout::Matrix(scalar, x, y) => StructuredLayout::Matrix(scalar, x, y),
            };
            return Ok((input, sl));
        }

        // Attempt to parse a custom type
        match match_identifier(input) {
            Ok((input, Identifier(name))) => match st.0.get(name) {
                Some(&SymbolType::Struct) => Ok((input, StructuredLayout::Custom(name.clone()))),
                _ => Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::SymbolIsNotAStruct,
                ))),
            },
            Err(err) => Err(err),
        }
    }
}

impl Parse for StructuredType {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        // Todo: Modifiers
        match StructuredLayout::parse(input, st) {
            Ok((rest, layout)) => Ok((rest, StructuredType(layout, Default::default()))),
            Err(err) => Err(err),
        }
    }
}

impl Parse for ObjectType {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        if input.is_empty() {
            return Err(nom::Err::Incomplete(nom::Needed::new(1)));
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
            &LexToken(Token::Id(Identifier(ref name)), _) => match &name[..] {
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

                _ => {
                    return Err(nom::Err::Error(ParseErrorContext(
                        input,
                        ParseErrorReason::UnknownType,
                    )))
                }
            },
            _ => {
                return Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::UnknownType,
                )))
            }
        };

        let rest = &input[1..];

        match object_type {
            ParseType::ByteAddressBuffer => Ok((rest, ObjectType::ByteAddressBuffer)),
            ParseType::RWByteAddressBuffer => Ok((rest, ObjectType::RWByteAddressBuffer)),

            ParseType::Buffer
            | ParseType::RWBuffer
            | ParseType::Texture1D
            | ParseType::Texture1DArray
            | ParseType::Texture2D
            | ParseType::Texture2DArray
            | ParseType::Texture2DMS
            | ParseType::Texture2DMSArray
            | ParseType::Texture3D
            | ParseType::TextureCube
            | ParseType::TextureCubeArray
            | ParseType::RWTexture1D
            | ParseType::RWTexture1DArray
            | ParseType::RWTexture2D
            | ParseType::RWTexture2DArray
            | ParseType::RWTexture3D => {
                let (buffer_arg, rest) = match nom::sequence::delimited(
                    match_left_angle_bracket,
                    parse_typed::<DataType>(st),
                    match_right_angle_bracket,
                )(rest)
                {
                    Ok((rest, ty)) => (ty, rest),
                    Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
                    Err(_) => (
                        DataType(
                            DataLayout::Vector(ScalarType::Float, 4),
                            TypeModifier::default(),
                        ),
                        rest,
                    ),
                };

                let ty = match object_type {
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
                };
                Ok((rest, ty))
            }

            ParseType::StructuredBuffer
            | ParseType::RWStructuredBuffer
            | ParseType::AppendStructuredBuffer
            | ParseType::ConsumeStructuredBuffer => {
                let (buffer_arg, rest) = match nom::sequence::delimited(
                    match_left_angle_bracket,
                    parse_typed::<StructuredType>(st),
                    match_right_angle_bracket,
                )(rest)
                {
                    Ok((rest, ty)) => (ty, rest),
                    Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
                    Err(_) => (
                        StructuredType(
                            StructuredLayout::Vector(ScalarType::Float, 4),
                            TypeModifier::default(),
                        ),
                        rest,
                    ),
                };

                let ty = match object_type {
                    ParseType::StructuredBuffer => ObjectType::StructuredBuffer(buffer_arg),
                    ParseType::RWStructuredBuffer => ObjectType::RWStructuredBuffer(buffer_arg),
                    ParseType::AppendStructuredBuffer => {
                        ObjectType::AppendStructuredBuffer(buffer_arg)
                    }
                    ParseType::ConsumeStructuredBuffer => {
                        ObjectType::ConsumeStructuredBuffer(buffer_arg)
                    }
                    _ => unreachable!(),
                };
                Ok((rest, ty))
            }

            ParseType::InputPatch => Ok((rest, ObjectType::InputPatch)),
            ParseType::OutputPatch => Ok((rest, ObjectType::OutputPatch)),
        }
    }
}

fn parse_voidtype(input: &[LexToken]) -> ParseResult<TypeLayout> {
    if input.is_empty() {
        Err(nom::Err::Incomplete(nom::Needed::new(1)))
    } else {
        match &input[0] {
            &LexToken(Token::Id(Identifier(ref name)), _) => {
                if name == "void" {
                    Ok((&input[1..], TypeLayout::Void))
                } else {
                    Err(nom::Err::Error(ParseErrorContext(
                        input,
                        ParseErrorReason::UnknownType,
                    )))
                }
            }
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }
}

impl Parse for TypeLayout {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        if let Ok((input, ty)) = ObjectType::parse(input, st) {
            return Ok((input, TypeLayout::Object(ty)));
        }

        if let Ok((input, ty)) = parse_voidtype(input) {
            return Ok((input, ty));
        }

        if let Ok((input, _)) = parse_token(Token::SamplerState)(input) {
            return Ok((input, TypeLayout::SamplerState));
        }

        match StructuredLayout::parse(input, st) {
            Ok((input, sl)) => Ok((input, TypeLayout::from(sl))),
            Err(err) => Err(err),
        }
    }
}

impl Parse for Type {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        // Todo: modifiers that aren't const
        let (input, is_const) = match parse_token(Token::Const)(input) {
            Ok((input, _)) => (input, true),
            Err(_) => (input, false),
        };

        let (input, tl) = TypeLayout::parse(input, st)?;

        let tm = TypeModifier {
            is_const,
            ..TypeModifier::default()
        };
        Ok((input, Type(tl, tm)))
    }
}

impl Parse for GlobalType {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        if input.is_empty() {
            return Err(nom::Err::Incomplete(nom::Needed::new(1)));
        }
        // Interpolation modifiers unimplemented
        // Non-standard combinations of storage classes unimplemented
        let (input, gs) = match input[0] {
            LexToken(Token::Static, _) => (&input[1..], Some(GlobalStorage::Static)),
            LexToken(Token::GroupShared, _) => (&input[1..], Some(GlobalStorage::GroupShared)),
            LexToken(Token::Extern, _) => (&input[1..], Some(GlobalStorage::Extern)),
            _ => (input, None),
        };
        let (input, ty) = Type::parse(input, st)?;
        let gt = GlobalType(ty, gs.unwrap_or_default(), None);
        Ok((input, gt))
    }
}

impl Parse for InputModifier {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], _: &SymbolTable) -> ParseResult<'t, Self> {
        match input {
            [LexToken(Token::In, _), rest @ ..] => Ok((rest, InputModifier::In)),
            [LexToken(Token::Out, _), rest @ ..] => Ok((rest, InputModifier::Out)),
            [LexToken(Token::InOut, _), rest @ ..] => Ok((rest, InputModifier::InOut)),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }
}

impl Parse for ParamType {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, it) = match InputModifier::parse(input, st) {
            Ok((input, it)) => (input, it),
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
            Err(_) => (input, InputModifier::default()),
        };
        // Todo: interpolation modifiers
        match Type::parse(input, st) {
            Ok((rest, ty)) => Ok((rest, ParamType(ty, it, None))),
            Err(err) => Err(err),
        }
    }
}

impl Parse for LocalType {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        // Todo: input modifiers
        match Type::parse(input, st) {
            Ok((rest, ty)) => Ok((rest, LocalType(ty, LocalStorage::default(), None))),
            Err(err) => Err(err),
        }
    }
}

fn parse_arraydim<'t>(
    input: &'t [LexToken],
    st: &SymbolTable,
) -> ParseResult<'t, Option<Located<Expression>>> {
    let (input, _) = parse_token(Token::LeftSquareBracket)(input)?;
    let (input, constant_expression) = match ExpressionNoSeq::parse(input, st) {
        Ok((rest, constant_expression)) => (rest, Some(constant_expression)),
        _ => (input, None),
    };
    let (input, _) = parse_token(Token::RightSquareBracket)(input)?;
    Ok((input, constant_expression))
}

fn expr_paren<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    // Try to parse an expression nested in parenthesis
    {
        let res: ParseResult<'t, Located<Expression>> = (|| {
            let (input, start) = parse_token(Token::LeftParen)(input)?;
            let (input, expr) = Expression::parse(input, st)?;
            let (input, _) = parse_token(Token::RightParen)(input)?;

            Ok((input, Located::new(expr.to_node(), start.to_loc())))
        })();
        if let Ok(res) = res {
            return Ok(res);
        }
    }

    // Try to parse a variable identifier
    if let Ok((input, name)) = VariableName::parse(input, st) {
        return Ok((
            input,
            Located::new(Expression::Variable(name.node), name.location),
        ));
    }

    // Try to parse a literal
    match input.first() {
        Some(LexToken(tok, ref loc)) => {
            let literal = match *tok {
                Token::LiteralInt(v) => Literal::UntypedInt(v),
                Token::LiteralUInt(v) => Literal::UInt(v),
                Token::LiteralLong(v) => Literal::Long(v),
                Token::LiteralHalf(v) => Literal::Half(v),
                Token::LiteralFloat(v) => Literal::Float(v),
                Token::LiteralDouble(v) => Literal::Double(v),
                Token::True => Literal::Bool(true),
                Token::False => Literal::Bool(false),
                _ => {
                    return Err(nom::Err::Error(ParseErrorContext(
                        input,
                        ParseErrorReason::WrongToken,
                    )))
                }
            };
            Ok((
                &input[1..],
                Located::new(Expression::Literal(literal), loc.clone()),
            ))
        }
        None => Err(nom::Err::Incomplete(nom::Needed::new(1))),
    }
}

fn expr_p1<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    #[derive(Clone)]
    enum Precedence1Postfix {
        Increment,
        Decrement,
        Call(Vec<Located<Expression>>),
        ArraySubscript(Located<Expression>),
        Member(String),
    }

    fn expr_p1_increment(input: &[LexToken]) -> ParseResult<Located<Precedence1Postfix>> {
        // TODO: ++ tokens
        let (input, start) = parse_token(Token::Plus)(input)?;
        let (input, _) = parse_token(Token::Plus)(input)?;
        Ok((
            input,
            Located::new(Precedence1Postfix::Increment, start.to_loc()),
        ))
    }

    fn expr_p1_decrement(input: &[LexToken]) -> ParseResult<Located<Precedence1Postfix>> {
        // TODO: -- tokens
        let (input, start) = parse_token(Token::Minus)(input)?;
        let (input, _) = parse_token(Token::Minus)(input)?;
        Ok((
            input,
            Located::new(Precedence1Postfix::Decrement, start.to_loc()),
        ))
    }

    fn expr_p1_member<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, Located<Precedence1Postfix>> {
        let (input, _) = parse_token(Token::Period)(input)?;
        let (input, member) = parse_typed::<VariableName>(st)(input)?;
        Ok((
            input,
            Located::new(
                Precedence1Postfix::Member(member.node.clone()),
                member.location,
            ),
        ))
    }

    fn expr_p1_call<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, Located<Precedence1Postfix>> {
        let (input, start) = parse_token(Token::LeftParen)(input)?;

        let (input, params) = nom::multi::separated_list0(
            parse_token(Token::Comma),
            parse_typed::<ExpressionNoSeq>(st),
        )(input)?;

        let (input, _) = parse_token(Token::RightParen)(input)?;

        Ok((
            input,
            Located::new(Precedence1Postfix::Call(params), start.to_loc()),
        ))
    }

    fn expr_p1_subscript<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, Located<Precedence1Postfix>> {
        let (input, start) = parse_token(Token::LeftSquareBracket)(input)?;
        let (input, subscript) = parse_typed::<ExpressionNoSeq>(st)(input)?;
        let (input, _) = parse_token(Token::RightSquareBracket)(input)?;
        Ok((
            input,
            Located::new(
                Precedence1Postfix::ArraySubscript(subscript),
                start.to_loc(),
            ),
        ))
    }

    fn expr_p1_right<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, Located<Precedence1Postfix>> {
        nom::branch::alt((
            expr_p1_increment,
            expr_p1_decrement,
            |input| expr_p1_call(input, st),
            |input| expr_p1_member(input, st),
            |input| expr_p1_subscript(input, st),
        ))(input)
    }

    fn cons<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
        let loc = if !input.is_empty() {
            input[0].1.clone()
        } else {
            return Err(nom::Err::Incomplete(nom::Needed::new(1)));
        };
        let (input, dtyl) = DataLayout::parse(input, st)?;
        let (input, _) = parse_token(Token::LeftParen)(input)?;
        let (input, list) = nom::multi::separated_list0(
            parse_token(Token::Comma),
            parse_typed::<ExpressionNoSeq>(st),
        )(input)?;
        let (input, _) = parse_token(Token::RightParen)(input)?;
        Ok((
            input,
            Located::new(Expression::NumericConstructor(dtyl, list), loc),
        ))
    }

    let input = match cons(input, st) {
        Ok((rest, rem)) => return Ok((rest, rem)),
        Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
        Err(_) => input,
    };

    let (input, left) = expr_paren(input, st)?;
    let (input, rights) = nom::multi::many0(|input| expr_p1_right(input, st))(input)?;

    let expr = {
        let loc = left.location.clone();
        let mut final_expression = left;
        for val in rights.iter() {
            final_expression = Located::new(
                match val.node.clone() {
                    Precedence1Postfix::Increment => Expression::UnaryOperation(
                        UnaryOp::PostfixIncrement,
                        Box::new(final_expression),
                    ),
                    Precedence1Postfix::Decrement => Expression::UnaryOperation(
                        UnaryOp::PostfixDecrement,
                        Box::new(final_expression),
                    ),
                    Precedence1Postfix::Call(params) => {
                        Expression::Call(Box::new(final_expression), params)
                    }
                    Precedence1Postfix::ArraySubscript(expr) => {
                        Expression::ArraySubscript(Box::new(final_expression), Box::new(expr))
                    }
                    Precedence1Postfix::Member(name) => {
                        Expression::Member(Box::new(final_expression), name)
                    }
                },
                loc.clone(),
            )
        }
        final_expression
    };

    Ok((input, expr))
}

fn unaryop_prefix(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
    fn unaryop_increment(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
        // TODO: ++ tokens
        let (input, start) = parse_token(Token::Plus)(input)?;
        let (input, _) = parse_token(Token::Plus)(input)?;
        Ok((
            input,
            Located::new(UnaryOp::PrefixIncrement, start.to_loc()),
        ))
    }

    fn unaryop_decrement(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
        // TODO: -- tokens
        let (input, start) = parse_token(Token::Minus)(input)?;
        let (input, _) = parse_token(Token::Minus)(input)?;
        Ok((
            input,
            Located::new(UnaryOp::PrefixDecrement, start.to_loc()),
        ))
    }

    fn unaryop_add(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
        let (input, start) = parse_token(Token::Plus)(input)?;
        Ok((input, Located::new(UnaryOp::Plus, start.to_loc())))
    }

    fn unaryop_subtract(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
        let (input, start) = parse_token(Token::Minus)(input)?;
        Ok((input, Located::new(UnaryOp::Minus, start.to_loc())))
    }

    fn unaryop_logical_not(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
        let (input, start) = parse_token(Token::ExclamationPoint)(input)?;
        Ok((input, Located::new(UnaryOp::LogicalNot, start.to_loc())))
    }

    fn unaryop_bitwise_not(input: &[LexToken]) -> ParseResult<Located<UnaryOp>> {
        let (input, start) = parse_token(Token::Tilde)(input)?;
        Ok((input, Located::new(UnaryOp::BitwiseNot, start.to_loc())))
    }

    nom::branch::alt((
        unaryop_increment,
        unaryop_decrement,
        unaryop_add,
        unaryop_subtract,
        unaryop_logical_not,
        unaryop_bitwise_not,
    ))(input)
}

fn expr_p2<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn expr_p2_unaryop<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, Located<Expression>> {
        let (input, unary) = unaryop_prefix(input)?;
        let (input, expr) = expr_p2(input, st)?;
        Ok((
            input,
            Located::new(
                Expression::UnaryOperation(unary.node.clone(), Box::new(expr)),
                unary.location,
            ),
        ))
    }

    fn expr_p2_cast<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, Located<Expression>> {
        let (input, start) = parse_token(Token::LeftParen)(input)?;
        let (input, cast) = parse_typed::<Type>(st)(input)?;
        let (input, _) = parse_token(Token::RightParen)(input)?;
        let (input, expr) = expr_p2(input, st)?;
        Ok((
            input,
            Located::new(Expression::Cast(cast, Box::new(expr)), start.to_loc()),
        ))
    }

    nom::branch::alt((
        |input| expr_p2_unaryop(input, st),
        |input| expr_p2_cast(input, st),
        |input| expr_p1(input, st),
    ))(input)
}

/// Combine binary operations
fn combine_rights(
    left: Located<Expression>,
    rights: Vec<(BinOp, Located<Expression>)>,
) -> Located<Expression> {
    let loc = left.location.clone();
    let mut final_expression = left;
    for val in rights.iter() {
        let (ref op, ref exp) = *val;
        final_expression = Located::new(
            Expression::BinaryOperation(
                op.clone(),
                Box::new(final_expression),
                Box::new(exp.clone()),
            ),
            loc.clone(),
        )
    }
    final_expression
}

/// Parse multiple binary operations
fn parse_binary_operations<'t>(
    operator_fn: impl Fn(&'t [LexToken]) -> ParseResult<'t, BinOp>,
    expression_fn: impl Fn(&'t [LexToken], &SymbolTable) -> ParseResult<'t, Located<Expression>>,
    input: &'t [LexToken],
    st: &SymbolTable,
) -> ParseResult<'t, Located<Expression>> {
    let (input, left) = expression_fn(input, st)?;
    let (input, rights) = nom::multi::many0(nom::sequence::tuple((operator_fn, |input| {
        expression_fn(input, st)
    })))(input)?;
    let expr = combine_rights(left, rights);
    Ok((input, expr))
}

fn expr_p3<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input.first() {
            Some(LexToken(tok, _)) => {
                let op = match *tok {
                    Token::Asterix => BinOp::Multiply,
                    Token::ForwardSlash => BinOp::Divide,
                    Token::Percent => BinOp::Modulus,
                    _ => {
                        return Err(nom::Err::Error(ParseErrorContext(
                            input,
                            ParseErrorReason::WrongToken,
                        )))
                    }
                };
                Ok((&input[1..], op))
            }
            None => Err(nom::Err::Incomplete(nom::Needed::new(1))),
        }
    }

    parse_binary_operations(parse_op, expr_p2, input, st)
}

fn expr_p4<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input.first() {
            Some(LexToken(tok, _)) => {
                let op = match *tok {
                    Token::Plus => BinOp::Add,
                    Token::Minus => BinOp::Subtract,
                    _ => {
                        return Err(nom::Err::Error(ParseErrorContext(
                            input,
                            ParseErrorReason::WrongToken,
                        )))
                    }
                };
                Ok((&input[1..], op))
            }
            None => Err(nom::Err::Incomplete(nom::Needed::new(1))),
        }
    }

    parse_binary_operations(parse_op, expr_p3, input, st)
}

fn expr_p5<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::LeftAngleBracket(FollowedBy::Token), _), LexToken(Token::LeftAngleBracket(_), _), rest @ ..] => {
                Ok((rest, BinOp::LeftShift))
            }
            [LexToken(Token::RightAngleBracket(FollowedBy::Token), _), LexToken(Token::RightAngleBracket(_), _), rest @ ..] => {
                Ok((rest, BinOp::RightShift))
            }
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p4, input, st)
}

fn expr_p6<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::LeftAngleBracket(FollowedBy::Token), _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::LessEqual))
            }
            [LexToken(Token::RightAngleBracket(FollowedBy::Token), _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::GreaterEqual))
            }
            [LexToken(Token::LeftAngleBracket(_), _), rest @ ..] => Ok((rest, BinOp::LessThan)),
            [LexToken(Token::RightAngleBracket(_), _), rest @ ..] => Ok((rest, BinOp::GreaterThan)),
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p5, input, st)
}

fn expr_p7<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::DoubleEquals, _), rest @ ..] => Ok((rest, BinOp::Equality)),
            [LexToken(Token::ExclamationEquals, _), rest @ ..] => Ok((rest, BinOp::Inequality)),
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p6, input, st)
}

fn expr_p8<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::Ampersand(_), _), rest @ ..] => Ok((rest, BinOp::BitwiseAnd)),
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p7, input, st)
}

fn expr_p9<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::Hat, _), rest @ ..] => Ok((rest, BinOp::BitwiseXor)),
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p8, input, st)
}

fn expr_p10<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::VerticalBar(_), _), rest @ ..] => Ok((rest, BinOp::BitwiseOr)),
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p9, input, st)
}

fn expr_p11<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::Ampersand(FollowedBy::Token), _), LexToken(Token::Ampersand(_), _), rest @ ..] => {
                Ok((rest, BinOp::BooleanAnd))
            }
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p10, input, st)
}

fn expr_p12<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::VerticalBar(FollowedBy::Token), _), LexToken(Token::VerticalBar(_), _), rest @ ..] => {
                Ok((rest, BinOp::BooleanOr))
            }
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p11, input, st)
}

fn expr_p13<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn ternary_right<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, (Located<Expression>, Located<Expression>)> {
        let (input, _) = parse_token(Token::QuestionMark)(input)?;
        let (input, left) = expr_p13(input, st)?;
        let (input, _) = parse_token(Token::Colon)(input)?;
        let (input, right) = expr_p13(input, st)?;
        Ok((input, (left, right)))
    }

    let (input, main) = expr_p12(input, st)?;
    match ternary_right(input, st) {
        Ok((input, (left, right))) => {
            let loc = main.location.clone();
            let expr = Located::new(
                Expression::TernaryConditional(Box::new(main), Box::new(left), Box::new(right)),
                loc,
            );
            Ok((input, expr))
        }
        _ => Ok((input, main)),
    }
}

fn expr_p14<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::Equals, _), rest @ ..] => Ok((rest, BinOp::Assignment)),
            [LexToken(Token::Plus, _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::SumAssignment))
            }
            [LexToken(Token::Minus, _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::DifferenceAssignment))
            }
            [LexToken(Token::Asterix, _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::ProductAssignment))
            }
            [LexToken(Token::ForwardSlash, _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::QuotientAssignment))
            }
            [LexToken(Token::Percent, _), LexToken(Token::Equals, _), rest @ ..] => {
                Ok((rest, BinOp::RemainderAssignment))
            }
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    fn binary_right<'t>(
        input: &'t [LexToken],
        st: &SymbolTable,
    ) -> ParseResult<'t, (BinOp, Located<Expression>)> {
        let (input, op) = parse_op(input)?;
        let (input, rhs) = expr_p14(input, st)?;
        Ok((input, (op, rhs)))
    }

    let (input, main) = expr_p13(input, st)?;
    match binary_right(input, st) {
        Ok((input, (op, right))) => {
            let loc = main.location.clone();
            let expr = Located::new(
                Expression::BinaryOperation(op, Box::new(main), Box::new(right)),
                loc,
            );
            Ok((input, expr))
        }
        _ => Ok((input, main)),
    }
}

fn expr_p15<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Located<Expression>> {
    fn parse_op(input: &[LexToken]) -> ParseResult<BinOp> {
        match input {
            [LexToken(Token::Comma, _), rest @ ..] => Ok((rest, BinOp::Sequence)),
            [] => Err(nom::Err::Incomplete(nom::Needed::new(1))),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }

    parse_binary_operations(parse_op, expr_p14, input, st)
}

impl Parse for Expression {
    type Output = Located<Self>;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self::Output> {
        expr_p15(input, st)
    }
}

#[test]
fn test_expression() {
    let st = SymbolTable::empty();
    let a_id = Identifier("a".to_string());
    let b_id = Identifier("b".to_string());
    let c_id = Identifier("c".to_string());
    let d_id = Identifier("d".to_string());
    let e_id = Identifier("e".to_string());
    let f_id = Identifier("f".to_string());
    let g_id = Identifier("g".to_string());

    fn loc(i: u64) -> FileLocation {
        FileLocation::Known(FileName(String::new()), Line(i), Column(i))
    }

    // Test `a + b + c` is `((a + b) + c)`
    let add_chain_tokens = &[
        LexToken(Token::Id(a_id.clone()), loc(0)),
        LexToken(Token::Plus, FileLocation::Unknown),
        LexToken(Token::Id(b_id.clone()), loc(1)),
        LexToken(Token::Plus, FileLocation::Unknown),
        LexToken(Token::Id(c_id.clone()), loc(2)),
        LexToken(Token::Semicolon, FileLocation::Unknown),
    ];
    let add_chain_expr = Expression::parse(add_chain_tokens, &st);
    assert_eq!(
        add_chain_expr,
        Ok((
            &[LexToken(Token::Semicolon, FileLocation::Unknown)][..],
            Located::new(
                Expression::BinaryOperation(
                    BinOp::Add,
                    Box::new(Located::new(
                        Expression::BinaryOperation(
                            BinOp::Add,
                            Box::new(Located::new(Expression::Variable(a_id.0.clone()), loc(0))),
                            Box::new(Located::new(Expression::Variable(b_id.0.clone()), loc(1))),
                        ),
                        loc(0)
                    )),
                    Box::new(Located::new(Expression::Variable(c_id.0.clone()), loc(2))),
                ),
                loc(0)
            )
        ))
    );

    // Test `a, b, c` is `((a, b), c)`
    let comma_chain_tokens = &[
        LexToken(Token::Id(a_id.clone()), loc(0)),
        LexToken(Token::Comma, FileLocation::Unknown),
        LexToken(Token::Id(b_id.clone()), loc(1)),
        LexToken(Token::Comma, FileLocation::Unknown),
        LexToken(Token::Id(c_id.clone()), loc(2)),
        LexToken(Token::Semicolon, FileLocation::Unknown),
    ];
    let comma_chain_expr = Expression::parse(comma_chain_tokens, &st);
    assert_eq!(
        comma_chain_expr,
        Ok((
            &[LexToken(Token::Semicolon, FileLocation::Unknown)][..],
            Located::new(
                Expression::BinaryOperation(
                    BinOp::Sequence,
                    Box::new(Located::new(
                        Expression::BinaryOperation(
                            BinOp::Sequence,
                            Box::new(Located::new(Expression::Variable(a_id.0.clone()), loc(0))),
                            Box::new(Located::new(Expression::Variable(b_id.0.clone()), loc(1))),
                        ),
                        loc(0)
                    )),
                    Box::new(Located::new(Expression::Variable(c_id.0.clone()), loc(2))),
                ),
                loc(0)
            )
        ))
    );

    // Test `a ? b ? c : d : e ? f : g` is `a ? (b ? c : d) : (e ? f : g)`
    let nested_ternary_tokens = &[
        LexToken(Token::Id(a_id.clone()), loc(0)),
        LexToken(Token::QuestionMark, FileLocation::Unknown),
        LexToken(Token::Id(b_id.clone()), loc(1)),
        LexToken(Token::QuestionMark, FileLocation::Unknown),
        LexToken(Token::Id(c_id.clone()), loc(2)),
        LexToken(Token::Colon, FileLocation::Unknown),
        LexToken(Token::Id(d_id.clone()), loc(3)),
        LexToken(Token::Colon, FileLocation::Unknown),
        LexToken(Token::Id(e_id.clone()), loc(4)),
        LexToken(Token::QuestionMark, FileLocation::Unknown),
        LexToken(Token::Id(f_id.clone()), loc(5)),
        LexToken(Token::Colon, FileLocation::Unknown),
        LexToken(Token::Id(g_id.clone()), loc(6)),
        LexToken(Token::Semicolon, FileLocation::Unknown),
    ];
    let comma_chain_expr = Expression::parse(nested_ternary_tokens, &st);
    assert_eq!(
        comma_chain_expr,
        Ok((
            &[LexToken(Token::Semicolon, FileLocation::Unknown)][..],
            Located::new(
                Expression::TernaryConditional(
                    Box::new(Located::new(Expression::Variable(a_id.0), loc(0))),
                    Box::new(Located::new(
                        Expression::TernaryConditional(
                            Box::new(Located::new(Expression::Variable(b_id.0), loc(1))),
                            Box::new(Located::new(Expression::Variable(c_id.0), loc(2))),
                            Box::new(Located::new(Expression::Variable(d_id.0), loc(3))),
                        ),
                        loc(1)
                    )),
                    Box::new(Located::new(
                        Expression::TernaryConditional(
                            Box::new(Located::new(Expression::Variable(e_id.0), loc(4))),
                            Box::new(Located::new(Expression::Variable(f_id.0), loc(5))),
                            Box::new(Located::new(Expression::Variable(g_id.0), loc(6))),
                        ),
                        loc(4)
                    )),
                ),
                loc(0)
            )
        ))
    );
}

/// Fake node for parsing an expression where the comma has a different meaning
/// at the top level, so skip that node
struct ExpressionNoSeq;

impl Parse for ExpressionNoSeq {
    type Output = Located<Expression>;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self::Output> {
        expr_p14(input, st)
    }
}

impl Parse for Initializer {
    type Output = Option<Initializer>;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self::Output> {
        fn init_expr<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Initializer> {
            let (input, expr) = parse_typed::<ExpressionNoSeq>(st)(input)?;
            Ok((input, Initializer::Expression(expr)))
        }

        fn init_aggregate<'t>(
            input: &'t [LexToken],
            st: &SymbolTable,
        ) -> ParseResult<'t, Initializer> {
            let (input, _) = parse_token(Token::LeftBrace)(input)?;
            let (input, exprs) = nom::multi::separated_list1(parse_token(Token::Comma), |input| {
                init_any(input, st)
            })(input)?;
            let (input, _) = parse_token(Token::RightBrace)(input)?;
            Ok((input, Initializer::Aggregate(exprs)))
        }

        fn init_any<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Initializer> {
            if let Ok((input, expr)) = init_expr(input, st) {
                return Ok((input, expr));
            }

            if let Ok((input, expr)) = init_aggregate(input, st) {
                return Ok((input, expr));
            }

            Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::ErrorKind(nom::error::ErrorKind::Alt),
            )))
        }

        if input.is_empty() {
            Err(nom::Err::Incomplete(nom::Needed::new(1)))
        } else {
            match input[0].0 {
                Token::Equals => match init_any(&input[1..], st) {
                    Ok((input, init)) => Ok((input, Some(init))),
                    Err(err) => Err(err),
                },
                _ => Ok((input, None)),
            }
        }
    }
}

#[test]
fn test_initializer() {
    fn initializer<'t>(input: &'t [LexToken]) -> ParseResult<'t, Option<Initializer>> {
        let st = SymbolTable::empty();
        Initializer::parse(input, &st)
    }

    assert_eq!(
        initializer(&[]),
        Err(nom::Err::Incomplete(nom::Needed::new(1)))
    );

    // Semicolon to trigger parsing to end
    let semicolon = LexToken::with_no_loc(Token::Semicolon);
    let done_toks = &[semicolon.clone()][..];

    // No initializer tests
    assert_eq!(initializer(&[semicolon.clone()]), Ok((done_toks, None)));

    // Expression initialization tests
    // = [expr]
    let expr_lit = [
        LexToken::with_no_loc(Token::Equals),
        LexToken::with_no_loc(Token::LiteralInt(4)),
        semicolon.clone(),
    ];
    let hst_lit = Located::none(Expression::Literal(Literal::UntypedInt(4)));
    assert_eq!(
        initializer(&expr_lit),
        Ok((done_toks, Some(Initializer::Expression(hst_lit))))
    );

    // Aggregate initialization tests
    // = { [expr], [expr], [expr] }
    fn loc_lit(i: u64) -> Initializer {
        Initializer::Expression(Located::none(Expression::Literal(Literal::UntypedInt(i))))
    }

    let aggr_1 = [
        LexToken::with_no_loc(Token::Equals),
        LexToken::with_no_loc(Token::LeftBrace),
        LexToken::with_no_loc(Token::LiteralInt(4)),
        LexToken::with_no_loc(Token::RightBrace),
        semicolon.clone(),
    ];
    let aggr_1_lit = loc_lit(4);
    assert_eq!(
        initializer(&aggr_1),
        Ok((done_toks, Some(Initializer::Aggregate(vec![aggr_1_lit]))))
    );

    let aggr_3 = [
        LexToken::with_no_loc(Token::Equals),
        LexToken::with_no_loc(Token::LeftBrace),
        LexToken::with_no_loc(Token::LiteralInt(4)),
        LexToken::with_no_loc(Token::Comma),
        LexToken::with_no_loc(Token::LiteralInt(2)),
        LexToken::with_no_loc(Token::Comma),
        LexToken::with_no_loc(Token::LiteralInt(1)),
        LexToken::with_no_loc(Token::RightBrace),
        semicolon.clone(),
    ];
    let aggr_3_lits = vec![loc_lit(4), loc_lit(2), loc_lit(1)];
    assert_eq!(
        initializer(&aggr_3),
        Ok((done_toks, Some(Initializer::Aggregate(aggr_3_lits))))
    );

    // = {} should fail
    let aggr_0 = [
        LexToken::with_no_loc(Token::Equals),
        LexToken::with_no_loc(Token::LeftBrace),
        LexToken::with_no_loc(Token::RightBrace),
        semicolon,
    ];
    assert!(match initializer(&aggr_0) {
        Err(_) => true,
        _ => false,
    });
}

impl Parse for VarDef {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, typename) = parse_typed::<LocalType>(st)(input)?;
        let (input, defs) = nom::multi::separated_list1(parse_token(Token::Comma), |input| {
            let (input, varname) = parse_typed::<VariableName>(st)(input)?;
            let (input, array_dim) =
                nom::combinator::opt(|input| parse_arraydim(input, st))(input)?;
            let (input, init) = parse_typed::<Initializer>(st)(input)?;
            let v = LocalVariableName {
                name: varname.to_node(),
                bind: match array_dim {
                    Some(ref expr) => VariableBind::Array(expr.clone()),
                    None => VariableBind::Normal,
                },
                init,
            };
            Ok((input, v))
        })(input)?;
        let defs = VarDef {
            local_type: typename,
            defs,
        };
        Ok((input, defs))
    }
}

impl Parse for InitStatement {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let res = nom::branch::alt((
            nom::combinator::map(parse_typed::<VarDef>(st), |vd| {
                InitStatement::Declaration(vd)
            }),
            nom::combinator::map(parse_typed::<Expression>(st), |e| {
                InitStatement::Expression(e)
            }),
        ))(input);
        match res {
            Ok((rest, res)) => Ok((rest, res)),
            Err(nom::Err::Incomplete(needed)) => Err(nom::Err::Incomplete(needed)),
            Err(_) => Ok((input, InitStatement::Empty)),
        }
    }
}

fn statement_attribute<'t>(input: &'t [LexToken], _: &SymbolTable) -> ParseResult<'t, ()> {
    let (input, _) = parse_token(Token::LeftSquareBracket)(input)?;
    let (input, _) = match_identifier(input)?;

    // Currently do not support arguments to attributes

    let (input, _) = parse_token(Token::RightSquareBracket)(input)?;
    Ok((input, ()))
}

#[test]
fn test_statement_attribute() {
    let st = SymbolTable::empty();
    let fastopt = &[
        LexToken::with_no_loc(Token::LeftSquareBracket),
        LexToken::with_no_loc(Token::Id(Identifier("fastopt".to_string()))),
        LexToken::with_no_loc(Token::RightSquareBracket),
    ];
    assert_eq!(statement_attribute(fastopt, &st), Ok((&[][..], ())));
}

impl Parse for Statement {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        // Parse and ignore attributes before a statement
        let input = match nom::multi::many0(|input| statement_attribute(input, st))(input) {
            Ok((rest, _)) => rest,
            Err(err) => return Err(err),
        };
        if input.is_empty() {
            return Err(nom::Err::Incomplete(nom::Needed::new(1)));
        }
        let (head, tail) = (input[0].clone(), &input[1..]);
        match head {
            LexToken(Token::Semicolon, _) => Ok((tail, Statement::Empty)),
            LexToken(Token::If, _) => {
                let (input, _) = parse_token(Token::LeftParen)(tail)?;
                let (input, cond) = parse_typed::<Expression>(st)(input)?;
                let (input, _) = parse_token(Token::RightParen)(input)?;
                let (input, inner_statement) = parse_typed::<Statement>(st)(input)?;
                let inner_statement = Box::new(inner_statement);
                if input.is_empty() {
                    return Err(nom::Err::Incomplete(nom::Needed::new(1)));
                }
                let (head, tail) = (input[0].clone(), &input[1..]);
                match head {
                    LexToken(Token::Else, _) => match Statement::parse(tail, st) {
                        Err(err) => Err(err),
                        Ok((tail, else_part)) => {
                            let s = Statement::IfElse(cond, inner_statement, Box::new(else_part));
                            Ok((tail, s))
                        }
                    },
                    _ => Ok((input, Statement::If(cond, inner_statement))),
                }
            }
            LexToken(Token::For, _) => {
                let (input, _) = parse_token(Token::LeftParen)(tail)?;
                let (input, init) = parse_typed::<InitStatement>(st)(input)?;
                let (input, _) = parse_token(Token::Semicolon)(input)?;
                let (input, cond) = parse_typed::<Expression>(st)(input)?;
                let (input, _) = parse_token(Token::Semicolon)(input)?;
                let (input, inc) = parse_typed::<Expression>(st)(input)?;
                let (input, _) = parse_token(Token::RightParen)(input)?;
                let (input, inner) = parse_typed::<Statement>(st)(input)?;
                Ok((input, Statement::For(init, cond, inc, Box::new(inner))))
            }
            LexToken(Token::While, _) => {
                let (input, _) = parse_token(Token::LeftParen)(tail)?;
                let (input, cond) = parse_typed::<Expression>(st)(input)?;
                let (input, _) = parse_token(Token::RightParen)(input)?;
                let (input, inner) = parse_typed::<Statement>(st)(input)?;
                Ok((input, Statement::While(cond, Box::new(inner))))
            }
            LexToken(Token::Break, _) => Ok((tail, Statement::Break)),
            LexToken(Token::Continue, _) => Ok((tail, Statement::Continue)),
            LexToken(Token::Return, _) => {
                let (input, expression_statement) = parse_typed::<Expression>(st)(tail)?;
                let (input, _) = parse_token(Token::Semicolon)(input)?;
                Ok((input, Statement::Return(expression_statement)))
            }
            LexToken(Token::LeftBrace, _) => {
                let (input, s) = statement_block(input, st)?;
                Ok((input, Statement::Block(s)))
            }
            _ => {
                // Try parsing a variable definition
                fn variable_def<'t>(
                    input: &'t [LexToken],
                    st: &SymbolTable,
                ) -> ParseResult<'t, Statement> {
                    let (input, var) = parse_typed::<VarDef>(st)(input)?;
                    let (input, _) = parse_token(Token::Semicolon)(input)?;
                    Ok((input, Statement::Var(var)))
                }
                let err = match variable_def(input, st) {
                    Ok((rest, statement)) => return Ok((rest, statement)),
                    Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
                    Err(e) => e,
                };
                // Try parsing an expression statement
                fn expr_statement<'t>(
                    input: &'t [LexToken],
                    st: &SymbolTable,
                ) -> ParseResult<'t, Statement> {
                    let (input, expression_statement) = parse_typed::<Expression>(st)(input)?;
                    let (input, _) = parse_token(Token::Semicolon)(input)?;
                    Ok((input, Statement::Expression(expression_statement)))
                }
                let err = match expr_statement(input, st) {
                    Ok((rest, statement)) => return Ok((rest, statement)),
                    Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
                    Err(e) => get_most_relevant_error(err, e),
                };
                // Return the most likely error
                Err(err)
            }
        }
    }
}

fn statement_block<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Vec<Statement>> {
    let mut statements = Vec::new();
    let mut rest = match parse_token(Token::LeftBrace)(input) {
        Ok((rest, _)) => rest,
        Err(err) => return Err(err),
    };
    loop {
        let last_def = Statement::parse(rest, st);
        if let Ok((remaining, root)) = last_def {
            statements.push(root);
            rest = remaining;
        } else {
            return match parse_token(Token::RightBrace)(rest) {
                Ok((rest, _)) => Ok((rest, statements)),
                Err(nom::Err::Incomplete(needed)) => Err(nom::Err::Incomplete(needed)),
                Err(_) => match last_def {
                    Ok(_) => unreachable!(),
                    Err(err) => Err(err),
                },
            };
        }
    }
}

impl Parse for StructMemberName {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, name) = parse_typed::<VariableName>(st)(input)?;
        let (input, array_dim) = match parse_arraydim(input, st) {
            Ok((input, array_dim)) => (input, Some(array_dim)),
            Err(_) => (input, None),
        };
        let member_name = StructMemberName {
            name: name.to_node(),
            bind: match array_dim {
                Some(ref expr) => VariableBind::Array(expr.clone()),
                None => VariableBind::Normal,
            },
        };
        Ok((input, member_name))
    }
}

impl Parse for StructMember {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, typename) = parse_typed::<Type>(st)(input)?;
        let (input, defs) = nom::multi::separated_list1(
            parse_token(Token::Comma),
            parse_typed::<StructMemberName>(st),
        )(input)?;
        let (input, _) = parse_token(Token::Semicolon)(input)?;
        let sm = StructMember { ty: typename, defs };
        Ok((input, sm))
    }
}

impl Parse for StructDefinition {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, _) = parse_token(Token::Struct)(input)?;
        let (input, structname) = parse_typed::<VariableName>(st)(input)?;
        let (input, _) = parse_token(Token::LeftBrace)(input)?;
        let (input, members) = nom::multi::many0(parse_typed::<StructMember>(st))(input)?;
        let (input, _) = parse_token(Token::RightBrace)(input)?;
        let (input, _) = parse_token(Token::Semicolon)(input)?;
        let sd = StructDefinition {
            name: structname.to_node(),
            members,
        };
        Ok((input, sd))
    }
}

impl Parse for ConstantVariableName {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, name) = parse_typed::<VariableName>(st)(input)?;
        let (input, array_dim) = nom::combinator::opt(|input| parse_arraydim(input, st))(input)?;
        let v = ConstantVariableName {
            name: name.to_node(),
            bind: match array_dim {
                Some(ref expr) => VariableBind::Array(expr.clone()),
                None => VariableBind::Normal,
            },
            offset: None,
        };
        Ok((input, v))
    }
}

impl Parse for ConstantVariable {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, typename) = parse_typed::<Type>(st)(input)?;
        let (input, defs) = nom::multi::separated_list1(
            parse_token(Token::Comma),
            parse_typed::<ConstantVariableName>(st),
        )(input)?;
        let (input, _) = parse_token(Token::Semicolon)(input)?;
        let var = ConstantVariable { ty: typename, defs };
        Ok((input, var))
    }
}

impl Parse for ConstantSlot {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], _: &SymbolTable) -> ParseResult<'t, Self> {
        match input {
            [LexToken(Token::Colon, _), LexToken(Token::Register(reg), _), rest @ ..] => match *reg
            {
                RegisterSlot::B(slot) => Ok((rest, ConstantSlot(slot))),
                _ => Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::WrongSlotType,
                ))),
            },
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }
}

impl Parse for ConstantBuffer {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, _) = parse_token(Token::ConstantBuffer)(input)?;
        let (input, name) = parse_typed::<VariableName>(st)(input)?;
        let (input, slot) = nom::combinator::opt(parse_typed::<ConstantSlot>(st))(input)?;
        let (input, members) = nom::sequence::delimited(
            parse_token(Token::LeftBrace),
            nom::multi::many0(parse_typed::<ConstantVariable>(st)),
            parse_token(Token::RightBrace),
        )(input)?;
        let cb = ConstantBuffer {
            name: name.to_node(),
            slot,
            members,
        };
        Ok((input, cb))
    }
}

impl Parse for GlobalSlot {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], _: &SymbolTable) -> ParseResult<'t, Self> {
        match input {
            [LexToken(Token::Colon, _), LexToken(Token::Register(reg), _), rest @ ..] => match *reg
            {
                RegisterSlot::T(slot) => Ok((rest, GlobalSlot::ReadSlot(slot))),
                RegisterSlot::U(slot) => Ok((rest, GlobalSlot::ReadWriteSlot(slot))),
                RegisterSlot::S(slot) => Ok((rest, GlobalSlot::SamplerSlot(slot))),
                _ => Err(nom::Err::Error(ParseErrorContext(
                    input,
                    ParseErrorReason::WrongSlotType,
                ))),
            },
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::WrongToken,
            ))),
        }
    }
}

impl Parse for GlobalVariableName {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, name) = parse_typed::<VariableName>(st)(input)?;
        let (input, array_dim) = nom::combinator::opt(|input| parse_arraydim(input, st))(input)?;
        let (input, slot) = nom::combinator::opt(parse_typed::<GlobalSlot>(st))(input)?;
        let (input, init) = parse_typed::<Initializer>(st)(input)?;
        let v = GlobalVariableName {
            name: name.to_node(),
            bind: match array_dim {
                Some(ref expr) => VariableBind::Array(expr.clone()),
                None => VariableBind::Normal,
            },
            slot,
            init,
        };
        Ok((input, v))
    }
}

impl Parse for GlobalVariable {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, typename) = parse_typed::<GlobalType>(st)(input)?;
        let (input, defs) = nom::multi::separated_list1(
            parse_token(Token::Comma),
            parse_typed::<GlobalVariableName>(st),
        )(input)?;
        let (input, _) = parse_token(Token::Semicolon)(input)?;
        let var = GlobalVariable {
            global_type: typename,
            defs,
        };
        Ok((input, var))
    }
}

fn parse_numthreads(input: &[LexToken]) -> ParseResult<()> {
    match input.first() {
        Some(LexToken(Token::Id(Identifier(name)), _)) => match &name[..] {
            "numthreads" => Ok((&input[1..], ())),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::UnexpectedAttribute(name.clone()),
            ))),
        },
        _ => Err(nom::Err::Error(ParseErrorContext(
            input,
            ParseErrorReason::WrongToken,
        ))),
    }
}

impl Parse for FunctionAttribute {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, _) = parse_token(Token::LeftSquareBracket)(input)?;

        // Only currently support [numthreads]
        let (input, _) = parse_numthreads(input)?;
        let (input, _) = parse_token(Token::LeftParen)(input)?;
        let (input, x) = parse_typed::<ExpressionNoSeq>(st)(input)?;
        let (input, _) = parse_token(Token::Comma)(input)?;
        let (input, y) = parse_typed::<ExpressionNoSeq>(st)(input)?;
        let (input, _) = parse_token(Token::Comma)(input)?;
        let (input, z) = parse_typed::<ExpressionNoSeq>(st)(input)?;
        let (input, _) = parse_token(Token::RightParen)(input)?;
        let attr = FunctionAttribute::NumThreads(x, y, z);

        let (input, _) = parse_token(Token::RightSquareBracket)(input)?;
        Ok((input, attr))
    }
}

fn parse_semantic(input: &[LexToken]) -> ParseResult<Semantic> {
    match input.first() {
        Some(LexToken(Token::Id(Identifier(name)), _)) => match &name[..] {
            "SV_DispatchThreadID" => Ok((&input[1..], Semantic::DispatchThreadId)),
            "SV_GroupID" => Ok((&input[1..], Semantic::GroupId)),
            "SV_GroupIndex" => Ok((&input[1..], Semantic::GroupIndex)),
            "SV_GroupThreadID" => Ok((&input[1..], Semantic::GroupThreadId)),
            _ => Err(nom::Err::Error(ParseErrorContext(
                input,
                ParseErrorReason::UnexpectedAttribute(name.clone()),
            ))),
        },
        _ => Err(nom::Err::Error(ParseErrorContext(
            input,
            ParseErrorReason::WrongToken,
        ))),
    }
}

impl Parse for FunctionParam {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, ty) = parse_typed::<ParamType>(st)(input)?;
        let (input, param) = parse_typed::<VariableName>(st)(input)?;

        // Parse semantic if present
        let (input, semantic) = match parse_token(Token::Colon)(input) {
            Ok((input, _)) => match parse_semantic(input) {
                Ok((input, semantic)) => (input, Some(semantic)),
                Err(err) => return Err(err),
            },
            Err(_) => (input, None),
        };

        let param = FunctionParam {
            name: param.to_node(),
            param_type: ty,
            semantic,
        };
        Ok((input, param))
    }
}

impl Parse for FunctionDefinition {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let (input, attributes) = nom::multi::many0(parse_typed::<FunctionAttribute>(st))(input)?;
        let (input, ret) = parse_typed::<Type>(st)(input)?;
        let (input, func_name) = parse_typed::<VariableName>(st)(input)?;
        let (input, params) = nom::sequence::delimited(
            parse_token(Token::LeftParen),
            nom::multi::separated_list0(
                parse_token(Token::Comma),
                parse_typed::<FunctionParam>(st),
            ),
            parse_token(Token::RightParen),
        )(input)?;
        let (input, body) = statement_block(input, st)?;
        let def = FunctionDefinition {
            name: func_name.to_node(),
            returntype: ret,
            params,
            body,
            attributes,
        };
        Ok((input, def))
    }
}

impl Parse for RootDefinition {
    type Output = Self;
    fn parse<'t>(input: &'t [LexToken], st: &SymbolTable) -> ParseResult<'t, Self> {
        let err = match StructDefinition::parse(input, st) {
            Ok((rest, structdef)) => return Ok((rest, RootDefinition::Struct(structdef))),
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
            Err(e) => e,
        };

        let err = match ConstantBuffer::parse(input, st) {
            Ok((rest, cbuffer)) => return Ok((rest, RootDefinition::ConstantBuffer(cbuffer))),
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
            Err(e) => get_most_relevant_error(err, e),
        };

        let err = match GlobalVariable::parse(input, st) {
            Ok((rest, globalvariable)) => {
                return Ok((rest, RootDefinition::GlobalVariable(globalvariable)))
            }
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
            Err(e) => get_most_relevant_error(err, e),
        };

        let err = match FunctionDefinition::parse(input, st) {
            Ok((rest, funcdef)) => return Ok((rest, RootDefinition::Function(funcdef))),
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
            Err(e) => get_most_relevant_error(err, e),
        };

        Err(err)
    }
}

fn rootdefinition_with_semicolon<'t>(
    input: &'t [LexToken],
    st: &SymbolTable,
) -> ParseResult<'t, RootDefinition> {
    let (input, def) = parse_typed::<RootDefinition>(st)(input)?;
    let (input, _) = nom::multi::many0(parse_token(Token::Semicolon))(input)?;
    Ok((input, def))
}

// Find the error with the longest tokens used
fn get_most_relevant_error<'a: 'c, 'b: 'c, 'c>(
    lhs: nom::Err<ParseErrorContext<'a>>,
    _: nom::Err<ParseErrorContext<'b>>,
) -> nom::Err<ParseErrorContext<'c>> {
    lhs
}

fn module(input: &[LexToken]) -> ParseResult<Vec<RootDefinition>> {
    let mut roots = Vec::new();
    let mut rest = input;
    let mut symbol_table = SymbolTable::empty();
    loop {
        let last_def = rootdefinition_with_semicolon(rest, &symbol_table);
        if let Ok((remaining, root)) = last_def {
            match root {
                RootDefinition::Struct(ref sd) => {
                    match symbol_table.0.insert(sd.name.clone(), SymbolType::Struct) {
                        Some(_) => {
                            let reason = ParseErrorReason::DuplicateStructSymbol;
                            return Err(nom::Err::Error(ParseErrorContext(input, reason)));
                        }
                        None => {}
                    }
                }
                _ => {}
            }
            roots.push(root);
            rest = remaining;
        } else {
            return match rest {
                a if a.len() == 1 && a[0].0 == Token::Eof => Ok((&[], roots)),
                _ => match last_def {
                    Ok(_) => unreachable!(),
                    Err(err) => Err(err),
                },
            };
        }
    }
}

pub fn parse(entry_point: String, source: &[LexToken]) -> Result<Module, ParseError> {
    let parse_result = module(source);

    match parse_result {
        Ok((rest, _)) if !rest.is_empty() => Err(ParseError(
            ParseErrorReason::FailedToParse,
            Some(rest.to_vec()),
            None,
        )),
        Ok((_, hlsl)) => Ok(Module {
            entry_point,
            root_definitions: hlsl,
        }),
        Err(nom::Err::Error(ParseErrorContext(_, err))) => Err(ParseError(err, None, None)),
        Err(nom::Err::Failure(ParseErrorContext(_, err))) => Err(ParseError(err, None, None)),
        Err(nom::Err::Incomplete(_)) => Err(ParseError(
            ParseErrorReason::UnexpectedEndOfStream,
            None,
            None,
        )),
    }
}

#[cfg(test)]
fn exp_var(var_name: &'static str, line: u64, column: u64) -> Located<Expression> {
    test_loc(line, column, Expression::Variable(var_name.to_string()))
}
#[cfg(test)]
fn bexp_var(var_name: &'static str, line: u64, column: u64) -> Box<Located<Expression>> {
    Box::new(test_loc(
        line,
        column,
        Expression::Variable(var_name.to_string()),
    ))
}

#[cfg(test)]
fn node<T>(input: &[LexToken]) -> ParseResult<<T as Parse>::Output>
where
    T: Parse,
{
    let st = SymbolTable::empty();
    T::parse(input, &st)
}

#[cfg(test)]
fn parse_result_from_str<T>(
) -> Box<dyn Fn(&'static str) -> Result<<T as Parse>::Output, ParseErrorReason>>
where
    T: Parse + 'static,
{
    use slp_transform_lexer::lex;
    use slp_transform_preprocess::preprocess_single;
    let parse_func = node::<T>;
    Box::new(move |string: &'static str| {
        let modified_string = string.to_string() + "\n";
        let preprocessed_text = preprocess_single(&modified_string, FileName(String::new()))
            .expect("preprocess failed");
        let lex_result = lex(&preprocessed_text);
        match lex_result {
            Ok(tokens) => {
                let stream = &tokens.stream;
                match parse_func(stream) {
                    Ok((rem, exp)) => {
                        if rem.len() == 1 && rem[0].0 == Token::Eof {
                            Ok(exp)
                        } else {
                            Err(ParseErrorReason::FailedToParse)
                        }
                    }
                    Err(nom::Err::Incomplete(_)) => Err(ParseErrorReason::UnexpectedEndOfStream),
                    _ => Err(ParseErrorReason::FailedToParse),
                }
            }
            Err(error) => panic!("Failed to lex `{:?}`", error),
        }
    })
}

#[cfg(test)]
fn parse_from_str<T>() -> Box<dyn Fn(&'static str) -> <T as Parse>::Output>
where
    T: Parse + 'static,
{
    use slp_transform_lexer::lex;
    use slp_transform_preprocess::preprocess_single;
    let parse_func = node::<T>;
    Box::new(move |string: &'static str| {
        let modified_string = string.to_string() + "\n";
        let preprocessed_text =
            preprocess_single(&modified_string, FileName("parser_test.hlsl".to_string()))
                .expect("preprocess failed");
        let lex_result = lex(&preprocessed_text);
        match lex_result {
            Ok(tokens) => {
                let stream = &tokens.stream;
                match parse_func(stream) {
                    Ok((rem, exp)) => {
                        if rem.len() == 1 && rem[0].0 == Token::Eof {
                            exp
                        } else {
                            panic!("Tokens remaining while parsing `{:?}`: {:?}", stream, rem)
                        }
                    }
                    Err(nom::Err::Incomplete(needed)) => {
                        panic!("Failed to parse `{:?}`: Needed {:?} more", stream, needed)
                    }
                    Err(nom::Err::Error(ParseErrorContext(_, err))) => {
                        panic!("Failed to parse `{:?}`: Error: {:?}", err, stream)
                    }
                    Err(nom::Err::Failure(ParseErrorContext(_, err))) => {
                        panic!("Failed to parse `{:?}`: Error: {:?}", err, stream)
                    }
                }
            }
            Err(error) => panic!("Failed to lex `{:?}`", error),
        }
    })
}

#[cfg(test)]
fn node_with_symbols<'t, T>(
    input: &'t [LexToken],
    st: &SymbolTable,
) -> ParseResult<'t, <T as Parse>::Output>
where
    T: Parse,
{
    T::parse(input, st)
}

#[cfg(test)]
fn parse_from_str_with_symbols<T>(
) -> Box<dyn Fn(&'static str, &SymbolTable) -> <T as Parse>::Output>
where
    T: Parse + 'static,
{
    use slp_transform_lexer::lex;
    use slp_transform_preprocess::preprocess_single;
    let parse_func = node_with_symbols::<T>;
    Box::new(move |string: &'static str, symbols: &SymbolTable| {
        let modified_string = string.to_string() + "\n";
        let preprocessed_text =
            preprocess_single(&modified_string, FileName("parser_test.hlsl".to_string()))
                .expect("preprocess failed");
        let lex_result = lex(&preprocessed_text);
        match lex_result {
            Ok(tokens) => {
                let stream = &tokens.stream;
                match parse_func(stream, symbols) {
                    Ok((rem, exp)) => {
                        if rem.len() == 1 && rem[0].0 == Token::Eof {
                            exp
                        } else {
                            panic!("Tokens remaining while parsing `{:?}`: {:?}", stream, rem)
                        }
                    }
                    Err(nom::Err::Incomplete(needed)) => {
                        panic!("Failed to parse `{:?}`: Needed {:?} more", stream, needed)
                    }
                    Err(nom::Err::Error(ParseErrorContext(_, err))) => {
                        panic!("Failed to parse `{:?}`: Error: {:?}", err, stream)
                    }
                    Err(nom::Err::Failure(ParseErrorContext(_, err))) => {
                        panic!("Failed to parse `{:?}`: Error: {:?}", err, stream)
                    }
                }
            }
            Err(error) => panic!("Failed to lex `{:?}`", error),
        }
    })
}

#[cfg(test)]
fn test_file_loc(line: u64, column: u64) -> FileLocation {
    FileLocation::Known(
        FileName("parser_test.hlsl".to_string()),
        Line(line),
        Column(column),
    )
}

#[cfg(test)]
fn test_loc(line: u64, column: u64, node: Expression) -> Located<Expression> {
    Located::new(node, test_file_loc(line, column))
}

#[test]
fn test_expr() {
    let expr = node::<Expression>;

    assert_eq!(
        expr(
            &[
                LexToken(Token::Id(Identifier("a".to_string())), test_file_loc(1, 1)),
                LexToken(Token::Asterix, test_file_loc(1, 2)),
                LexToken(Token::Id(Identifier("b".to_string())), test_file_loc(1, 3)),
                LexToken(Token::Eof, test_file_loc(1, 4))
            ][..]
        ),
        Ok((
            &[LexToken(Token::Eof, test_file_loc(1, 4))][..],
            test_loc(
                1,
                1,
                Expression::BinaryOperation(
                    BinOp::Multiply,
                    bexp_var("a", 1, 1),
                    bexp_var("b", 1, 3)
                )
            )
        ))
    );

    let expr_str = parse_from_str::<Expression>();
    let expr_str_fail = parse_result_from_str::<Expression>();
    let expr_str_with_symbols = parse_from_str_with_symbols::<Expression>();

    assert_eq!(expr_str("a"), exp_var("a", 1, 1));
    assert_eq!(
        expr_str("4"),
        test_loc(1, 1, Expression::Literal(Literal::UntypedInt(4)))
    );
    assert_eq!(
        expr_str("a+b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::Add, bexp_var("a", 1, 1), bexp_var("b", 1, 3))
        )
    );
    assert_eq!(
        expr_str("a*b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::Multiply, bexp_var("a", 1, 1), bexp_var("b", 1, 3))
        )
    );
    assert_eq!(
        expr_str("a + b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::Add, bexp_var("a", 1, 1), bexp_var("b", 1, 5))
        )
    );

    assert_eq!(
        expr_str("a-b+c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Add,
                Box::new(test_loc(
                    1,
                    1,
                    Expression::BinaryOperation(
                        BinOp::Subtract,
                        bexp_var("a", 1, 1),
                        bexp_var("b", 1, 3)
                    )
                )),
                bexp_var("c", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a-b*c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Subtract,
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    3,
                    Expression::BinaryOperation(
                        BinOp::Multiply,
                        bexp_var("b", 1, 3),
                        bexp_var("c", 1, 5)
                    )
                ))
            )
        )
    );
    assert_eq!(
        expr_str("a*b-c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Subtract,
                Box::new(test_loc(
                    1,
                    1,
                    Expression::BinaryOperation(
                        BinOp::Multiply,
                        bexp_var("a", 1, 1),
                        bexp_var("b", 1, 3)
                    )
                )),
                bexp_var("c", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a-b*c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Subtract,
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    3,
                    Expression::BinaryOperation(
                        BinOp::Multiply,
                        bexp_var("b", 1, 3),
                        bexp_var("c", 1, 5)
                    )
                ))
            )
        )
    );
    assert_eq!(
        expr_str("a*b-c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Subtract,
                Box::new(test_loc(
                    1,
                    1,
                    Expression::BinaryOperation(
                        BinOp::Multiply,
                        bexp_var("a", 1, 1),
                        bexp_var("b", 1, 3)
                    )
                )),
                bexp_var("c", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a*(b-c)"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Multiply,
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    3,
                    Expression::BinaryOperation(
                        BinOp::Subtract,
                        bexp_var("b", 1, 4),
                        bexp_var("c", 1, 6)
                    )
                ))
            )
        )
    );
    assert_eq!(
        expr_str("a*b/c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Divide,
                Box::new(test_loc(
                    1,
                    1,
                    Expression::BinaryOperation(
                        BinOp::Multiply,
                        bexp_var("a", 1, 1),
                        bexp_var("b", 1, 3)
                    )
                )),
                bexp_var("c", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("(a*b)/c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Divide,
                Box::new(test_loc(
                    1,
                    1,
                    Expression::BinaryOperation(
                        BinOp::Multiply,
                        bexp_var("a", 1, 2),
                        bexp_var("b", 1, 4)
                    )
                )),
                bexp_var("c", 1, 7)
            )
        )
    );
    assert_eq!(
        expr_str("a*(b/c)"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Multiply,
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    3,
                    Expression::BinaryOperation(
                        BinOp::Divide,
                        bexp_var("b", 1, 4),
                        bexp_var("c", 1, 6)
                    )
                ))
            )
        )
    );

    let numeric_cast = "(float3) x";
    let numeric_cast_out = {
        let cast = Expression::Cast(Type::floatn(3), bexp_var("x", 1, 10));
        test_loc(1, 1, cast)
    };
    assert_eq!(expr_str(numeric_cast), numeric_cast_out);

    let ambiguous_sum_or_cast = "(a) + (b)";
    let ambiguous_sum_or_cast_out_sum = {
        let bin = Expression::BinaryOperation(BinOp::Add, bexp_var("a", 1, 1), bexp_var("b", 1, 7));
        test_loc(1, 1, bin)
    };
    let ambiguous_sum_or_cast_out_cast = {
        let unary = Expression::UnaryOperation(UnaryOp::Plus, bexp_var("b", 1, 7));
        let cast = Expression::Cast(Type::custom("a"), Box::new(test_loc(1, 5, unary)));
        test_loc(1, 1, cast)
    };
    assert_eq!(
        expr_str(ambiguous_sum_or_cast),
        ambiguous_sum_or_cast_out_sum
    );
    let st_a_is_type = SymbolTable({
        let mut map = HashMap::new();
        map.insert("a".to_string(), SymbolType::Struct);
        map
    });
    assert_eq!(
        expr_str_with_symbols(ambiguous_sum_or_cast, &st_a_is_type),
        ambiguous_sum_or_cast_out_cast
    );

    let numeric_cons = "float2(x, y)";
    let numeric_cons_out = {
        let x = exp_var("x", 1, 8);
        let y = exp_var("y", 1, 11);
        let ty = DataLayout::Vector(ScalarType::Float, 2);
        let cons = Expression::NumericConstructor(ty, vec![x, y]);
        test_loc(1, 1, cons)
    };
    assert_eq!(expr_str(numeric_cons), numeric_cons_out);

    let fake_cons = "(float2)(x, y)";
    let fake_cons_out = {
        let x = bexp_var("x", 1, 10);
        let y = bexp_var("y", 1, 13);
        let binop = test_loc(1, 9, Expression::BinaryOperation(BinOp::Sequence, x, y));
        let cons = Expression::Cast(Type::floatn(2), Box::new(binop));
        test_loc(1, 1, cons)
    };
    assert_eq!(expr_str(fake_cons), fake_cons_out);

    assert_eq!(
        expr_str("a++"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::PostfixIncrement, bexp_var("a", 1, 1))
        )
    );
    assert_eq!(
        expr_str("a--"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::PostfixDecrement, bexp_var("a", 1, 1))
        )
    );
    assert_eq!(
        expr_str("++a"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::PrefixIncrement, bexp_var("a", 1, 3))
        )
    );
    assert_eq!(
        expr_str("--a"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::PrefixDecrement, bexp_var("a", 1, 3))
        )
    );
    assert_eq!(
        expr_str("+a"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::Plus, bexp_var("a", 1, 2))
        )
    );
    assert_eq!(
        expr_str("-a"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::Minus, bexp_var("a", 1, 2))
        )
    );
    assert_eq!(
        expr_str("!a"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::LogicalNot, bexp_var("a", 1, 2))
        )
    );
    assert_eq!(
        expr_str("~a"),
        test_loc(
            1,
            1,
            Expression::UnaryOperation(UnaryOp::BitwiseNot, bexp_var("a", 1, 2))
        )
    );

    assert_eq!(
        expr_str("a << b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::LeftShift, bexp_var("a", 1, 1), bexp_var("b", 1, 6))
        )
    );
    assert_eq!(
        expr_str("a >> b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::RightShift,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a < b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::LessThan, bexp_var("a", 1, 1), bexp_var("b", 1, 5))
        )
    );
    assert_eq!(
        expr_str("a <= b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::LessEqual, bexp_var("a", 1, 1), bexp_var("b", 1, 6))
        )
    );
    assert_eq!(
        expr_str("a > b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::GreaterThan,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a >= b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::GreaterEqual,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a == b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::Equality, bexp_var("a", 1, 1), bexp_var("b", 1, 6))
        )
    );
    assert_eq!(
        expr_str("a != b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Inequality,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a & b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::BitwiseAnd,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a | b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::BitwiseOr, bexp_var("a", 1, 1), bexp_var("b", 1, 5))
        )
    );
    assert_eq!(
        expr_str("a ^ b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::BitwiseXor,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a && b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::BooleanAnd,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a || b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(BinOp::BooleanOr, bexp_var("a", 1, 1), bexp_var("b", 1, 6))
        )
    );

    assert_eq!(
        expr_str_fail("a < < b"),
        Err(ParseErrorReason::FailedToParse)
    );
    assert_eq!(
        expr_str_fail("a > > b"),
        Err(ParseErrorReason::FailedToParse)
    );
    assert_eq!(
        expr_str_fail("a < = b"),
        Err(ParseErrorReason::FailedToParse)
    );
    assert_eq!(
        expr_str_fail("a > = b"),
        Err(ParseErrorReason::FailedToParse)
    );
    assert_eq!(
        expr_str_fail("a = = b"),
        Err(ParseErrorReason::FailedToParse)
    );
    assert_eq!(
        expr_str_fail("a ! = b"),
        Err(ParseErrorReason::FailedToParse)
    );

    assert_eq!(
        expr_str("a[b]"),
        test_loc(
            1,
            1,
            Expression::ArraySubscript(bexp_var("a", 1, 1), bexp_var("b", 1, 3))
        )
    );
    assert_eq!(
        expr_str("d+a[b+c]"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Add,
                bexp_var("d", 1, 1),
                Box::new(test_loc(
                    1,
                    3,
                    Expression::ArraySubscript(
                        bexp_var("a", 1, 3),
                        Box::new(test_loc(
                            1,
                            5,
                            Expression::BinaryOperation(
                                BinOp::Add,
                                bexp_var("b", 1, 5),
                                bexp_var("c", 1, 7)
                            )
                        ))
                    )
                ))
            )
        )
    );
    assert_eq!(
        expr_str(" d + a\t[ b\n+ c ]"),
        test_loc(
            1,
            2,
            Expression::BinaryOperation(
                BinOp::Add,
                bexp_var("d", 1, 2),
                Box::new(test_loc(
                    1,
                    6,
                    Expression::ArraySubscript(
                        bexp_var("a", 1, 6),
                        Box::new(test_loc(
                            1,
                            10,
                            Expression::BinaryOperation(
                                BinOp::Add,
                                bexp_var("b", 1, 10),
                                bexp_var("c", 2, 3)
                            )
                        ))
                    )
                ))
            )
        )
    );

    assert_eq!(
        expr_str("array.Load"),
        test_loc(
            1,
            1,
            Expression::Member(bexp_var("array", 1, 1), "Load".to_string())
        )
    );
    assert_eq!(
        expr_str("array.Load()"),
        test_loc(
            1,
            1,
            Expression::Call(
                Box::new(test_loc(
                    1,
                    1,
                    Expression::Member(bexp_var("array", 1, 1), "Load".to_string())
                )),
                vec![]
            )
        )
    );
    assert_eq!(
        expr_str(" array . Load ( ) "),
        test_loc(
            1,
            2,
            Expression::Call(
                Box::new(test_loc(
                    1,
                    2,
                    Expression::Member(bexp_var("array", 1, 2), "Load".to_string())
                )),
                vec![]
            )
        )
    );
    assert_eq!(
        expr_str("array.Load(a)"),
        test_loc(
            1,
            1,
            Expression::Call(
                Box::new(test_loc(
                    1,
                    1,
                    Expression::Member(bexp_var("array", 1, 1), "Load".to_string())
                )),
                vec![exp_var("a", 1, 12)]
            )
        )
    );
    assert_eq!(
        expr_str("array.Load(a,b)"),
        test_loc(
            1,
            1,
            Expression::Call(
                Box::new(test_loc(
                    1,
                    1,
                    Expression::Member(bexp_var("array", 1, 1), "Load".to_string())
                )),
                vec![exp_var("a", 1, 12), exp_var("b", 1, 14)]
            )
        )
    );
    assert_eq!(
        expr_str("array.Load(a, b)"),
        test_loc(
            1,
            1,
            Expression::Call(
                Box::new(test_loc(
                    1,
                    1,
                    Expression::Member(bexp_var("array", 1, 1), "Load".to_string())
                )),
                vec![exp_var("a", 1, 12), exp_var("b", 1, 15)]
            )
        )
    );

    assert_eq!(
        expr_str("(float) b"),
        test_loc(1, 1, Expression::Cast(Type::float(), bexp_var("b", 1, 9)))
    );

    assert_eq!(
        expr_str("float2(b)"),
        test_loc(
            1,
            1,
            Expression::NumericConstructor(
                DataLayout::Vector(ScalarType::Float, 2),
                vec![exp_var("b", 1, 8)]
            )
        )
    );

    assert_eq!(
        expr_str("a = b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Assignment,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 5)
            )
        )
    );
    assert_eq!(
        expr_str("a = b = c"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::Assignment,
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    5,
                    Expression::BinaryOperation(
                        BinOp::Assignment,
                        bexp_var("b", 1, 5),
                        bexp_var("c", 1, 9)
                    )
                ))
            )
        )
    );

    assert_eq!(
        expr_str("a += b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::SumAssignment,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );

    assert_eq!(
        expr_str("a -= b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::DifferenceAssignment,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a *= b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::ProductAssignment,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a /= b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::QuotientAssignment,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );
    assert_eq!(
        expr_str("a %= b"),
        test_loc(
            1,
            1,
            Expression::BinaryOperation(
                BinOp::RemainderAssignment,
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 6)
            )
        )
    );

    assert_eq!(
        expr_str("a ? b : c"),
        test_loc(
            1,
            1,
            Expression::TernaryConditional(
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 5),
                bexp_var("c", 1, 9)
            )
        )
    );
    assert_eq!(
        expr_str("a ? b ? c : d : e"),
        test_loc(
            1,
            1,
            Expression::TernaryConditional(
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    5,
                    Expression::TernaryConditional(
                        bexp_var("b", 1, 5),
                        bexp_var("c", 1, 9),
                        bexp_var("d", 1, 13)
                    )
                )),
                bexp_var("e", 1, 17)
            )
        )
    );
    assert_eq!(
        expr_str("a ? b : c ? d : e"),
        test_loc(
            1,
            1,
            Expression::TernaryConditional(
                bexp_var("a", 1, 1),
                bexp_var("b", 1, 5),
                Box::new(test_loc(
                    1,
                    9,
                    Expression::TernaryConditional(
                        bexp_var("c", 1, 9),
                        bexp_var("d", 1, 13),
                        bexp_var("e", 1, 17)
                    )
                ))
            )
        )
    );
    assert_eq!(
        expr_str("a ? b ? c : d : e ? f : g"),
        test_loc(
            1,
            1,
            Expression::TernaryConditional(
                bexp_var("a", 1, 1),
                Box::new(test_loc(
                    1,
                    5,
                    Expression::TernaryConditional(
                        bexp_var("b", 1, 5),
                        bexp_var("c", 1, 9),
                        bexp_var("d", 1, 13)
                    )
                )),
                Box::new(test_loc(
                    1,
                    17,
                    Expression::TernaryConditional(
                        bexp_var("e", 1, 17),
                        bexp_var("f", 1, 21),
                        bexp_var("g", 1, 25)
                    )
                ))
            )
        )
    );
}

#[test]
fn test_statement() {
    let statement_str = parse_from_str::<Statement>();

    // Empty statement
    assert_eq!(statement_str(";"), Statement::Empty);

    // Expression statements
    assert_eq!(
        statement_str("func();"),
        Statement::Expression(test_loc(
            1,
            1,
            Expression::Call(bexp_var("func", 1, 1), vec![])
        ))
    );
    assert_eq!(
        statement_str(" func ( ) ; "),
        Statement::Expression(test_loc(
            1,
            2,
            Expression::Call(bexp_var("func", 1, 2), vec![])
        ))
    );

    // For loop init statement
    let init_statement_str = parse_from_str::<InitStatement>();
    let vardef_str = parse_from_str::<VarDef>();

    assert_eq!(
        init_statement_str("x"),
        InitStatement::Expression(exp_var("x", 1, 1))
    );
    assert_eq!(vardef_str("uint x"), VarDef::one("x", Type::uint().into()));
    assert_eq!(
        init_statement_str("uint x"),
        InitStatement::Declaration(VarDef::one("x", Type::uint().into()))
    );
    assert_eq!(
        init_statement_str("uint x = y"),
        InitStatement::Declaration(VarDef::one_with_expr(
            "x",
            Type::uint().into(),
            exp_var("y", 1, 10)
        ))
    );

    // Variable declarations
    assert_eq!(
        statement_str("uint x = y;"),
        Statement::Var(VarDef::one_with_expr(
            "x",
            Type::uint().into(),
            exp_var("y", 1, 10)
        ))
    );
    assert_eq!(
        statement_str("float x[3];"),
        Statement::Var(VarDef {
            local_type: Type::from_layout(TypeLayout::float()).into(),
            defs: vec![LocalVariableName {
                name: "x".to_string(),
                bind: VariableBind::Array(Some(test_loc(
                    1,
                    9,
                    Expression::Literal(Literal::UntypedInt(3))
                ))),
                init: None,
            }]
        })
    );

    // Blocks
    assert_eq!(
        statement_str("{one();two();}"),
        Statement::Block(vec![
            Statement::Expression(test_loc(
                1,
                2,
                Expression::Call(bexp_var("one", 1, 2), vec![])
            )),
            Statement::Expression(test_loc(
                1,
                8,
                Expression::Call(bexp_var("two", 1, 8), vec![])
            ))
        ])
    );
    assert_eq!(
        statement_str(" { one(); two(); } "),
        Statement::Block(vec![
            Statement::Expression(test_loc(
                1,
                4,
                Expression::Call(bexp_var("one", 1, 4), vec![])
            )),
            Statement::Expression(test_loc(
                1,
                11,
                Expression::Call(bexp_var("two", 1, 11), vec![])
            ))
        ])
    );

    // If statement
    assert_eq!(
        statement_str("if(a)func();"),
        Statement::If(
            exp_var("a", 1, 4),
            Box::new(Statement::Expression(test_loc(
                1,
                6,
                Expression::Call(bexp_var("func", 1, 6), vec![])
            )))
        )
    );
    assert_eq!(
        statement_str("if (a) func(); "),
        Statement::If(
            exp_var("a", 1, 5),
            Box::new(Statement::Expression(test_loc(
                1,
                8,
                Expression::Call(bexp_var("func", 1, 8), vec![])
            )))
        )
    );
    assert_eq!(
        statement_str("if (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::If(
            exp_var("a", 1, 5),
            Box::new(Statement::Block(vec![
                Statement::Expression(test_loc(
                    3,
                    2,
                    Expression::Call(bexp_var("one", 3, 2), vec![])
                )),
                Statement::Expression(test_loc(
                    4,
                    2,
                    Expression::Call(bexp_var("two", 4, 2), vec![])
                ))
            ]))
        )
    );

    // If-else statement
    assert_eq!(
        statement_str("if (a) one(); else two();"),
        Statement::IfElse(
            exp_var("a", 1, 5),
            Box::new(Statement::Expression(test_loc(
                1,
                8,
                Expression::Call(bexp_var("one", 1, 8), vec![])
            ))),
            Box::new(Statement::Expression(test_loc(
                1,
                20,
                Expression::Call(bexp_var("two", 1, 20), vec![])
            )))
        )
    );

    // While loops
    assert_eq!(
        statement_str("while (a)\n{\n\tone();\n\ttwo();\n}"),
        Statement::While(
            exp_var("a", 1, 8),
            Box::new(Statement::Block(vec![
                Statement::Expression(test_loc(
                    3,
                    2,
                    Expression::Call(bexp_var("one", 3, 2), vec![])
                )),
                Statement::Expression(test_loc(
                    4,
                    2,
                    Expression::Call(bexp_var("two", 4, 2), vec![])
                ))
            ]))
        )
    );

    // For loops
    assert_eq!(
        statement_str("for(a;b;c)func();"),
        Statement::For(
            InitStatement::Expression(exp_var("a", 1, 5)),
            exp_var("b", 1, 7),
            exp_var("c", 1, 9),
            Box::new(Statement::Expression(test_loc(
                1,
                11,
                Expression::Call(bexp_var("func", 1, 11), vec![])
            )))
        )
    );
    assert_eq!(
        statement_str("for (uint i = 0; i; i++) { func(); }"),
        Statement::For(
            InitStatement::Declaration(VarDef::one_with_expr(
                "i",
                Type::uint().into(),
                test_loc(1, 15, Expression::Literal(Literal::UntypedInt(0)))
            )),
            exp_var("i", 1, 18),
            test_loc(
                1,
                21,
                Expression::UnaryOperation(UnaryOp::PostfixIncrement, bexp_var("i", 1, 21))
            ),
            Box::new(Statement::Block(vec![Statement::Expression(test_loc(
                1,
                28,
                Expression::Call(bexp_var("func", 1, 28), vec![])
            ))]))
        )
    );
}

#[test]
fn test_rootdefinition() {
    let rootdefinition_str = parse_from_str::<RootDefinition>();
    let rootdefinition_str_with_symbols = parse_from_str_with_symbols::<RootDefinition>();

    let structdefinition_str = parse_from_str::<StructDefinition>();

    let test_struct_str = "struct MyStruct { uint a; float b; };";
    let test_struct_ast = StructDefinition {
        name: "MyStruct".to_string(),
        members: vec![
            StructMember {
                ty: Type::uint(),
                defs: vec![StructMemberName {
                    name: "a".to_string(),
                    bind: VariableBind::Normal,
                }],
            },
            StructMember {
                ty: Type::float(),
                defs: vec![StructMemberName {
                    name: "b".to_string(),
                    bind: VariableBind::Normal,
                }],
            },
        ],
    };
    assert_eq!(structdefinition_str(test_struct_str), test_struct_ast);
    assert_eq!(
        rootdefinition_str(test_struct_str),
        RootDefinition::Struct(test_struct_ast)
    );

    let functiondefinition_str = parse_from_str::<FunctionDefinition>();

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
    assert_eq!(functiondefinition_str(test_func_str), test_func_ast);
    assert_eq!(
        rootdefinition_str(test_func_str),
        RootDefinition::Function(test_func_ast)
    );
    let untyped_int = Literal::UntypedInt;
    let elit = Expression::Literal;
    let loc = test_loc;
    let numthreads = FunctionAttribute::NumThreads(
        loc(1, 13, elit(untyped_int(16))),
        loc(1, 17, elit(untyped_int(16))),
        loc(1, 21, elit(untyped_int(1))),
    );
    assert_eq!(
        rootdefinition_str("[numthreads(16, 16, 1)] void func(float x) { }"),
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
        })
    );

    let constantvariable_str = parse_from_str::<ConstantVariable>();

    let test_cbuffervar_str = "float4x4 wvp;";
    let test_cbuffervar_ast = ConstantVariable {
        ty: Type::float4x4(),
        defs: vec![ConstantVariableName {
            name: "wvp".to_string(),
            bind: VariableBind::Normal,
            offset: None,
        }],
    };
    assert_eq!(
        constantvariable_str(test_cbuffervar_str),
        test_cbuffervar_ast
    );

    let cbuffer_str = parse_from_str::<ConstantBuffer>();

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
    assert_eq!(cbuffer_str(test_cbuffer1_str), test_cbuffer1_ast);
    assert_eq!(
        rootdefinition_str(test_cbuffer1_str),
        RootDefinition::ConstantBuffer(test_cbuffer1_ast)
    );

    let cbuffer_register_str = parse_from_str::<ConstantSlot>();
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
    let test_cbuffer2_ast_xy_m1 = ConstantVariableName {
        name: "x".to_string(),
        bind: VariableBind::Normal,
        offset: None,
    };
    let test_cbuffer2_ast_xy_m2 = ConstantVariableName {
        name: "y".to_string(),
        bind: VariableBind::Array(Some(test_loc(
            1,
            60,
            Expression::Literal(Literal::UntypedInt(2)),
        ))),
        offset: None,
    };
    let test_cbuffer2_ast_xy = ConstantVariable {
        ty: Type::float(),
        defs: vec![test_cbuffer2_ast_xy_m1, test_cbuffer2_ast_xy_m2],
    };
    let test_cbuffer2_ast = ConstantBuffer {
        name: "globals".to_string(),
        slot: Some(ConstantSlot(12)),
        members: vec![test_cbuffer2_ast_wvp, test_cbuffer2_ast_xy],
    };
    assert_eq!(cbuffer_str(test_cbuffer2_str), test_cbuffer2_ast);
    assert_eq!(
        rootdefinition_str(test_cbuffer2_str),
        RootDefinition::ConstantBuffer(test_cbuffer2_ast)
    );

    let globalvariable_str = parse_from_str::<GlobalVariable>();
    let globalvariable_str_with_symbols = parse_from_str_with_symbols::<GlobalVariable>();

    let test_buffersrv_str = "Buffer g_myBuffer : register(t1);";
    let test_buffersrv_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::Buffer(DataType(
            DataLayout::Vector(ScalarType::Float, 4),
            TypeModifier::default(),
        )))
        .into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv_str), test_buffersrv_ast);
    assert_eq!(
        rootdefinition_str(test_buffersrv_str),
        RootDefinition::GlobalVariable(test_buffersrv_ast)
    );

    let test_buffersrv2_str = "Buffer<uint4> g_myBuffer : register(t1);";
    let test_buffersrv2_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::Buffer(DataType(
            DataLayout::Vector(ScalarType::UInt, 4),
            TypeModifier::default(),
        )))
        .into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv2_str), test_buffersrv2_ast);
    assert_eq!(
        rootdefinition_str(test_buffersrv2_str),
        RootDefinition::GlobalVariable(test_buffersrv2_ast)
    );

    let test_buffersrv3_str = "Buffer<vector<int, 4>> g_myBuffer : register(t1);";
    let test_buffersrv3_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::Buffer(DataType(
            DataLayout::Vector(ScalarType::Int, 4),
            TypeModifier::default(),
        )))
        .into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    assert_eq!(globalvariable_str(test_buffersrv3_str), test_buffersrv3_ast);
    assert_eq!(
        rootdefinition_str(test_buffersrv3_str),
        RootDefinition::GlobalVariable(test_buffersrv3_ast)
    );

    let test_buffersrv4_str = "StructuredBuffer<CustomType> g_myBuffer : register(t1);";
    let test_buffersrv4_ast = GlobalVariable {
        global_type: Type::from_object(ObjectType::StructuredBuffer(StructuredType(
            StructuredLayout::Custom("CustomType".to_string()),
            TypeModifier::default(),
        )))
        .into(),
        defs: vec![GlobalVariableName {
            name: "g_myBuffer".to_string(),
            bind: VariableBind::Normal,
            slot: Some(GlobalSlot::ReadSlot(1)),
            init: None,
        }],
    };
    let test_buffersrv4_symbols = SymbolTable({
        let mut map = HashMap::new();
        map.insert("CustomType".to_string(), SymbolType::Struct);
        map
    });
    assert_eq!(
        globalvariable_str_with_symbols(test_buffersrv4_str, &test_buffersrv4_symbols),
        test_buffersrv4_ast
    );
    assert_eq!(
        rootdefinition_str_with_symbols(test_buffersrv4_str, &test_buffersrv4_symbols),
        RootDefinition::GlobalVariable(test_buffersrv4_ast)
    );

    let test_static_const_str = "static const int c_numElements = 4;";
    let test_static_const_ast = GlobalVariable {
        global_type: GlobalType(
            Type(
                TypeLayout::int(),
                TypeModifier {
                    is_const: true,
                    ..TypeModifier::default()
                },
            ),
            GlobalStorage::Static,
            None,
        ),
        defs: vec![GlobalVariableName {
            name: "c_numElements".to_string(),
            bind: VariableBind::Normal,
            slot: None,
            init: Some(Initializer::Expression(test_loc(
                1,
                34,
                Expression::Literal(Literal::UntypedInt(4)),
            ))),
        }],
    };
    assert_eq!(
        globalvariable_str(test_static_const_str),
        test_static_const_ast
    );
    assert_eq!(
        rootdefinition_str(test_static_const_str),
        RootDefinition::GlobalVariable(test_static_const_ast)
    );

    let test_const_arr_str = "static const int data[4] = { 0, 1, 2, 3 };";
    let test_const_arr_ast_lits = vec![
        Initializer::Expression(test_loc(1, 30, Expression::Literal(Literal::UntypedInt(0)))),
        Initializer::Expression(test_loc(1, 33, Expression::Literal(Literal::UntypedInt(1)))),
        Initializer::Expression(test_loc(1, 36, Expression::Literal(Literal::UntypedInt(2)))),
        Initializer::Expression(test_loc(1, 39, Expression::Literal(Literal::UntypedInt(3)))),
    ];
    let test_const_arr_ast_gvn = GlobalVariableName {
        name: "data".to_string(),
        bind: VariableBind::Array(Some(test_loc(
            1,
            23,
            Expression::Literal(Literal::UntypedInt(4)),
        ))),
        slot: None,
        init: Some(Initializer::Aggregate(test_const_arr_ast_lits)),
    };
    let test_const_arr_ast = GlobalVariable {
        global_type: GlobalType(
            Type(
                TypeLayout::int(),
                TypeModifier {
                    is_const: true,
                    ..TypeModifier::default()
                },
            ),
            GlobalStorage::Static,
            None,
        ),
        defs: vec![test_const_arr_ast_gvn],
    };
    assert_eq!(globalvariable_str(test_const_arr_str), test_const_arr_ast);
    assert_eq!(
        rootdefinition_str(test_const_arr_str),
        RootDefinition::GlobalVariable(test_const_arr_ast)
    );

    let test_groupshared_str = "groupshared float4 local_data[32];";
    let test_groupshared_ast_gvn = GlobalVariableName {
        name: "local_data".to_string(),
        bind: VariableBind::Array(Some(test_loc(
            1,
            31,
            Expression::Literal(Literal::UntypedInt(32)),
        ))),
        slot: None,
        init: None,
    };
    let test_groupshared_ast = GlobalVariable {
        global_type: GlobalType(Type::floatn(4), GlobalStorage::GroupShared, None),
        defs: vec![test_groupshared_ast_gvn],
    };
    assert_eq!(
        globalvariable_str(test_groupshared_str),
        test_groupshared_ast
    );
    assert_eq!(
        rootdefinition_str(test_groupshared_str),
        RootDefinition::GlobalVariable(test_groupshared_ast)
    );
}
