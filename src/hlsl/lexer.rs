use std::str;
use std::error;
use std::fmt;
use super::tokens::*;
use nom::{IResult,Needed,Err,ErrorKind};

#[derive(PartialEq, Debug, Clone)]
pub enum LexError {
    Unknown,
    FailedToParse(Vec<u8>),
    UnexpectedEndOfStream,
}

impl error::Error for LexError {
    fn description(&self) -> &str {
        match *self {
            LexError::Unknown => "unknown lexer error",
            LexError::FailedToParse(_) => "failed to parse stream",
            LexError::UnexpectedEndOfStream => "unexpected end of stream",
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

named!(digit<u64>, alt!(
    tag!("0") => { |_| { 0 } } |
    tag!("1") => { |_| { 1 } } |
    tag!("2") => { |_| { 2 } } |
    tag!("3") => { |_| { 3 } } |
    tag!("4") => { |_| { 4 } } |
    tag!("5") => { |_| { 5 } } |
    tag!("6") => { |_| { 6 } } |
    tag!("7") => { |_| { 7 } } |
    tag!("8") => { |_| { 8 } } |
    tag!("9") => { |_| { 9 } }
));

named!(digits<u64>, chain!(
    digits: many1!(digit),
    || {
        let mut value = 0u64;
        for digit in digits {
            value = value * 10;
            value += digit;
        };
        value
    }
));

enum IntType { Uint, Long }
named!(int_type<IntType>, alt!(
    tag!("u") => { |_| { IntType::Uint } } |
    tag!("U") => { |_| { IntType::Uint } } |
    tag!("l") => { |_| { IntType::Long } } |
    tag!("L") => { |_| { IntType::Long } }
));

named!(literal_int<Token>, chain!(
    value: digits ~
    int_type_opt: opt!(int_type),
    || {
        match int_type_opt {
            None => Token::LiteralInt(value),
            Some(IntType::Uint) => Token::LiteralUint(value),
            Some(IntType::Long) => Token::LiteralLong(value),
        }
    }
));

fn literal_float<'a>(input: &'a [u8]) -> IResult<&'a [u8], Token> {

    type DigitSequence = Vec<u64>;
    #[derive(PartialEq, Debug, Clone)]
    struct Fraction(DigitSequence, DigitSequence);

    named!(digit_sequence<DigitSequence>, many1!(digit));

    named!(fractional_constant<Fraction>, alt!(
        chain!(left: opt!(digit_sequence) ~ tag!(".") ~ right: digit_sequence, || { Fraction(match &left { &Some(ref l) => l.clone(), &None => vec![] }, right) }) |
        chain!(left: digit_sequence ~ tag!("."), || { Fraction(left, vec![]) })
    ));

    enum FloatType { Half, Float, Double };
    named!(suffix<FloatType>, alt!(
        tag!("h") => { |_| FloatType::Half } |
        tag!("H") => { |_| FloatType::Half } |
        tag!("f") => { |_| FloatType::Float } |
        tag!("F") => { |_| FloatType::Float } |
        tag!("l") => { |_| FloatType::Double } |
        tag!("L") => { |_| FloatType::Double }
    ));

    enum Sign { Positive, Negative };
    named!(sign<Sign>, alt!(
        tag!("+") => { |_| Sign::Positive } |
        tag!("-") => { |_| Sign::Negative }
    ));

    #[derive(PartialEq, Debug, Clone)]
    struct Exponent(i64);
    named!(exponent<Exponent>, chain!(
        alt!(tag!("e") | tag!("E")) ~
        s_opt: opt!(sign) ~
        exponent: digits,
        || { Exponent(match s_opt { Some(Sign::Negative) => -(exponent as i64), _ => exponent as i64 }) }
    ));

    fn produce(left: DigitSequence, right: DigitSequence, exponent: i64, float_type: Option<FloatType>) -> Token {

        let mut left_combined = 0f64;
        for digit in left {
            left_combined = left_combined * 10f64;
            left_combined += digit as f64;
        };
        let left_float = left_combined as f64;

        let mut right_combined = 0f64;
        let right_len = right.len();
        for digit in right {
            right_combined = right_combined * 10f64;
            right_combined += digit as f64;
        };
        let mut right_float = right_combined as f64;
        for _ in 0..right_len {
            right_float = right_float / 10f64;
        }

        let mantissa = left_float + right_float;
        let mut value64 = mantissa;
        if exponent > 0 {
            for _ in 0..exponent {
                value64 = value64 * 10f64;
            }
        } else {
            for _ in 0..(-exponent) {
                value64 = value64 / 10f64;
            }
        }

        match float_type.unwrap_or(FloatType::Float) {
            FloatType::Half => Token::LiteralHalf(value64 as f32),
            FloatType::Float => Token::LiteralFloat(value64 as f32),
            FloatType::Double => Token::LiteralDouble(value64),
        }
    }

    alt!(input,
        chain!(
            fraction: fractional_constant ~
            exponent_opt: opt!(exponent) ~
            suffix_opt: opt!(suffix),
            || {
                let exponent = exponent_opt.unwrap_or(Exponent(0));
                let Fraction(left, right) = fraction.clone();
                let Exponent(exp) = exponent.clone();
                produce(left, right, exp, suffix_opt)
            }
        ) |
        chain!(
            digits: digit_sequence ~
            exponent: exponent ~
            suffix_opt: opt!(suffix),
            || { produce(digits, vec![], exponent.0, suffix_opt) }
        )
    )
}

fn identifier_firstchar<'a>(input: &'a [u8]) -> IResult<&'a [u8], u8> {
    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        let byte = input[0];
        let ch = byte as char;
        if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch == '_') {
            IResult::Done(&input[1..], byte)
        } else {
            IResult::Error(Err::Position(ErrorKind::Custom(0), input))
        }
    }
}

fn identifier_char<'a>(input: &'a [u8]) -> IResult<&'a [u8], u8> {
    if input.len() == 0 {
        IResult::Incomplete(Needed::Size(1))
    } else {
        let byte = input[0];
        let ch = byte as char;
        if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch == '_') || (ch >= '0' && ch <= '9') {
            IResult::Done(&input[1..], byte)
        } else {
            IResult::Error(Err::Position(ErrorKind::Custom(0), input))
        }
    }
}

fn identifier<'a>(input: &'a [u8]) -> IResult<&'a [u8], Identifier> {
    let mut chars = Vec::new();
    let first_result = identifier_firstchar(input);
    
    let mut stream = match first_result {
        IResult::Incomplete(needed) => return IResult::Incomplete(needed),
        IResult::Error(err) => return IResult::Error(err),
        IResult::Done(output, ch) => { chars.push(ch); output },
    };

    loop {
        stream = match identifier_char(stream) {
            IResult::Incomplete(_) => break,
            IResult::Error(_) => break,
            IResult::Done(output, ch) => { chars.push(ch); output },
        }
    }

    // If we match a reserved word, fail instead. The token parser
    // will then match the reserved word next
    // This is done instead of just running the reserved word parsing
    // first to avoid having the deal with detecting the end of a name
    // for when parsing an identifier that has a reserved word as a
    // sub string at the start
    // Maybe just combine the identifier and reserved word parsing?
    if let IResult::Done(slice, _) = reserved_word(&chars[..]) {
        if slice.len() == 0 {
            return IResult::Error(Err::Code(ErrorKind::Custom(1)))
        }
    }

    IResult::Done(stream, Identifier(str::from_utf8(&chars[..]).unwrap().to_string()))
}

fn whitespace_ignore(_: Vec<&[u8]>) -> Result<(), ()> { Result::Ok(()) }
named!(whitespace<()>, map_res!(
    many1!(alt!(
        tag!(" ") | tag!("\n") | tag!("\r") | tag!("\t")
    )),
    whitespace_ignore
));

// Reserved words
named!(reserved_word_if, complete!(tag!("if")));
named!(reserved_word_for, complete!(tag!("for")));
named!(reserved_word_while, complete!(tag!("while")));
named!(reserved_word_switch, complete!(tag!("switch")));
named!(reserved_word_return, complete!(tag!("return")));
named!(reserved_word_struct, complete!(tag!("struct")));
named!(reserved_word_samplerstate, complete!(tag!("SamplerState")));
named!(reserved_word_cbuffer, complete!(tag!("cbuffer")));
named!(reserved_word_register, complete!(tag!("register")));
named!(reserved_word_true, complete!(tag!("true")));
named!(reserved_word_false, complete!(tag!("false")));
named!(reserved_word_packoffset, complete!(tag!("packoffset")));

// Unused reserved words
named!(reserved_word_auto, complete!(tag!("auto")));
named!(reserved_word_case, complete!(tag!("case")));
named!(reserved_word_catch, complete!(tag!("catch")));
named!(reserved_word_char, complete!(tag!("char")));
named!(reserved_word_class, complete!(tag!("class")));
named!(reserved_word_const_cast, complete!(tag!("const_cast")));
named!(reserved_word_default, complete!(tag!("default")));
named!(reserved_word_delete, complete!(tag!("delete")));
named!(reserved_word_dynamic_cast, complete!(tag!("dynamic_cast")));
named!(reserved_word_enum, complete!(tag!("enum")));
named!(reserved_word_explicit, complete!(tag!("explicit")));
named!(reserved_word_friend, complete!(tag!("friend")));
named!(reserved_word_goto, complete!(tag!("goto")));
named!(reserved_word_long, complete!(tag!("long")));
named!(reserved_word_mutable, complete!(tag!("mutable")));
named!(reserved_word_new, complete!(tag!("new")));
named!(reserved_word_operator, complete!(tag!("operator")));
named!(reserved_word_private, complete!(tag!("private")));
named!(reserved_word_protected, complete!(tag!("protected")));
named!(reserved_word_public, complete!(tag!("public")));
named!(reserved_word_reinterpret_cast, complete!(tag!("reinterpret_cast")));
named!(reserved_word_short, complete!(tag!("short")));
named!(reserved_word_signed, complete!(tag!("signed")));
named!(reserved_word_sizeof, complete!(tag!("sizeof")));
named!(reserved_word_static_cast, complete!(tag!("static_cast")));
named!(reserved_word_template, complete!(tag!("template")));
named!(reserved_word_this, complete!(tag!("this")));
named!(reserved_word_throw, complete!(tag!("throw")));
named!(reserved_word_try, complete!(tag!("try")));
named!(reserved_word_typename, complete!(tag!("typename")));
named!(reserved_word_union, complete!(tag!("union")));
named!(reserved_word_unsigned, complete!(tag!("unsigned")));
named!(reserved_word_using, complete!(tag!("using")));
named!(reserved_word_virtual, complete!(tag!("virtual")));

named!(reserved_word_s0, alt!(
    reserved_word_if |
    reserved_word_for |
    reserved_word_while |
    reserved_word_switch |
    reserved_word_return |
    reserved_word_struct |
    reserved_word_samplerstate |
    reserved_word_cbuffer |
    reserved_word_register |
    reserved_word_true |
    reserved_word_false |
    reserved_word_packoffset
));

named!(reserved_word_s1, alt!(
    reserved_word_auto |
    reserved_word_case |
    reserved_word_catch |
    reserved_word_char |
    reserved_word_class |
    reserved_word_const_cast |
    reserved_word_default |
    reserved_word_delete |
    reserved_word_dynamic_cast |
    reserved_word_enum
));

named!(reserved_word_s2, alt!(
    reserved_word_explicit |
    reserved_word_friend |
    reserved_word_goto |
    reserved_word_long |
    reserved_word_mutable |
    reserved_word_new |
    reserved_word_operator |
    reserved_word_private |
    reserved_word_protected |
    reserved_word_public |
    reserved_word_reinterpret_cast |
    reserved_word_short
));

named!(reserved_word_s3, alt!(
    reserved_word_signed |
    reserved_word_sizeof |
    reserved_word_static_cast |
    reserved_word_template |
    reserved_word_this |
    reserved_word_throw |
    reserved_word_try |
    reserved_word_typename |
    reserved_word_union |
    reserved_word_unsigned |
    reserved_word_using |
    reserved_word_virtual
));

// Distribute these among subfunctions to avoid recursion limits in macros
named!(reserved_word, alt!(
    reserved_word_s0 |
    reserved_word_s1 |
    reserved_word_s2 |
    reserved_word_s3
));

enum RegisterType { T, U, B }
named!(register<Token>, chain!(
    reserved_word_register ~
    opt!(whitespace) ~
    tag!("(") ~
    opt!(whitespace) ~
    slot_type: alt!(tag!("t") => { |_| { RegisterType::T }} | tag!("u") => { |_| { RegisterType::U }} | tag!("b") => { |_| { RegisterType::B }}) ~
    num: digits ~
    tag!(")"),
    || { Token::Register(match slot_type {
        RegisterType::T => RegisterSlot::T(num as u32),
        RegisterType::U => RegisterSlot::U(num as u32),
        RegisterType::B => RegisterSlot::B(num as u32),
    }) }
));

fn lookahead_token(input: &[u8]) -> IResult<&[u8], Option<Token>> {
    match token_no_whitespace(input) {
        IResult::Done(_, o) => IResult::Done(input, Some(o)),
        IResult::Error(_) => IResult::Done(input, None),
        IResult::Incomplete(_) => IResult::Done(input, None),
    }
}

named!(leftanglebracket<Token>, chain!(
    tag!("<") ~
    next: lookahead_token,
    || { 
        match next {
            Some(_) => Token::LeftAngleBracket(FollowedBy::Token),
            _ => Token::LeftAngleBracket(FollowedBy::Whitespace)
        }
    }
));

named!(rightanglebracket<Token>, chain!(
    tag!(">") ~
    next: lookahead_token,
    || { 
        match next {
            Some(_) => Token::RightAngleBracket(FollowedBy::Token),
            _ => Token::RightAngleBracket(FollowedBy::Whitespace)
        }
    }
));

named!(symbol_equals<Token>, chain!(
    tag!("=") ~
    next: opt!(tag!("=")),
    || {
        match next {
            Some(_) => Token::DoubleEquals,
            _ => Token::Equals,
        }
    }
));

named!(symbol_exclamation<Token>, chain!(
    tag!("!") ~
    next: opt!(tag!("=")),
    || {
        match next {
            Some(_) => Token::ExclamationEquals,
            _ => Token::ExclamationPoint,
        }
    }
));

named!(token_no_whitespace<Token>, alt!(

    identifier => { |id| Token::Id(id) } |
    literal_float => { |tok| tok } |
    literal_int => { |tok| tok } |
    reserved_word_true => { |_| Token::True} |
    reserved_word_false => { |_| Token::False } |

    tag!("{") => { |_| Token::LeftBrace } |
    tag!("}") => { |_| Token::RightBrace } |
    tag!("(") => { |_| Token::LeftParen } |
    tag!(")") => { |_| Token::RightParen } |
    tag!("[") => { |_| Token::LeftSquareBracket } |
    tag!("]") => { |_| Token::RightSquareBracket } |

    leftanglebracket |
    rightanglebracket |

    tag!(";") => { |_| Token::Semicolon } |
    tag!(",") => { |_| Token::Comma } |

    tag!("+") => { |_| Token::Plus } |
    tag!("-") => { |_| Token::Minus } |
    tag!("/") => { |_| Token::ForwardSlash } |
    tag!("%") => { |_| Token::Percent } |
    tag!("*") => { |_| Token::Asterix } |
    tag!("|") => { |_| Token::VerticalBar } |
    tag!("&") => { |_| Token::Ampersand } |
    tag!("^") => { |_| Token::Hat } |
    symbol_equals |
    tag!("#") => { |_| Token::Hash } |
    tag!("@") => { |_| Token::At } |
    symbol_exclamation |
    tag!("~") => { |_| Token::Tilde } |
    tag!(".") => { |_| Token::Period } |

    reserved_word_if => { |_| { Token::If } } |
    reserved_word_for => { |_| { Token::For } } |
    reserved_word_while => { |_| { Token::While } } |
    reserved_word_switch => { |_| { Token::Switch } } |
    reserved_word_return => { |_| { Token::Return } } |

    reserved_word_struct => { |_| { Token::Struct } } |
    reserved_word_samplerstate => { |_| { Token::SamplerState } } |
    reserved_word_cbuffer => { |_| { Token::ConstantBuffer } } |
    register |
    tag!(":") => { |_| { Token::Colon } }
));

named!(token<Token>, delimited!(opt!(whitespace), alt!(token_no_whitespace), opt!(whitespace)));

named!(token_stream<TokenStream>, map!(many0!(token), |v| { TokenStream(v) }));

pub fn lex(input: &[u8]) -> Result<TokenStream, LexError> {
    match token_stream(input) {
        IResult::Done(rest, TokenStream(mut result)) => {
            if rest == [] {
                let result_with_eof = { result.push(Token::Eof); result };
                Ok(TokenStream(result_with_eof))
            } else {
                Err(LexError::FailedToParse(rest.to_vec()))
            }
        },
        IResult::Incomplete(_) => {
            Err(LexError::UnexpectedEndOfStream)
        },
        IResult::Error(_) => {
            Err(LexError::Unknown)
        },
    }
}

#[test]
fn test_token() {
    assert_eq!(token(&b""[..]), IResult::Incomplete(Needed::Size(1)));
    assert_eq!(token(&b";"[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b" ;"[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b"; "[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b" ; "[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b"name"[..]), IResult::Done(&b""[..], Token::Id(Identifier("name".to_string()))));

    assert_eq!(token(&b"12 "[..]), IResult::Done(&b""[..], Token::LiteralInt(12)));
    assert_eq!(token(&b"12u"[..]), IResult::Done(&b""[..], Token::LiteralUint(12)));
    assert_eq!(token(&b"12l"[..]), IResult::Done(&b""[..], Token::LiteralLong(12)));
    assert_eq!(token(&b"12L"[..]), IResult::Done(&b""[..], Token::LiteralLong(12)));

    assert_eq!(token(&b"1.0f"[..]), IResult::Done(&b""[..], Token::LiteralFloat(1.0f32)));
    assert_eq!(token(&b"2.0"[..]), IResult::Done(&b""[..], Token::LiteralFloat(2.0f32)));
    assert_eq!(token(&b"2.0L"[..]), IResult::Done(&b""[..], Token::LiteralDouble(2.0f64)));
    assert_eq!(token(&b"0.5h"[..]), IResult::Done(&b""[..], Token::LiteralHalf(0.5f32)));

    assert_eq!(token(&b"{"[..]), IResult::Done(&b""[..], Token::LeftBrace));
    assert_eq!(token(&b"}"[..]), IResult::Done(&b""[..], Token::RightBrace));
    assert_eq!(token(&b"("[..]), IResult::Done(&b""[..], Token::LeftParen));
    assert_eq!(token(&b")"[..]), IResult::Done(&b""[..], Token::RightParen));
    assert_eq!(token(&b"["[..]), IResult::Done(&b""[..], Token::LeftSquareBracket));
    assert_eq!(token(&b"]"[..]), IResult::Done(&b""[..], Token::RightSquareBracket));

    assert_eq!(token(&b"<"[..]), IResult::Done(&b""[..], Token::LeftAngleBracket(FollowedBy::Whitespace)));
    assert_eq!(token(&b">"[..]), IResult::Done(&b""[..], Token::RightAngleBracket(FollowedBy::Whitespace)));
    assert_eq!(token(&b"<<"[..]), IResult::Done(&b"<"[..], Token::LeftAngleBracket(FollowedBy::Token)));
    assert_eq!(token(&b">>"[..]), IResult::Done(&b">"[..], Token::RightAngleBracket(FollowedBy::Token)));
    assert_eq!(token(&b"<>"[..]), IResult::Done(&b">"[..], Token::LeftAngleBracket(FollowedBy::Token)));
    assert_eq!(token(&b"><"[..]), IResult::Done(&b"<"[..], Token::RightAngleBracket(FollowedBy::Token)));

    assert_eq!(token(&b";"[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b","[..]), IResult::Done(&b""[..], Token::Comma));

    assert_eq!(token(&b"+"[..]), IResult::Done(&b""[..], Token::Plus));
    assert_eq!(token(&b"-"[..]), IResult::Done(&b""[..], Token::Minus));
    assert_eq!(token(&b"/"[..]), IResult::Done(&b""[..], Token::ForwardSlash));
    assert_eq!(token(&b"%"[..]), IResult::Done(&b""[..], Token::Percent));
    assert_eq!(token(&b"*"[..]), IResult::Done(&b""[..], Token::Asterix));
    assert_eq!(token(&b"|"[..]), IResult::Done(&b""[..], Token::VerticalBar));
    assert_eq!(token(&b"&"[..]), IResult::Done(&b""[..], Token::Ampersand));
    assert_eq!(token(&b"^"[..]), IResult::Done(&b""[..], Token::Hat));
    assert_eq!(token(&b"="[..]), IResult::Done(&b""[..], Token::Equals));
    assert_eq!(token(&b"#"[..]), IResult::Done(&b""[..], Token::Hash));
    assert_eq!(token(&b"@"[..]), IResult::Done(&b""[..], Token::At));
    assert_eq!(token(&b"!"[..]), IResult::Done(&b""[..], Token::ExclamationPoint));
    assert_eq!(token(&b"~"[..]), IResult::Done(&b""[..], Token::Tilde));
    assert_eq!(token(&b"."[..]), IResult::Done(&b""[..], Token::Period));

    assert_eq!(token(&b"struct"[..]), IResult::Done(&b""[..], Token::Struct));
    assert_eq!(token(&b"SamplerState"[..]), IResult::Done(&b""[..], Token::SamplerState));
    assert_eq!(token(&b"cbuffer"[..]), IResult::Done(&b""[..], Token::ConstantBuffer));
    assert_eq!(token(&b"register(t4)"[..]), IResult::Done(&b""[..], Token::Register(RegisterSlot::T(4))));
    assert_eq!(token(&b":"[..]), IResult::Done(&b""[..], Token::Colon));

    assert_eq!(token(&b"structName"[..]), IResult::Done(&b""[..], Token::Id(Identifier("structName".to_string()))));
}

#[test]
fn test_token_stream() {
    assert_eq!(token_stream(&b""[..]), IResult::Done(&b""[..], TokenStream(vec![])));

    fn token_id(name: &'static str) -> Token { Token::Id(Identifier(name.to_string())) }

    assert_eq!(token_stream(&b" a "[..]), IResult::Done(&b""[..], TokenStream(vec![
        token_id("a"),
    ])));

    assert_eq!(token_stream(&b"void func();"[..]), IResult::Done(&b""[..], TokenStream(vec![
        token_id("void"),
        token_id("func"),
        Token::LeftParen,
        Token::RightParen,
        Token::Semicolon,
    ])));

    assert_eq!(token_stream(&b"-12 "[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::Minus,
        Token::LiteralInt(12),
    ])));
    assert_eq!(token_stream(&b"-12l"[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::Minus,
        Token::LiteralLong(12),
    ])));
    assert_eq!(token_stream(&b"-12L"[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::Minus,
        Token::LiteralLong(12),
    ])));

    assert_eq!(token_stream(&b"<<"[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::LeftAngleBracket(FollowedBy::Token),
        Token::LeftAngleBracket(FollowedBy::Whitespace),
    ])));
    assert_eq!(token_stream(&b">>"[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::RightAngleBracket(FollowedBy::Token),
        Token::RightAngleBracket(FollowedBy::Whitespace),
    ])));
}
