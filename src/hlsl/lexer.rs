use super::tokens::*;
use nom::{IResult,Needed,Err,ErrorKind};
use std::str;

named!(digit<i64>, alt!(
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

named!(digits<i64>, chain!(
    digits: many1!(digit),
    || {
        let mut value = 0i64;
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
    negate: opt!(tag!("-")) ~
    digits: digits ~
    int_type_opt: opt!(int_type),
    || {
        let mut value = digits;
        match negate {
            Some(_) => value = -value,
            None => { }
        };
        match int_type_opt {
            None => Token::LiteralInt(value as i32),
            Some(IntType::Uint) => Token::LiteralUint(value as u32),
            Some(IntType::Long) => Token::LiteralLong(value),
        }
    }
));

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
// Find a better way to do this that backs out as early as possible
named!(reserved_word_if<()>, chain!(tag!("i") ~ tag!("f"), || { }));
named!(reserved_word_for<()>, chain!(tag!("f") ~ tag!("o") ~ tag!("r"), || { }));
named!(reserved_word_while<()>, chain!(tag!("w") ~ tag!("h") ~ tag!("i") ~ tag!("l") ~ tag!("e"), || { }));
named!(reserved_word_switch<()>, chain!(tag!("s") ~ tag!("w") ~ tag!("i") ~ tag!("t") ~ tag!("c") ~ tag!("h"), || { }));
named!(reserved_word_struct<()>, chain!(tag!("s") ~ tag!("t") ~ tag!("r") ~ tag!("u") ~ tag!("c") ~ tag!("t"), || { }));
named!(reserved_word_samplerstate<()>, chain!(tag!("S") ~ tag!("a") ~ tag!("m") ~ tag!("p") ~ tag!("l") ~ tag!("e") ~ tag!("r") ~ tag!("S") ~ tag!("t") ~ tag!("a") ~ tag!("t") ~ tag!("e") , || { }));
named!(reserved_word_cbuffer<()>, chain!(tag!("c") ~ tag!("b") ~ tag!("u") ~ tag!("f") ~ tag!("f") ~ tag!("e") ~ tag!("r"), || { }));
named!(reserved_word_register<()>, chain!(tag!("r") ~ tag!("e") ~ tag!("g") ~ tag!("i") ~ tag!("s") ~ tag!("t") ~ tag!("e") ~ tag!("r"), || { }));

named!(reserved_word<()>, alt!(
    reserved_word_if |
    reserved_word_for |
    reserved_word_while |
    reserved_word_switch |
    reserved_word_struct |
    reserved_word_samplerstate |
    reserved_word_cbuffer |
    reserved_word_register
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

named!(token_no_whitespace<Token>, alt!(

    identifier => { |id| Token::Id(id) } |
    literal_int => { |tok| tok } |

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
    tag!("=") => { |_| Token::Equals } |
    tag!("#") => { |_| Token::Hash } |
    tag!("@") => { |_| Token::At } |
    tag!("!") => { |_| Token::ExclamationPoint } |
    tag!("~") => { |_| Token::Tilde } |
    tag!(".") => { |_| Token::Period } |

    reserved_word_if => { |_| { Token::If } } |
    reserved_word_for => { |_| { Token::For } } |
    reserved_word_while => { |_| { Token::While } } |
    reserved_word_switch => { |_| { Token::Switch } } |

    reserved_word_struct => { |_| { Token::Struct } } |
    reserved_word_samplerstate => { |_| { Token::SamplerState } } |
    reserved_word_cbuffer => { |_| { Token::ConstantBuffer } } |
    register |
    tag!(":") => { |_| { Token::Colon } }
));

named!(token<Token>, delimited!(opt!(whitespace), alt!(token_no_whitespace), opt!(whitespace)));

named!(pub token_stream<TokenStream>, map!(many0!(token), |v| { TokenStream(v) }));

#[test]
fn test_token() {
    assert_eq!(token(&b""[..]), IResult::Incomplete(Needed::Size(1)));
    assert_eq!(token(&b";"[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b" ;"[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b"; "[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b" ; "[..]), IResult::Done(&b""[..], Token::Semicolon));
    assert_eq!(token(&b"name"[..]), IResult::Done(&b""[..], Token::Id(Identifier("name".to_string()))));

    assert_eq!(token(&b"12"[..]), IResult::Done(&b""[..], Token::LiteralInt(12)));
    assert_eq!(token(&b"-12"[..]), IResult::Done(&b""[..], Token::LiteralInt(-12)));
    assert_eq!(token(&b"12u"[..]), IResult::Done(&b""[..], Token::LiteralUint(12)));
    assert_eq!(token(&b"12l"[..]), IResult::Done(&b""[..], Token::LiteralLong(12)));
    assert_eq!(token(&b"12L"[..]), IResult::Done(&b""[..], Token::LiteralLong(12)));
    assert_eq!(token(&b"-12l"[..]), IResult::Done(&b""[..], Token::LiteralLong(-12)));
    assert_eq!(token(&b"-12L"[..]), IResult::Done(&b""[..], Token::LiteralLong(-12)));

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

    assert_eq!(token_stream(&b"<<"[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::LeftAngleBracket(FollowedBy::Token),
        Token::LeftAngleBracket(FollowedBy::Whitespace),
    ])));
    assert_eq!(token_stream(&b">>"[..]), IResult::Done(&b""[..], TokenStream(vec![
        Token::RightAngleBracket(FollowedBy::Token),
        Token::RightAngleBracket(FollowedBy::Whitespace),
    ])));
}
