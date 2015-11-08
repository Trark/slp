
use super::super::tokens::Identifier;
use super::super::tokens::TokenStream;
use super::super::tokens::Token;
use super::super::tokens::FollowedBy;
use super::super::tokens::RegisterSlot;
use super::super::lexer::token_stream;
use super::super::parser::parse;
use super::super::ast_to_ir;
use nom::IResult;

fn token_id(name: &'static str) -> Token { Token::Id(Identifier(name.to_string())) }

#[test]
fn cs1() {

    // Test a small compute shader (A simplified form of one the basic hlsl compute examples)
    let cs1 = include_bytes!("cs1.hlsl");

    let tokens_res = token_stream(cs1);
    assert_eq!(tokens_res, IResult::Done(&b""[..], TokenStream(vec![
        Token::Struct,
        token_id("BufType"),
        Token::LeftBrace,
        token_id("int"),
        token_id("i"),
        Token::Semicolon,
        token_id("float"),
        token_id("f"),
        Token::Semicolon,
        Token::RightBrace,
        Token::Semicolon,
        token_id("StructuredBuffer"),
        Token::LeftAngleBracket(FollowedBy::Token),
        token_id("BufType"),
        Token::RightAngleBracket(FollowedBy::Whitespace),
        token_id("Buffer0"),
        Token::Colon,
        Token::Register(RegisterSlot::T(0)),
        Token::Semicolon,
        token_id("StructuredBuffer"),
        Token::LeftAngleBracket(FollowedBy::Token),
        token_id("BufType"),
        Token::RightAngleBracket(FollowedBy::Whitespace),
        token_id("Buffer1"),
        Token::Colon,
        Token::Register(RegisterSlot::T(1)),
        Token::Semicolon,
        token_id("RWStructuredBuffer"),
        Token::LeftAngleBracket(FollowedBy::Token),
        token_id("BufType"),
        Token::RightAngleBracket(FollowedBy::Whitespace),
        token_id("BufferOut"),
        Token::Colon,
        Token::Register(RegisterSlot::U(0)),
        Token::Semicolon,
        Token::LeftSquareBracket,
        token_id("numthreads"),
        Token::LeftParen,
        Token::LiteralInt(1),
        Token::Comma,
        Token::LiteralInt(1),
        Token::Comma,
        Token::LiteralInt(1),
        Token::RightParen,
        Token::RightSquareBracket,
        token_id("void"),
        token_id("CSMain"),
        Token::LeftParen,
        token_id("uint3"),
        token_id("DTid"),
        Token::Colon,
        token_id("SV_DispatchThreadID"),
        Token::RightParen,
        Token::LeftBrace,
        token_id("BufferOut"),
        Token::LeftSquareBracket,
        token_id("DTid"),
        Token::Period,
        token_id("x"),
        Token::RightSquareBracket,
        Token::Period,
        token_id("i"),
        Token::Equals,
        token_id("Buffer0"),
        Token::LeftSquareBracket,
        token_id("DTid"),
        Token::Period,
        token_id("x"),
        Token::RightSquareBracket,
        Token::Period,
        token_id("i"),
        Token::Plus,
        token_id("Buffer1"),
        Token::LeftSquareBracket,
        token_id("DTid"),
        Token::Period,
        token_id("x"),
        Token::RightSquareBracket,
        Token::Period,
        token_id("i"),
        Token::Semicolon,
        token_id("BufferOut"),
        Token::LeftSquareBracket,
        token_id("DTid"),
        Token::Period,
        token_id("x"),
        Token::RightSquareBracket,
        Token::Period,
        token_id("f"),
        Token::Equals,
        token_id("Buffer0"),
        Token::LeftSquareBracket,
        token_id("DTid"),
        Token::Period,
        token_id("x"),
        Token::RightSquareBracket,
        Token::Period,
        token_id("f"),
        Token::Plus,
        token_id("Buffer1"),
        Token::LeftSquareBracket,
        token_id("DTid"),
        Token::Period,
        token_id("x"),
        Token::RightSquareBracket,
        Token::Period,
        token_id("f"),
        Token::Semicolon,
        Token::RightBrace,
    ])));

    let tokens = match tokens_res { IResult::Done(_, TokenStream(toks)) => toks, _ => panic!() };
    let parse_result = parse("CSMAIN".to_string(), &tokens[..]);
    assert!(!parse_result.is_none());

    let ast = match parse_result { Some(ast) => ast, _ => panic!() };
    let ir_result = ast_to_ir::parse(&ast);
    assert!(ir_result.is_ok());
}
