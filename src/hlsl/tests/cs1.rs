
use FileLocation;
use File;
use Line;
use Column;
use super::super::preprocess::*;
use super::super::tokens::*;
use super::super::lexer::lex;
use super::super::parser::parse;
use super::super::typer::typeparse;

fn token_id(name: &'static str) -> Token { Token::Id(Identifier(name.to_string())) }

// Test a small compute shader (A simplified form of one the basic hlsl compute examples)
const CS1: &'static str = include_str!("cs1.hlsl");

#[test]
fn cs1_lex() {

    // Normalise line ending so we don't have to deal with git potentially
    // changing them
    let cs1_str = CS1.to_string().replace("\r\n", "\n");

    let cs1_preprocessed = preprocess(&cs1_str).expect("cs1 failed preprocess");

    let tokens_res = lex(&cs1_preprocessed);

    let expected_tokens_data = [
        (Token::Struct, 2, 1),
        (token_id("BufType"), 2, 8),
        (Token::LeftBrace, 3, 1),
        (token_id("int"), 4, 5),
        (token_id("i"), 4, 9),
        (Token::Semicolon, 4, 10),
        (token_id("float"), 5, 5),
        (token_id("f"), 5, 11),
        (Token::Semicolon, 5, 12),
        (Token::RightBrace, 6, 1),
        (Token::Semicolon, 6, 2),
        (token_id("StructuredBuffer"), 8, 1),
        (Token::LeftAngleBracket(FollowedBy::Token), 8, 17),
        (token_id("BufType"), 8, 18),
        (Token::RightAngleBracket(FollowedBy::Whitespace), 8, 25),
        (token_id("Buffer0"), 8, 27),
        (Token::Colon, 8, 35),
        (Token::Register(RegisterSlot::T(0)), 8, 37),
        (Token::Semicolon, 8, 49),
        (token_id("StructuredBuffer"), 9, 1),
        (Token::LeftAngleBracket(FollowedBy::Token), 9, 17),
        (token_id("BufType"), 9, 18),
        (Token::RightAngleBracket(FollowedBy::Whitespace), 9, 25),
        (token_id("Buffer1"), 9, 27),
        (Token::Colon, 9, 35),
        (Token::Register(RegisterSlot::T(1)), 9, 37),
        (Token::Semicolon, 9, 49),
        (token_id("RWStructuredBuffer"), 10, 1),
        (Token::LeftAngleBracket(FollowedBy::Token), 10, 19),
        (token_id("BufType"), 10, 20),
        (Token::RightAngleBracket(FollowedBy::Whitespace), 10, 27),
        (token_id("BufferOut"), 10, 29),
        (Token::Colon, 10, 39),
        (Token::Register(RegisterSlot::U(0)), 10, 41),
        (Token::Semicolon, 10, 53),
        (Token::LeftSquareBracket, 12, 1),
        (token_id("numthreads"), 12, 2),
        (Token::LeftParen, 12, 12),
        (Token::LiteralInt(1), 12, 13),
        (Token::Comma, 12, 14),
        (Token::LiteralInt(1), 12, 16),
        (Token::Comma, 12, 17),
        (Token::LiteralInt(1), 12, 19),
        (Token::RightParen, 12, 20),
        (Token::RightSquareBracket, 12, 21),
        (token_id("void"), 13, 1),
        (token_id("CSMain"), 13, 6),
        (Token::LeftParen, 13, 12),
        (token_id("uint3"), 13, 14),
        (token_id("DTid"), 13, 20),
        (Token::Colon, 13, 25),
        (token_id("SV_DispatchThreadID"), 13, 27),
        (Token::RightParen, 13, 47),
        (Token::LeftBrace, 14, 1),
        (token_id("BufferOut"), 15, 5),
        (Token::LeftSquareBracket, 15, 14),
        (token_id("DTid"), 15, 15),
        (Token::Period, 15, 19),
        (token_id("x"), 15, 20),
        (Token::RightSquareBracket, 15, 21),
        (Token::Period, 15, 22),
        (token_id("i"), 15, 23),
        (Token::Equals, 15, 25),
        (token_id("Buffer0"), 15, 27),
        (Token::LeftSquareBracket, 15, 34),
        (token_id("DTid"), 15, 35),
        (Token::Period, 15, 39),
        (token_id("x"), 15, 40),
        (Token::RightSquareBracket, 15, 41),
        (Token::Period, 15, 42),
        (token_id("i"), 15, 43),
        (Token::Plus, 15, 45),
        (token_id("Buffer1"), 15, 47),
        (Token::LeftSquareBracket, 15, 54),
        (token_id("DTid"), 15, 55),
        (Token::Period, 15, 59),
        (token_id("x"), 15, 60),
        (Token::RightSquareBracket, 15, 61),
        (Token::Period, 15, 62),
        (token_id("i"), 15, 63),
        (Token::Semicolon, 15, 64),
        (token_id("BufferOut"), 16, 5),
        (Token::LeftSquareBracket, 16, 14),
        (token_id("DTid"), 16, 15),
        (Token::Period, 16, 19),
        (token_id("x"), 16, 20),
        (Token::RightSquareBracket, 16, 21),
        (Token::Period, 16, 22),
        (token_id("f"), 16, 23),
        (Token::Equals, 16, 25),
        (token_id("Buffer0"), 16, 27),
        (Token::LeftSquareBracket, 16, 34),
        (token_id("DTid"), 16, 35),
        (Token::Period, 16, 39),
        (token_id("x"), 16, 40),
        (Token::RightSquareBracket, 16, 41),
        (Token::Period, 16, 42),
        (token_id("f"), 16, 43),
        (Token::Plus, 16, 45),
        (token_id("Buffer1"), 16, 47),
        (Token::LeftSquareBracket, 16, 54),
        (token_id("DTid"), 16, 55),
        (Token::Period, 16, 59),
        (token_id("x"), 16, 60),
        (Token::RightSquareBracket, 16, 61),
        (Token::Period, 16, 62),
        (token_id("f"), 16, 63),
        (Token::Semicolon, 16, 64),
        (Token::RightBrace, 17, 1),
        (Token::Eof, 17, 2),
    ];
    let expected_tokens = Tokens { stream: expected_tokens_data.iter().map(|&(ref tok, ref line, ref column)|
        LexToken(tok.clone(), FileLocation(File::Unknown, Line(*line), Column(*column)))
    ).collect::<Vec<_>>()};

    match tokens_res {
        Ok(tokens) => {
            for (lexed_ftoken, &(ref expected_token, _, _)) in tokens.stream.iter().zip(expected_tokens_data.iter()) {
                let lexed_token: &Token = &lexed_ftoken.0;
                let expected_token: &Token = expected_token;
                assert_eq!(lexed_token, expected_token);
            }
            for (lexed_ftoken, &(_, ref expected_line, ref expected_column)) in tokens.stream.iter().zip(expected_tokens_data.iter()) {
                let lexed_loc = &lexed_ftoken.1;
                let expected_loc = FileLocation(File::Unknown, Line(*expected_line), Column(*expected_column));
                assert_eq!(*lexed_loc, expected_loc);
            }
            assert_eq!(tokens, expected_tokens);
        }
        Err(err) => assert!(false, "Failed to lex cs1: {}", err),
    }
}

#[test]
fn cs1_parse() {
    let tokens = lex(&preprocess(CS1).unwrap()).unwrap();
    let parse_result = parse("CSMain".to_string(), &tokens.get_nonstream_tokens());
    assert!(parse_result.is_ok(), "{:?}", parse_result);
}

#[test]
fn cs1_typecheck() {
    let tokens = lex(&preprocess(CS1).unwrap()).unwrap();
    let ast = parse("CSMain".to_string(), &tokens.get_nonstream_tokens()).unwrap();
    let ir_result = typeparse(&ast);
    assert!(ir_result.is_ok(), "{:?}", ir_result);
}
