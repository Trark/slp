
use super::super::tokens::Identifier;
use super::super::tokens::Tokens;
use super::super::tokens::Token;
use super::super::tokens::StreamToken;
use super::super::tokens::StreamLocation;
use super::super::tokens::FollowedBy;
use super::super::tokens::RegisterSlot;
use super::super::lexer::lex;
use super::super::parser::parse;
use super::super::typer::typeparse;

fn token_id(name: &'static str) -> Token { Token::Id(Identifier(name.to_string())) }

// Test a small compute shader (A simplified form of one the basic hlsl compute examples)
const CS1: &'static [u8] = include_bytes!("cs1.hlsl");

#[test]
fn cs1_lex() {

    // Normalise line ending so we don't have to deal with git potentially
    // changing them
    let cs1_str = String::from_utf8(CS1.to_vec()).unwrap().replace("\r\n", "\n");
    let cs1_norm = cs1_str.as_bytes();

    let tokens_res = lex(cs1_norm);

    let expected_tokens_data = [
        (Token::Struct, 1),
        (token_id("BufType"), 8),
        (Token::LeftBrace, 16),
        (token_id("int"), 22),
        (token_id("i"), 26),
        (Token::Semicolon, 27),
        (token_id("float"), 33),
        (token_id("f"), 39),
        (Token::Semicolon, 40),
        (Token::RightBrace, 42),
        (Token::Semicolon, 43),
        (token_id("StructuredBuffer"), 46),
        (Token::LeftAngleBracket(FollowedBy::Token), 62),
        (token_id("BufType"), 63),
        (Token::RightAngleBracket(FollowedBy::Whitespace), 70),
        (token_id("Buffer0"), 72),
        (Token::Colon, 80),
        (Token::Register(RegisterSlot::T(0)), 82),
        (Token::Semicolon, 94),
        (token_id("StructuredBuffer"), 96),
        (Token::LeftAngleBracket(FollowedBy::Token), 112),
        (token_id("BufType"), 113),
        (Token::RightAngleBracket(FollowedBy::Whitespace), 120),
        (token_id("Buffer1"), 122),
        (Token::Colon, 130),
        (Token::Register(RegisterSlot::T(1)), 132),
        (Token::Semicolon, 144),
        (token_id("RWStructuredBuffer"), 146),
        (Token::LeftAngleBracket(FollowedBy::Token), 164),
        (token_id("BufType"), 165),
        (Token::RightAngleBracket(FollowedBy::Whitespace), 172),
        (token_id("BufferOut"), 174),
        (Token::Colon, 184),
        (Token::Register(RegisterSlot::U(0)), 186),
        (Token::Semicolon, 198),
        (Token::LeftSquareBracket, 201),
        (token_id("numthreads"), 202),
        (Token::LeftParen, 212),
        (Token::LiteralInt(1), 213),
        (Token::Comma, 214),
        (Token::LiteralInt(1), 216),
        (Token::Comma, 217),
        (Token::LiteralInt(1), 219),
        (Token::RightParen, 220),
        (Token::RightSquareBracket, 221),
        (token_id("void"), 223),
        (token_id("CSMain"), 228),
        (Token::LeftParen, 234),
        (token_id("uint3"), 236),
        (token_id("DTid"), 242),
        (Token::Colon, 247),
        (token_id("SV_DispatchThreadID"), 249),
        (Token::RightParen, 269),
        (Token::LeftBrace, 271),
        (token_id("BufferOut"), 277),
        (Token::LeftSquareBracket, 286),
        (token_id("DTid"), 287),
        (Token::Period, 291),
        (token_id("x"), 292),
        (Token::RightSquareBracket, 293),
        (Token::Period, 294),
        (token_id("i"), 295),
        (Token::Equals, 297),
        (token_id("Buffer0"), 299),
        (Token::LeftSquareBracket, 306),
        (token_id("DTid"), 307),
        (Token::Period, 311),
        (token_id("x"), 312),
        (Token::RightSquareBracket, 313),
        (Token::Period, 314),
        (token_id("i"), 315),
        (Token::Plus, 317),
        (token_id("Buffer1"), 319),
        (Token::LeftSquareBracket, 326),
        (token_id("DTid"), 327),
        (Token::Period, 331),
        (token_id("x"), 332),
        (Token::RightSquareBracket, 333),
        (Token::Period, 334),
        (token_id("i"), 335),
        (Token::Semicolon, 336),
        (token_id("BufferOut"), 342),
        (Token::LeftSquareBracket, 351),
        (token_id("DTid"), 352),
        (Token::Period, 356),
        (token_id("x"), 357),
        (Token::RightSquareBracket, 358),
        (Token::Period, 359),
        (token_id("f"), 360),
        (Token::Equals, 362),
        (token_id("Buffer0"), 364),
        (Token::LeftSquareBracket, 371),
        (token_id("DTid"), 372),
        (Token::Period, 376),
        (token_id("x"), 377),
        (Token::RightSquareBracket, 378),
        (Token::Period, 379),
        (token_id("f"), 380),
        (Token::Plus, 382),
        (token_id("Buffer1"), 384),
        (Token::LeftSquareBracket, 391),
        (token_id("DTid"), 392),
        (Token::Period, 396),
        (token_id("x"), 397),
        (Token::RightSquareBracket, 398),
        (Token::Period, 399),
        (token_id("f"), 400),
        (Token::Semicolon, 401),
        (Token::RightBrace, 403),
        (Token::Eof, 404),
    ];
    let expected_tokens = Tokens { stream: expected_tokens_data.iter().map(|&(ref tok, ref loc)| StreamToken(tok.clone(), StreamLocation(*loc))).collect::<Vec<_>>() };

    match tokens_res {
        Ok(tokens) => {
            for (lexed_stoken, &(ref expected_token, _)) in tokens.stream.iter().zip(expected_tokens_data.iter()) {
                let lexed_token: &Token = &lexed_stoken.0;
                let expected_token: &Token = expected_token;
                assert_eq!(lexed_token, expected_token);
            }
            for (lexed_stoken, &(_, ref expected_location)) in tokens.stream.iter().zip(expected_tokens_data.iter()) {
                let lexed_loc: &StreamLocation = &lexed_stoken.1;
                let expected_loc: u64 = *expected_location;
                assert_eq!(lexed_loc.0, expected_loc);
            }
            assert_eq!(tokens, expected_tokens);
        }
        Err(err) => assert!(false, "Failed to lex cs1: {}", err),
    }
}

#[test]
fn cs1_parse() {
    let tokens = lex(CS1).unwrap();
    let parse_result = parse("CSMain".to_string(), &tokens.get_nonstream_tokens());
    assert!(parse_result.is_ok(), "{:?}", parse_result);
}

#[test]
fn cs1_typecheck() {
    let tokens = lex(CS1).unwrap();
    let ast = parse("CSMain".to_string(), &tokens.get_nonstream_tokens()).unwrap();
    let ir_result = typeparse(&ast);
    assert!(ir_result.is_ok(), "{:?}", ir_result);
}
