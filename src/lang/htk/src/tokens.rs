
use slp_shared::*;

#[derive(PartialEq, Debug, Clone)]
pub struct Identifier(pub String);

#[derive(PartialEq, Debug, Clone)]
pub enum FollowedBy {
    Token,
    Whitespace,
}

#[derive(PartialEq, Debug, Clone)]
pub enum RegisterSlot {
    T(u32),
    U(u32),
    B(u32),
}

#[derive(PartialEq, Debug, Clone)]
pub enum OffsetSlot {
    T(u32),
    U(u32),
    B(u32),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    Eof, // Marks the end of a stream

    Id(Identifier),
    LiteralInt(u64), // Int (Hlsl ints do not have sign, the - is an operator on the literal)
    LiteralUInt(u64), // Int with explicit unsigned type
    LiteralLong(u64), // Int with explicit long type
    LiteralHalf(f32),
    LiteralFloat(f32),
    LiteralDouble(f64),
    True,
    False,

    LeftBrace,
    RightBrace,
    LeftParen,
    RightParen,
    LeftSquareBracket,
    RightSquareBracket,
    LeftAngleBracket(FollowedBy),
    RightAngleBracket(FollowedBy),
    Semicolon,
    Comma,
    QuestionMark,

    Plus,
    Minus,
    ForwardSlash,
    Percent,
    Asterix,
    VerticalBar(FollowedBy),
    Ampersand(FollowedBy),
    Hat,
    Equals,
    Hash,
    At,
    ExclamationPoint,
    Tilde,
    Period,
    DoubleEquals,
    ExclamationEquals,

    If,
    Else,
    For,
    While,
    Switch,
    Return,

    Struct,
    SamplerState,
    ConstantBuffer,
    Register(RegisterSlot),
    PackOffset(OffsetSlot),
    Colon,

    In,
    Out,
    InOut,

    Const,

    Extern,
    Static,
    GroupShared,

    Auto,
    Case,
    Catch,
    Char,
    Class,
    ConstCast,
    Default,
    Delete,
    DynamicCast,
    Enum,
    Explicit,
    Friend,
    Goto,
    Long,
    Mutable,
    New,
    Operator,
    Private,
    Protected,
    Public,
    ReinterpretCast,
    Short,
    Signed,
    SizeOf,
    StaticCast,
    Template,
    This,
    Throw,
    Try,
    Typename,
    Union,
    Unsigned,
    Using,
    Virtual,
}

#[derive(PartialEq, Debug, Clone)]
pub struct LexToken(pub Token, pub FileLocation);

impl LexToken {
    pub fn to_loc(self) -> FileLocation {
        self.1
    }
    pub fn with_no_loc(token: Token) -> LexToken {
        LexToken(token, FileLocation::none())
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Tokens {
    pub stream: Vec<LexToken>,
}
