
#[derive(PartialEq, Debug, Clone)]
pub struct TypeName(pub String);

#[derive(PartialEq, Debug, Clone)]
pub enum BinOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    Assignment,
}

#[derive(PartialEq, Debug, Clone)]
pub enum UnaryOp {
    PrefixIncrement,
    PrefixDecrement,
    PostfixIncrement,
    PostfixDecrement,
    Plus,
    Minus,
    LogicalNot,
    BitwiseNot,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    LiteralUint(u32),
    LiteralInt(i32),
    LiteralLong(i64),
    LiteralFloat(f32),
    LiteralDouble(f64),
    Variable(String),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(Box<Expression>, Vec<Expression>),
    Cast(TypeName, Box<Expression>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub name: String,
    pub typename: TypeName,
    pub assignment: Option<Expression>,
}

impl VarDef {
    pub fn new(name: String, typename: TypeName, assignment: Option<Expression>) -> VarDef {
        VarDef { name: name, typename: typename, assignment: assignment }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Condition {
    Expr(Expression),
    Assignment(VarDef),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Empty,
    Expression(Expression),
    Var(VarDef),
    Block(Vec<Statement>),
    If(Condition, Box<Statement>),
    For(Condition, Condition, Condition, Box<Statement>),
    While(Condition, Box<Statement>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructMember {
    pub name: String,
    pub typename: TypeName,
}

#[derive(PartialEq, Debug, Clone)]
pub struct StructDefinition {
    pub name: TypeName,
    pub members: Vec<StructMember>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantSlot(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub enum PackSubOffset {
    None,
    X,
    Y,
    Z,
    W,
}

#[derive(PartialEq, Debug, Clone)]
pub struct PackOffset(pub u32, pub PackSubOffset);

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantVariable {
    pub name: String,
    pub typename: TypeName,
    pub offset: Option<PackOffset>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantBuffer {
    pub name: String,
    pub slot: Option<ConstantSlot>,
    pub members: Vec<ConstantVariable>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct SamplerSlot(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub struct ReadResourceSlot(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub struct ReadWriteResourceSlot(pub u32);

#[derive(PartialEq, Debug, Clone)]
pub enum GlobalSlot {
    ReadSlot(u32),
    ReadWriteSlot(u32),
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub name: String,
    pub typename: TypeName,
    pub slot: Option<GlobalSlot>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionParam {
    pub name: String,
    pub typename: TypeName,
}

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub returntype: TypeName,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum RootDefinition {
    Struct(StructDefinition),
    SamplerState,
    ConstantBuffer(ConstantBuffer),
    GlobalVariable(GlobalVariable),
    Function(FunctionDefinition),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Module {
    pub entry_point: String,
    pub root_definitions: Vec<RootDefinition>,
}
