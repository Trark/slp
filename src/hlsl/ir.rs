
use std::collections::HashMap;
use super::ast;

pub use super::ast::ScalarType as ScalarType;
pub use super::ast::DataType as DataType;
pub use super::ast::StructuredType as StructuredType;
pub use super::ast::ObjectType as ObjectType;
pub use super::ast::FunctionType as FunctionType;
pub use super::ast::Type as Type;
pub use super::ast::BinOp as BinOp;
pub use super::ast::UnaryOp as UnaryOp;

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic {
    Texture2DLoad,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    LiteralInt(u64),
    LiteralUint(u64),
    LiteralLong(u64),
    LiteralFloat(f32),
    LiteralDouble(f64),
    Variable(String),
    UnaryOperation(UnaryOp, Box<Expression>),
    BinaryOperation(BinOp, Box<Expression>, Box<Expression>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(Box<Expression>, Vec<Expression>),
    Cast(Type, Box<Expression>),
    Intrinsic(Intrinsic),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarDef {
    pub name: String,
    pub typename: Type,
    pub assignment: Option<Expression>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Condition {
    Expr(Expression),
    Assignment(VarDef),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Expression(Expression),
    Var(VarDef),
    Block(Vec<Statement>),
    If(Condition, Box<Statement>),
    For(Condition, Condition, Condition, Box<Statement>),
    While(Condition, Box<Statement>),
    Return(Expression),
}

pub use super::ast::StructMember as StructMember;
pub use super::ast::StructDefinition as StructDefinition;

pub use super::ast::PackSubOffset as PackSubOffset;
pub use super::ast::PackOffset as PackOffset;
pub use super::ast::ConstantVariable as ConstantVariable;

#[derive(PartialEq, Debug, Clone)]
pub struct ConstantBuffer {
    pub name: String,
    pub members: Vec<ConstantVariable>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalVariable {
    pub name: String,
    pub typename: Type,
}

pub use super::ast::FunctionAttribute as FunctionAttribute;
pub use super::ast::FunctionParam as FunctionParam;

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub returntype: Type,
    pub params: Vec<FunctionParam>,
    pub body: Vec<Statement>,
    pub attributes: Vec<FunctionAttribute>,
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
pub struct GlobalEntry {
    pub name: String,
    pub typename: ast::Type,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalTable {
    pub r_resources: HashMap<u32, GlobalEntry>,
    pub rw_resources: HashMap<u32, GlobalEntry>,
    pub samplers: HashMap<u32, String>,
    pub constants: HashMap<u32, String>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Module {
    pub entry_point: String,
    pub global_table: GlobalTable,
    pub root_definitions: Vec<RootDefinition>,
}

impl GlobalTable {

    pub fn new() -> GlobalTable {
        GlobalTable {
            r_resources: HashMap::new(),
            rw_resources: HashMap::new(),
            samplers: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    pub fn generate(module: &ast::Module) -> GlobalTable {
        let mut table = GlobalTable::new();
        for root_def in &module.root_definitions {
            match root_def {
                &ast::RootDefinition::ConstantBuffer(ref cbuffer) => {
                    match cbuffer.slot {
                        Some(ref slot_index) => {
                            table.constants.insert(slot_index.0, cbuffer.name.clone());
                        },
                        None => { },
                    }
                },
                &ast::RootDefinition::GlobalVariable(ref global) => {
                    let entry = GlobalEntry { name: global.name.clone(), typename: global.typename.clone() };
                    match global.slot {
                        Some(ast::GlobalSlot::ReadSlot(ref slot_index)) => {
                            table.r_resources.insert(*slot_index, entry);
                        },
                        Some(ast::GlobalSlot::ReadWriteSlot(ref slot_index)) => {
                            table.rw_resources.insert(*slot_index, entry);
                        },
                        None => { },
                    }
                },
                _ => { }
            }
        };
        table
    }
}

