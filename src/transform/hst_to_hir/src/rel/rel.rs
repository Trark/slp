
//! # Rel: Reduced Expression Language
//!
//! Minilanguage for reducing expressions into parts

use slp_lang_hir as hir;

pub use slp_lang_hir::InputModifier;

/// Id to reference a variable binding within a rel sequence
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct BindId(pub u64);

#[derive(PartialEq, Debug, Clone)]
pub struct ConstructorSlot {
    pub arity: u32,
    pub expr: BindId,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Command {
    Literal(hir::Literal),
    Variable(hir::VariableRef),
    Global(hir::GlobalId),
    ConstantVariable(hir::ConstantBufferId, String),
    Swizzle(BindId, Vec<hir::SwizzleSlot>),
    ArraySubscript(BindId, BindId),
    TextureIndex(hir::DataType, BindId, BindId),
    Member(BindId, String),
    Call(hir::FunctionId, Vec<BindId>),
    NumericConstructor(hir::DataLayout, Vec<ConstructorSlot>),
    Cast(hir::Type, BindId),
    Intrinsic0(hir::Intrinsic0),
    Intrinsic1(hir::Intrinsic1, BindId),
    Intrinsic2(hir::Intrinsic2, BindId, BindId),
    Intrinsic3(hir::Intrinsic3, BindId, BindId, BindId),
}

#[derive(PartialEq, Debug, Clone)]
pub enum BindType {
    /// Bind takes the BindId of a command
    Direct(Command),
    /// Bind conditionally takes either one set of commands or another
    Select(BindId, Box<Sequence>, Box<Sequence>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Bind {
    pub id: BindId,
    pub bind_type: BindType,
    pub required_input: InputModifier,
}

impl Bind {
    #[cfg(test)]
    pub fn direct(id: u64, command: Command) -> Bind {
        Bind {
            id: BindId(id),
            bind_type: BindType::Direct(command),
            required_input: InputModifier::In,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Sequence {
    pub binds: Vec<Bind>,
    pub last: BindType,
}
