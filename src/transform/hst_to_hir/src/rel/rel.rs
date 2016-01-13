
//! # Rel: Reduced Expression Language
//!
//! Minilanguage for reducing expressions into parts

use slp_lang_hir as hir;
use std::collections::HashSet;

pub use slp_lang_hir::InputModifier;

/// Id to reference a variable binding within a rel sequence
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct BindId(pub u64);

#[derive(PartialEq, Debug, Clone)]
pub struct ConstructorSlot {
    pub arity: u32,
    pub expr: BindId,
}

/// When the commands value will change
#[derive(PartialEq, Debug, Clone)]
pub enum MutationParam {
    /// When anything happens (most general form of a list of variables)
    Mutable,
    /// Never (constant expressions)
    Const,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Command {
    Literal(hir::Literal),
    Variable(hir::VariableRef),
    Global(hir::GlobalId),
    ConstantVariable(hir::ConstantBufferId, String),
    TernaryConditional(BindId, Box<Sequence>, Box<Sequence>),
    Swizzle(BindId, Vec<hir::SwizzleSlot>),
    ArraySubscript(BindId, BindId),
    Texture2DIndex(hir::DataType, BindId, BindId),
    RWTexture2DIndex(hir::DataType, BindId, BindId),
    Member(BindId, String),
    Call(hir::FunctionId, Vec<BindId>),
    NumericConstructor(hir::DataLayout, Vec<ConstructorSlot>),
    Cast(hir::Type, BindId),
    Intrinsic0(hir::Intrinsic0),
    Intrinsic1(hir::Intrinsic1, BindId),
    Intrinsic2(hir::Intrinsic2, BindId, BindId),
    Intrinsic3(hir::Intrinsic3, BindId, BindId, BindId),
    Trivial(hir::Expression, MutationParam),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Bind {
    pub id: BindId,
    pub value: Command,
    pub required_input: InputModifier,
    pub ty: hir::Type,
}

impl Bind {
    #[cfg(test)]
    pub fn direct(id: u64, command: Command, ty: hir::Type) -> Bind {
        Bind {
            id: BindId(id),
            value: command,
            required_input: InputModifier::In,
            ty: ty,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Sequence {
    pub binds: Vec<Bind>,
    pub last: BindId,
}

impl Sequence {
    pub fn find_used_locals(&self) -> HashSet<hir::VariableRef> {
        let mut used = HashSet::new();
        for bind in &self.binds {
            match bind.value {
                Command::Variable(ref var) => {
                    used.insert(var.clone());
                }
                Command::TernaryConditional(_, ref lhs, ref rhs) => {
                    for var in lhs.find_used_locals() {
                        used.insert(var);
                    }
                    for var in rhs.find_used_locals() {
                        used.insert(var);
                    }
                }
                Command::Trivial(ref hir, _) => hir_find_used_locals(hir, &mut used),
                _ => {}
            }
        }
        used
    }
}

pub fn hir_find_used_locals(expr: &hir::Expression, set: &mut HashSet<hir::VariableRef>) {
    use slp_lang_hir::Expression;
    match *expr {
        Expression::Literal(_) => {}
        Expression::Variable(ref var_ref) => {
            set.insert(var_ref.clone());
        }
        Expression::Global(_) => {}
        Expression::ConstantVariable(_, _) => {}
        Expression::TernaryConditional(ref cond, ref left, ref right) => {
            hir_find_used_locals(cond, set);
            hir_find_used_locals(left, set);
            hir_find_used_locals(right, set);
        }
        Expression::Swizzle(ref vec, _) => {
            hir_find_used_locals(vec, set);
        }
        Expression::ArraySubscript(ref arr, ref index) => {
            hir_find_used_locals(arr, set);
            hir_find_used_locals(index, set);
        }
        Expression::Member(ref expr, _) => hir_find_used_locals(expr, set),
        Expression::Call(_, ref exprs) => {
            for expr in exprs {
                hir_find_used_locals(expr, set);
            }
        }
        Expression::NumericConstructor(_, ref elements) => {
            for element in elements {
                hir_find_used_locals(&element.expr, set);
            }
        }
        Expression::Cast(_, ref expr) => hir_find_used_locals(expr, set),
        Expression::Intrinsic0(_) => {}
        Expression::Intrinsic1(_, ref e1) => {
            hir_find_used_locals(e1, set);
        }
        Expression::Intrinsic2(_, ref e1, ref e2) => {
            hir_find_used_locals(e1, set);
            hir_find_used_locals(e2, set);
        }
        Expression::Intrinsic3(_, ref e1, ref e2, ref e3) => {
            hir_find_used_locals(e1, set);
            hir_find_used_locals(e2, set);
            hir_find_used_locals(e3, set);
        }
    }
}
