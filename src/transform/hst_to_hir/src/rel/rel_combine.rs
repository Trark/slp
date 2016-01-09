
//! Combine transform to turn rel expressions into hir expressions

use super::rel::*;
use slp_lang_hir as hir;
use std::error;
use std::fmt;

#[derive(PartialEq, Debug, Clone)]
pub enum CombineError {
    FailedToResolveMultiPartExpression,
}

pub type CombineResult<T> = Result<T, CombineError>;

impl error::Error for CombineError {
    fn description(&self) -> &str {
        match *self {
            CombineError::FailedToResolveMultiPartExpression => {
                "expression too complex to resolve (rel parser internal error)"
            }
        }
    }
}

impl fmt::Display for CombineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

struct BuiltQueue(Vec<(BindId, hir::Expression)>);

impl BuiltQueue {
    fn new() -> BuiltQueue {
        BuiltQueue(Vec::new())
    }
    fn take(mut self, id: &BindId) -> Option<(BuiltQueue, hir::Expression)> {
        match self.0.pop() {
            Some((pop_id, expr)) => {
                if *id == pop_id {
                    Some((self, expr))
                } else {
                    None
                }
            }
            None => None,
        }
    }
    fn with_next(mut self, id: BindId, expr: hir::Expression) -> BuiltQueue {
        self.0.push((id, expr));
        self
    }
    fn empty(&self) -> bool {
        self.0.len() == 0
    }
}


#[derive(PartialEq, Debug, Clone)]
pub enum CombinedExpression {
    Single(hir::Expression),
    Multi(Vec<hir::Statement>),
}

fn combine_trivial(seq: &Sequence) -> Option<hir::Expression> {

    fn build_command(command: &Command, q: BuiltQueue) -> Option<(BuiltQueue, hir::Expression)> {
        match *command {
            Command::Literal(ref lit) => Some((q, hir::Expression::Literal(lit.clone()))),
            Command::Variable(ref var) => Some((q, hir::Expression::Variable(var.clone()))),
            Command::Global(ref id) => Some((q, hir::Expression::Global(id.clone()))),
            Command::ConstantVariable(ref id, ref name) => {
                let var = hir::Expression::ConstantVariable(id.clone(), name.clone());
                Some((q, var))
            }
            Command::Swizzle(ref val, ref swizzle) => {
                let (q, e) = match q.take(val) {
                    Some(res) => res,
                    None => return None,
                };
                Some((q, hir::Expression::Swizzle(Box::new(e), swizzle.clone())))
            }
            Command::ArraySubscript(ref arr_val, ref ind_val) => {
                let (q, index) = match q.take(ind_val) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let (q, arr) = match q.take(arr_val) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                Some((q, hir::Expression::ArraySubscript(arr, index)))
            }
            Command::TextureIndex(_, _, _) => None,
            Command::Member(ref val, ref name) => {
                let (q, e) = match q.take(val) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                Some((q, hir::Expression::Member(e, name.clone())))
            }
            Command::Call(ref id, ref params) => {
                let mut exprs = vec![];
                let mut q = q;
                for param in params.iter().rev() {
                    match q.take(param) {
                        Some((next_q, e)) => {
                            q = next_q;
                            exprs.push(e);
                        }
                        None => return None,
                    };
                }
                let exprs = exprs.into_iter().rev().collect::<Vec<_>>();
                Some((q, hir::Expression::Call(id.clone(), exprs)))
            }
            Command::NumericConstructor(ref dtyl, ref cons_slot) => {
                let mut slots = vec![];
                let mut q = q;
                for slot in cons_slot.iter().rev() {
                    match q.take(&slot.expr) {
                        Some((next_q, e)) => {
                            q = next_q;
                            let out = hir::ConstructorSlot {
                                arity: slot.arity,
                                expr: e,
                            };
                            slots.push(out);
                        }
                        None => return None,
                    };
                }
                let slots = slots.into_iter().rev().collect::<Vec<_>>();
                Some((q, hir::Expression::NumericConstructor(dtyl.clone(), slots)))
            }
            Command::Cast(ref ty, ref val) => {
                let (q, e) = match q.take(val) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                Some((q, hir::Expression::Cast(ty.clone(), e)))
            }
            Command::Intrinsic0(ref i) => {
                let r = hir::Expression::Intrinsic0(i.clone());
                Some((q, r))
            }
            Command::Intrinsic1(ref i, ref p1) => {
                let (q, e1) = match q.take(p1) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let r = hir::Expression::Intrinsic1(i.clone(), e1);
                Some((q, r))
            }
            Command::Intrinsic2(ref i, ref p1, ref p2) => {
                let (q, e2) = match q.take(p2) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let (q, e1) = match q.take(p1) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let r = hir::Expression::Intrinsic2(i.clone(), e1, e2);
                Some((q, r))
            }
            Command::Intrinsic3(ref i, ref p1, ref p2, ref p3) => {
                let (q, e3) = match q.take(p3) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let (q, e2) = match q.take(p2) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let (q, e1) = match q.take(p1) {
                    Some((q, e)) => (q, Box::new(e)),
                    None => return None,
                };
                let r = hir::Expression::Intrinsic3(i.clone(), e1, e2, e3);
                Some((q, r))
            }
        }
    }

    fn build_bind_type(bind_type: &BindType,
                       q: BuiltQueue)
                       -> Option<(BuiltQueue, hir::Expression)> {
        match *bind_type {
            BindType::Direct(ref command) => build_command(command, q),
            BindType::Select(ref cond, ref lhs, ref rhs) => {
                let (q, cond_val) = match q.take(cond) {
                    Some(res) => res,
                    None => return None,
                };
                let lhs_val = match combine_trivial(lhs) {
                    Some(lhs_val) => lhs_val,
                    None => return None,
                };
                let rhs_val = match combine_trivial(rhs) {
                    Some(rhs_val) => rhs_val,
                    None => return None,
                };
                let cond_box = Box::new(cond_val);
                let lhs_val = Box::new(lhs_val);
                let rhs_val = Box::new(rhs_val);
                let expr = hir::Expression::TernaryConditional(cond_box, lhs_val, rhs_val);
                Some((q, expr))
            }
        }
    }

    fn build_bind(bind: &Bind, q: BuiltQueue) -> Option<BuiltQueue> {
        match build_bind_type(&bind.bind_type, q) {
            Some((q, e)) => Some(q.with_next(bind.id, e)),
            None => None,
        }
    }

    let mut queue = BuiltQueue::new();
    for bind in &seq.binds {
        queue = match build_bind(bind, queue) {
            Some(q) => q,
            None => return None,
        };
    }

    match build_bind_type(&seq.last, queue) {
        Some((q, e)) => {
            if q.empty() {
                Some(e)
            } else {
                None
            }
        }
        None => None,
    }
}

fn combine_complex(_: Sequence) -> CombineResult<Vec<hir::Statement>> {
    Err(CombineError::FailedToResolveMultiPartExpression)
}

pub fn combine(seq: Sequence) -> CombineResult<CombinedExpression> {
    match combine_trivial(&seq) {
        Some(exp) => return Ok(CombinedExpression::Single(exp)),
        None => {}
    }
    let parts = try!(combine_complex(seq));
    Ok(CombinedExpression::Multi(parts))
}

#[test]
fn test_combine_single_variable() {

    let var_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_seq = Sequence {
        binds: vec![],
        last: BindType::Direct(Command::Variable(var_ref.clone())),
    };
    let var_expr = combine(var_seq).expect("combine failed");
    let var_0 = hir::Expression::Variable(var_ref);
    assert_eq!(var_expr, CombinedExpression::Single(var_0));
}

#[test]
fn test_combine_binary_operation() {
    let var_0_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_1_ref = hir::VariableRef(hir::VariableId(1), hir::ScopeRef(0));
    let dty = hir::DataType(hir::DataLayout::Scalar(hir::ScalarType::Float),
                            hir::TypeModifier::default());
    let add = hir::Intrinsic2::Add(dty.clone());
    let bin_seq = Sequence {
        binds: vec![
            Bind::direct(0, Command::Variable(var_0_ref.clone())),
            Bind::direct(1, Command::Variable(var_1_ref.clone())),
        ],
        last: BindType::Direct(Command::Intrinsic2(add.clone(), BindId(0), BindId(1))),
    };
    let bin_expr = combine(bin_seq).expect("combine failed");
    let var_0 = hir::Expression::Variable(var_0_ref);
    let var_1 = hir::Expression::Variable(var_1_ref);
    let add = hir::Expression::Intrinsic2(add, Box::new(var_0), Box::new(var_1));
    assert_eq!(bin_expr, CombinedExpression::Single(add));
}
