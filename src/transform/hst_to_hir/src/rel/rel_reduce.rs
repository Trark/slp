
//! Reduce transform to turn pel expressions into rel expressions

use super::rel::*;
use pel;
use typer::FunctionOverload;
use slp_lang_hir as hir;
use std::error;
use std::fmt;

#[derive(PartialEq, Debug, Clone)]
pub enum ReduceError {
    EmptyReduction,
}

pub type ReduceResult<T> = Result<T, ReduceError>;

impl error::Error for ReduceError {
    fn description(&self) -> &str {
        match *self {
            ReduceError::EmptyReduction => "reduced expression is empty",
        }
    }
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

pub trait ReduceContext {
    fn find_overload(&self, id: &hir::FunctionId) -> FunctionOverload;
}

struct SequenceBuilder {
    binds: Vec<Bind>,
}

impl SequenceBuilder {
    fn new() -> SequenceBuilder {
        SequenceBuilder { binds: vec![] }
    }

    fn push(mut self, bt: BindType, im: InputModifier) -> (SequenceBuilder, BindId) {
        let id = BindId(self.binds.len() as u64);
        self.binds.push(Bind {
            id: id.clone(),
            bind_type: bt,
            required_input: im,
        });
        (self, id)
    }

    fn finalize(mut self) -> ReduceResult<Sequence> {
        match self.binds.pop() {
            Some(last) => {
                Ok(Sequence {
                    binds: self.binds,
                    last: last.bind_type,
                })
            }
            None => Err(ReduceError::EmptyReduction),
        }
    }
}

fn reduce_node(expr: pel::Expression,
               input_modifier: InputModifier,
               sb: SequenceBuilder,
               context: &ReduceContext)
               -> ReduceResult<(SequenceBuilder, BindId)> {
    let (sb, bt) = match expr {
        pel::Expression::Literal(lit) => (sb, BindType::Direct(Command::Literal(lit))),
        pel::Expression::Variable(var) => (sb, BindType::Direct(Command::Variable(var))),
        pel::Expression::Global(id) => (sb, BindType::Direct(Command::Global(id))),
        pel::Expression::ConstantVariable(id, name) => {
            (sb, BindType::Direct(Command::ConstantVariable(id, name)))
        }
        pel::Expression::TernaryConditional(cond, lhs, rhs) => {
            let (sb, cond_id) = try!(reduce_node(*cond, InputModifier::In, sb, context));
            let lhs_seq = Box::new(try!(reduce(*lhs, context)));
            let rhs_seq = Box::new(try!(reduce(*rhs, context)));
            (sb, BindType::Select(cond_id, lhs_seq, rhs_seq))
        }
        pel::Expression::Swizzle(val, swizzle) => {
            let (sb, val_id) = try!(reduce_node(*val, input_modifier.clone(), sb, context));
            (sb, BindType::Direct(Command::Swizzle(val_id, swizzle)))
        }
        pel::Expression::ArraySubscript(arr, index) => {
            let (sb, arr_id) = try!(reduce_node(*arr, InputModifier::In, sb, context));
            let (sb, index_id) = try!(reduce_node(*index, InputModifier::In, sb, context));
            let bt = BindType::Direct(Command::ArraySubscript(arr_id, index_id));
            (sb, bt)
        }
        pel::Expression::Texture2DIndex(dty, tex, index) => {
            let (sb, tex_id) = try!(reduce_node(*tex, InputModifier::In, sb, context));
            let (sb, index_id) = try!(reduce_node(*index, InputModifier::In, sb, context));
            let bt = BindType::Direct(if input_modifier == InputModifier::In {
                Command::Intrinsic2(hir::Intrinsic2::Texture2DLoad(dty), tex_id, index_id)
            } else {
                Command::TextureIndex(dty, tex_id, index_id)
            });
            (sb, bt)
        }
        pel::Expression::RWTexture2DIndex(dty, tex, index) => {
            let (sb, tex_id) = try!(reduce_node(*tex, InputModifier::In, sb, context));
            let (sb, index_id) = try!(reduce_node(*index, InputModifier::In, sb, context));
            let bt = BindType::Direct(if input_modifier == InputModifier::In {
                Command::Intrinsic2(hir::Intrinsic2::RWTexture2DLoad(dty), tex_id, index_id)
            } else {
                Command::TextureIndex(dty, tex_id, index_id)
            });
            (sb, bt)
        }
        pel::Expression::Member(composite, member) => {
            let (sb, composite_id) = try!(reduce_node(*composite,
                                                      input_modifier.clone(),
                                                      sb,
                                                      context));
            (sb, BindType::Direct(Command::Member(composite_id, member)))
        }
        pel::Expression::Call(id, args) => {
            let overload = context.find_overload(&id);
            let fold_fn = |res: ReduceResult<(SequenceBuilder, Vec<BindId>)>, param| {
                let (sb, mut vec) = try!(res);
                let (arg, im) = param;
                let (sb, arg) = try!(reduce_node(arg, im, sb, context));
                vec.push(arg);
                Ok((sb, vec))
            };
            let initial = Vec::with_capacity(args.len());
            let im_iter = overload.2.iter().map(|p| p.1.clone());
            let param_iter = args.into_iter().zip(im_iter);
            let (sb, arg_ids) = try!(param_iter.fold(Ok((sb, initial)), fold_fn));
            (sb, BindType::Direct(Command::Call(id, arg_ids)))
        }
        pel::Expression::NumericConstructor(dtyl, cons) => {
            let fold_fn = |res: ReduceResult<(SequenceBuilder, Vec<ConstructorSlot>)>, arg| {
                let (sb, mut vec) = try!(res);
                let pel::ConstructorSlot { arity, expr } = arg;
                let (sb, arg) = try!(reduce_node(expr, InputModifier::In, sb, context));
                vec.push(ConstructorSlot {
                    arity: arity,
                    expr: arg,
                });
                Ok((sb, vec))
            };
            let initial = Vec::with_capacity(cons.len());
            let (sb, con_ids) = try!(cons.into_iter().fold(Ok((sb, initial)), fold_fn));
            (sb,
             BindType::Direct(Command::NumericConstructor(dtyl, con_ids)))
        }
        pel::Expression::Cast(ty, node) => {
            let (sb, node_id) = try!(reduce_node(*node, input_modifier.clone(), sb, context));
            (sb, BindType::Direct(Command::Cast(ty, node_id)))
        }
        pel::Expression::Intrinsic0(i) => (sb, BindType::Direct(Command::Intrinsic0(i))),
        pel::Expression::Intrinsic1(i, e1) => {
            let im1 = i.get_param1_input_modifier();
            let (sb, e1_id) = try!(reduce_node(*e1, im1, sb, context));
            (sb, BindType::Direct(Command::Intrinsic1(i, e1_id)))
        }
        pel::Expression::Intrinsic2(i, e1, e2) => {
            let im1 = i.get_param1_input_modifier();
            let (sb, e1_id) = try!(reduce_node(*e1, im1, sb, context));
            let im2 = i.get_param2_input_modifier();
            let (sb, e2_id) = try!(reduce_node(*e2, im2, sb, context));
            (sb, BindType::Direct(Command::Intrinsic2(i, e1_id, e2_id)))
        }
        pel::Expression::Intrinsic3(i, e1, e2, e3) => {
            let im1 = i.get_param1_input_modifier();
            let (sb, e1_id) = try!(reduce_node(*e1, im1, sb, context));
            let im2 = i.get_param2_input_modifier();
            let (sb, e2_id) = try!(reduce_node(*e2, im2, sb, context));
            let im3 = i.get_param3_input_modifier();
            let (sb, e3_id) = try!(reduce_node(*e3, im3, sb, context));
            let bt = BindType::Direct(Command::Intrinsic3(i, e1_id, e2_id, e3_id));
            (sb, bt)
        }
    };
    let (sb, id) = sb.push(bt, input_modifier);
    Ok((sb, id))
}

pub fn reduce(root_expr: pel::Expression, context: &ReduceContext) -> ReduceResult<Sequence> {
    let builder = SequenceBuilder::new();
    let builder = try!(reduce_node(root_expr, InputModifier::In, builder, context)).0;
    builder.finalize()
}

#[cfg(test)]
struct TestReduceContext;

#[cfg(test)]
impl ReduceContext for TestReduceContext {
    fn find_overload(&self, _: &hir::FunctionId) -> FunctionOverload {
        panic!("call in tests")
    }
}

#[test]
fn test_reduce_single_variable() {
    let var = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let pel = pel::Expression::Variable(var.clone());
    let rel = reduce(pel, &TestReduceContext);
    let expected_rel = Sequence {
        binds: vec![],
        last: BindType::Direct(Command::Variable(var)),
    };
    assert_eq!(rel, Ok(expected_rel));
}

#[test]
fn test_reduce_binary_operation() {
    let var_0_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_0 = pel::Expression::Variable(var_0_ref.clone());
    let var_1_ref = hir::VariableRef(hir::VariableId(1), hir::ScopeRef(0));
    let var_1 = pel::Expression::Variable(var_1_ref.clone());
    let dty = hir::DataType(hir::DataLayout::Scalar(hir::ScalarType::Float),
                            hir::TypeModifier::default());
    let add = hir::Intrinsic2::Add(dty.clone());
    let pel = pel::Expression::Intrinsic2(add.clone(), Box::new(var_0), Box::new(var_1));
    let rel = reduce(pel, &TestReduceContext);
    let expected_rel = Sequence {
        binds: vec![
            Bind::direct(0, Command::Variable(var_0_ref)),
            Bind::direct(1, Command::Variable(var_1_ref)),
        ],
        last: BindType::Direct(Command::Intrinsic2(add, BindId(0), BindId(1))),
    };
    assert_eq!(rel, Ok(expected_rel));
}
