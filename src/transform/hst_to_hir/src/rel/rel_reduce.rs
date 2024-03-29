//! Reduce transform to turn pel expressions into rel expressions

use super::rel::*;
use crate::pel;
use crate::typer::FunctionOverload;
use slp_lang_hir as hir;
#[cfg(test)]
use slp_lang_hir::ToExpressionType;
use slp_lang_hir::Type;
#[cfg(test)]
use std::collections::HashMap;
use std::fmt;

#[derive(PartialEq, Debug, Clone)]
pub enum ReduceError {
    EmptyReduction,
    FailedTypeParse,
}

pub type ReduceResult<T> = Result<T, ReduceError>;

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ReduceError::EmptyReduction => write!(f, "reduced expression is empty"),
            ReduceError::FailedTypeParse => write!(f, "failed to parse type reducing pel to rel"),
        }
    }
}

pub trait ReduceContext: hir::TypeContext {
    fn find_overload(&self, id: &hir::FunctionId) -> FunctionOverload;
}

struct SequenceBuilder {
    binds: Vec<Bind>,
}

impl SequenceBuilder {
    fn new() -> SequenceBuilder {
        SequenceBuilder { binds: vec![] }
    }

    fn push(mut self, bt: Command, im: InputModifier, ty: Type) -> (SequenceBuilder, BindId) {
        let id = BindId(self.binds.len() as u64);
        self.binds.push(Bind {
            id: id.clone(),
            value: bt,
            required_input: im,
            ty: ty,
        });
        (self, id)
    }

    fn finalize(self) -> ReduceResult<Sequence> {
        let id = match self.binds.last() {
            Some(last) => last.id.clone(),
            None => return Err(ReduceError::EmptyReduction),
        };
        Ok(Sequence {
            binds: self.binds,
            last: Some(id),
        })
    }
}

fn simplify_node(expr: pel::Expression) -> pel::Expression {
    // Rewrite swizzled assignments into a dedicated node.
    // This is because we treat assignment like any other function call
    // with the assign value being bound as an out param. We can't swizzle
    // out params into cl functions, so these get rewritten unoptimally, but
    // we can swizzle assign in cl, so this node represents this combined
    // operation and will generate more optimal cl code (it will be trivial
    // in the rel language instead of requiring complex combination)
    use pel::Expression::Intrinsic2 as I2;
    use pel::Expression::Swizzle;
    use slp_lang_hir::Intrinsic2::AssignSwizzle;
    use slp_lang_hir::Intrinsic2::Assignment as Assign;
    if let I2(Assign(ty), e1, e2) = expr {
        let unboxed_e1 = *e1;
        if let Swizzle(val, swizzle) = unboxed_e1 {
            I2(AssignSwizzle(ty, swizzle), val, e2)
        } else {
            I2(Assign(ty), Box::new(unboxed_e1), e2)
        }
    } else {
        expr
    }
}

fn reduce_node(
    expr: pel::Expression,
    input_modifier: InputModifier,
    sb: SequenceBuilder,
    context: &dyn ReduceContext,
) -> ReduceResult<(SequenceBuilder, BindId)> {
    let ty = match pel::TypeParser::get_expression_type(&expr, context.as_type_context()) {
        Ok(ety) => ety.0,
        Err(_) => return Err(ReduceError::FailedTypeParse),
    };
    let expr = simplify_node(expr);
    let (sb, bt) = match expr {
        pel::Expression::Literal(lit) => (sb, Command::Literal(lit)),
        pel::Expression::Variable(var) => (sb, Command::Variable(var)),
        pel::Expression::Global(id) => (sb, Command::Global(id)),
        pel::Expression::ConstantVariable(id, name) => (sb, Command::ConstantVariable(id, name)),
        pel::Expression::TernaryConditional(cond, lhs, rhs) => {
            let (sb, cond_id) = reduce_node(*cond, InputModifier::In, sb, context)?;
            let lhs_seq = Box::new(reduce(*lhs, context)?);
            let rhs_seq = Box::new(reduce(*rhs, context)?);
            (sb, Command::TernaryConditional(cond_id, lhs_seq, rhs_seq))
        }
        pel::Expression::Swizzle(val, swizzle) => {
            let (sb, val_id) = reduce_node(*val, input_modifier.clone(), sb, context)?;
            (sb, Command::Swizzle(val_id, swizzle))
        }
        pel::Expression::ArraySubscript(arr, index) => {
            let (sb, arr_id) = reduce_node(*arr, InputModifier::In, sb, context)?;
            let (sb, index_id) = reduce_node(*index, InputModifier::In, sb, context)?;
            let bt = Command::ArraySubscript(arr_id, index_id);
            (sb, bt)
        }
        pel::Expression::Texture2DIndex(dty, tex, index) => {
            let (sb, tex_id) = reduce_node(*tex, InputModifier::In, sb, context)?;
            let (sb, index_id) = reduce_node(*index, InputModifier::In, sb, context)?;
            let bt = if input_modifier == InputModifier::In {
                Command::Intrinsic2(hir::Intrinsic2::Texture2DLoad(dty), tex_id, index_id)
            } else {
                Command::Texture2DIndex(dty, tex_id, index_id)
            };
            (sb, bt)
        }
        pel::Expression::RWTexture2DIndex(dty, tex, index) => {
            let (sb, tex_id) = reduce_node(*tex, InputModifier::In, sb, context)?;
            let (sb, index_id) = reduce_node(*index, InputModifier::In, sb, context)?;
            let bt = if input_modifier == InputModifier::In {
                Command::Intrinsic2(hir::Intrinsic2::RWTexture2DLoad(dty), tex_id, index_id)
            } else {
                Command::RWTexture2DIndex(dty, tex_id, index_id)
            };
            (sb, bt)
        }
        pel::Expression::Member(composite, member) => {
            let (sb, composite_id) = reduce_node(*composite, input_modifier.clone(), sb, context)?;
            (sb, Command::Member(composite_id, member))
        }
        pel::Expression::Call(id, args) => {
            let overload = context.find_overload(&id);
            let fold_fn = |res: ReduceResult<(SequenceBuilder, Vec<BindId>)>, param| {
                let (sb, mut vec) = res?;
                let (arg, im) = param;
                let (sb, arg) = reduce_node(arg, im, sb, context)?;
                vec.push(arg);
                Ok((sb, vec))
            };
            let initial = Vec::with_capacity(args.len());
            let im_iter = overload.2.iter().map(|p| p.1.clone());
            let param_iter = args.into_iter().zip(im_iter);
            let (sb, arg_ids) = param_iter.fold(Ok((sb, initial)), fold_fn)?;
            (sb, Command::Call(id, arg_ids))
        }
        pel::Expression::NumericConstructor(dtyl, cons) => {
            let fold_fn = |res: ReduceResult<(SequenceBuilder, Vec<ConstructorSlot>)>, arg| {
                let (sb, mut vec) = res?;
                let pel::ConstructorSlot { arity, expr } = arg;
                let (sb, arg) = reduce_node(expr, InputModifier::In, sb, context)?;
                vec.push(ConstructorSlot {
                    arity: arity,
                    expr: arg,
                });
                Ok((sb, vec))
            };
            let initial = Vec::with_capacity(cons.len());
            let (sb, con_ids) = cons.into_iter().fold(Ok((sb, initial)), fold_fn)?;
            (sb, Command::NumericConstructor(dtyl, con_ids))
        }
        pel::Expression::Cast(ty, node) => {
            let (sb, node_id) = reduce_node(*node, input_modifier.clone(), sb, context)?;
            (sb, Command::Cast(ty, node_id))
        }
        pel::Expression::Intrinsic0(i) => (sb, Command::Intrinsic0(i)),
        pel::Expression::Intrinsic1(i, e1) => {
            let im1 = i.get_param1_input_modifier();
            let (sb, e1_id) = reduce_node(*e1, im1, sb, context)?;
            (sb, Command::Intrinsic1(i, e1_id))
        }
        pel::Expression::Intrinsic2(i, e1, e2) => {
            let im1 = i.get_param1_input_modifier();
            let (sb, e1_id) = reduce_node(*e1, im1, sb, context)?;
            let im2 = i.get_param2_input_modifier();
            let (sb, e2_id) = reduce_node(*e2, im2, sb, context)?;
            (sb, Command::Intrinsic2(i, e1_id, e2_id))
        }
        pel::Expression::Intrinsic3(i, e1, e2, e3) => {
            let im1 = i.get_param1_input_modifier();
            let (sb, e1_id) = reduce_node(*e1, im1, sb, context)?;
            let im2 = i.get_param2_input_modifier();
            let (sb, e2_id) = reduce_node(*e2, im2, sb, context)?;
            let im3 = i.get_param3_input_modifier();
            let (sb, e3_id) = reduce_node(*e3, im3, sb, context)?;
            let bt = Command::Intrinsic3(i, e1_id, e2_id, e3_id);
            (sb, bt)
        }
    };
    let (sb, id) = sb.push(bt, input_modifier, ty);
    Ok((sb, id))
}

pub fn reduce(root_expr: pel::Expression, context: &dyn ReduceContext) -> ReduceResult<Sequence> {
    let builder = SequenceBuilder::new();
    let builder = reduce_node(root_expr, InputModifier::In, builder, context)?.0;
    builder.finalize()
}

#[cfg(test)]
struct TestReduceContext {
    locals: HashMap<hir::VariableRef, Type>,
    globals: HashMap<hir::GlobalId, Type>,
}

#[cfg(test)]
impl ReduceContext for TestReduceContext {
    fn find_overload(&self, _: &hir::FunctionId) -> FunctionOverload {
        panic!("call in tests")
    }
}

#[cfg(test)]
impl hir::TypeContext for TestReduceContext {
    fn get_local(&self, var: &hir::VariableRef) -> Result<hir::ExpressionType, hir::TypeError> {
        match self.locals.get(var) {
            Some(ty) => Ok(ty.clone().to_lvalue()),
            None => panic!("TestReduceContext::get_local"),
        }
    }
    fn get_global(&self, id: &hir::GlobalId) -> Result<hir::ExpressionType, hir::TypeError> {
        match self.globals.get(id) {
            Some(ty) => Ok(ty.clone().to_lvalue()),
            None => panic!("TestReduceContext::get_global"),
        }
    }
    fn get_constant(
        &self,
        _: &hir::ConstantBufferId,
        _: &str,
    ) -> Result<hir::ExpressionType, hir::TypeError> {
        panic!("TestReduceContext::get_constant")
    }
    fn get_struct_member(
        &self,
        _: &hir::StructId,
        _: &str,
    ) -> Result<hir::ExpressionType, hir::TypeError> {
        panic!("TestReduceContext::get_struct_member")
    }
    fn get_function_return(
        &self,
        _: &hir::FunctionId,
    ) -> Result<hir::ExpressionType, hir::TypeError> {
        panic!("TestReduceContext::get_function_return")
    }
}

#[cfg(test)]
impl TestReduceContext {
    fn new() -> TestReduceContext {
        TestReduceContext {
            locals: HashMap::new(),
            globals: HashMap::new(),
        }
    }
    fn local(mut self, var: hir::VariableRef, ty: Type) -> TestReduceContext {
        self.locals.insert(var, ty);
        self
    }
    fn global(mut self, id: hir::GlobalId, ty: Type) -> TestReduceContext {
        self.globals.insert(id, ty);
        self
    }
}

#[test]
fn test_reduce_single_variable() {
    let var = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let pel = pel::Expression::Variable(var.clone());
    let c = TestReduceContext::new().local(var.clone(), Type::float());
    let rel = reduce(pel, &c);
    let expected_rel = Sequence {
        binds: vec![Bind::direct(0, Command::Variable(var), Type::float())],
        last: Some(BindId(0)),
    };
    assert_eq!(rel, Ok(expected_rel));
}

#[test]
fn test_reduce_binary_operation() {
    let var_0_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_0 = pel::Expression::Variable(var_0_ref.clone());
    let var_1_ref = hir::VariableRef(hir::VariableId(1), hir::ScopeRef(0));
    let var_1 = pel::Expression::Variable(var_1_ref.clone());
    let dty = hir::DataType(
        hir::DataLayout::Scalar(hir::ScalarType::Float),
        hir::TypeModifier::default(),
    );
    let ty = Type::from_data(dty.clone());
    let add = hir::Intrinsic2::Add(dty.clone());
    let pel = pel::Expression::Intrinsic2(add.clone(), Box::new(var_0), Box::new(var_1));
    let c = TestReduceContext::new()
        .local(var_0_ref.clone(), ty.clone())
        .local(var_1_ref.clone(), ty.clone());
    let rel = reduce(pel, &c);
    let expected_rel = Sequence {
        binds: vec![
            Bind::direct(0, Command::Variable(var_0_ref), ty.clone()),
            Bind::direct(1, Command::Variable(var_1_ref), ty.clone()),
            Bind::direct(2, Command::Intrinsic2(add, BindId(0), BindId(1)), ty),
        ],
        last: Some(BindId(2)),
    };
    assert_eq!(rel, Ok(expected_rel));
}

#[test]
fn test_reduce_texture_assignment() {
    let lit_zero = hir::Literal::Int(0);
    let lit_zero_pel = pel::Expression::Literal(lit_zero.clone());
    let dtyl_index = hir::DataLayout::Vector(hir::ScalarType::Int, 2);
    let cons = vec![
        pel::ConstructorSlot {
            arity: 1,
            expr: lit_zero_pel.clone(),
        },
        pel::ConstructorSlot {
            arity: 1,
            expr: lit_zero_pel,
        },
    ];
    let lit_zero2 = Box::new(pel::Expression::NumericConstructor(
        dtyl_index.clone(),
        cons,
    ));

    let dtyl = hir::DataLayout::Vector(hir::ScalarType::Float, 4);
    let dty = hir::DataType(dtyl, hir::TypeModifier::default());
    let ty = Type::from_data(dty.clone());
    let tex_0 = hir::GlobalId(0);
    let tex_0_pel = Box::new(pel::Expression::Global(tex_0.clone()));
    let ti_0 = pel::Expression::RWTexture2DIndex(dty.clone(), tex_0_pel, lit_zero2.clone());
    let tex_1 = hir::GlobalId(1);
    let tex_1_pel = Box::new(pel::Expression::Global(tex_1.clone()));
    let ti_1 = pel::Expression::Texture2DIndex(dty.clone(), tex_1_pel, lit_zero2.clone());

    let assign = hir::Intrinsic2::Assignment(Type::from_data(dty.clone()));
    let pel = pel::Expression::Intrinsic2(assign.clone(), Box::new(ti_0), Box::new(ti_1));

    let tex_ty = Type::from_layout(hir::TypeLayout::Object(hir::ObjectType::RWTexture2D(
        dty.clone(),
    )));
    let c = TestReduceContext::new()
        .global(tex_0, tex_ty.clone())
        .global(tex_1, tex_ty.clone());
    let rel = reduce(pel, &c);
    let expected_rel = Sequence {
        binds: vec![
            Bind::direct(0, Command::Global(tex_0), tex_ty.clone()),
            Bind::direct(1, Command::Literal(lit_zero.clone()), Type::int()),
            Bind::direct(2, Command::Literal(lit_zero.clone()), Type::int()),
            Bind::direct(
                3,
                Command::NumericConstructor(
                    dtyl_index.clone(),
                    vec![
                        ConstructorSlot {
                            arity: 1,
                            expr: BindId(1),
                        },
                        ConstructorSlot {
                            arity: 1,
                            expr: BindId(2),
                        },
                    ],
                ),
                Type::intn(2),
            ),
            Bind {
                id: BindId(4),
                value: Command::RWTexture2DIndex(dty.clone(), BindId(0), BindId(3)),
                required_input: InputModifier::Out,
                ty: ty.clone(),
            },
            Bind::direct(5, Command::Global(tex_1), tex_ty),
            Bind::direct(6, Command::Literal(lit_zero.clone()), Type::int()),
            Bind::direct(7, Command::Literal(lit_zero.clone()), Type::int()),
            Bind::direct(
                8,
                Command::NumericConstructor(
                    dtyl_index,
                    vec![
                        ConstructorSlot {
                            arity: 1,
                            expr: BindId(6),
                        },
                        ConstructorSlot {
                            arity: 1,
                            expr: BindId(7),
                        },
                    ],
                ),
                Type::intn(2),
            ),
            Bind::direct(
                9,
                Command::Intrinsic2(hir::Intrinsic2::Texture2DLoad(dty), BindId(5), BindId(8)),
                ty.clone(),
            ),
            Bind::direct(10, Command::Intrinsic2(assign, BindId(4), BindId(9)), ty),
        ],
        last: Some(BindId(10)),
    };
    assert_eq!(rel, Ok(expected_rel));
}
