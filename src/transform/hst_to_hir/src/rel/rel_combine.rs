
//! Combine transform to turn rel expressions into hir expressions

use super::rel::*;
use slp_lang_hir as hir;
use std::error;
use std::fmt;
use std::collections::HashMap;

#[derive(PartialEq, Debug, Clone)]
pub enum CombineError {
    FailedToResolveMultiPartExpression,
    FailedToAllocateTemporary,
    FailedToRegisterLocal,
    LastBindDoesNotExist,
}

pub type CombineResult<T> = Result<T, CombineError>;

impl error::Error for CombineError {
    fn description(&self) -> &str {
        match *self {
            CombineError::FailedToResolveMultiPartExpression => {
                "expression too complex to resolve (rel parser internal error)"
            }
            CombineError::FailedToAllocateTemporary => "failed to allocate temporary",
            CombineError::FailedToRegisterLocal => "failed to register local",
            CombineError::LastBindDoesNotExist => "failed to find last part of rel sequence",
        }
    }
}

impl fmt::Display for CombineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

/// Handler for allocating variables inside combined expressions
pub trait CombineContext {
    /// Create a new local variable id in the scope the expression will live in
    fn allocate_local(&mut self,
                      debug_name: String,
                      ty: hir::Type)
                      -> CombineResult<hir::VariableId>;

    /// Register that the combined expression will use the given variable in
    /// the scope of the original expression, and return the ref used in the
    /// scope of the combined expression
    ///
    /// For example, the combined expression will likely be either put into a
    /// scope block, in which the refs will have the scope ref increased by one,
    /// or in a helper function, in which the returned ref must be the ref to
    /// the function argument
    ///
    /// This will only be called once total for each ref
    fn register_local(&mut self, var: hir::VariableRef) -> CombineResult<hir::VariableRef>;
}

pub struct FakeCombineContext;

impl CombineContext for FakeCombineContext {
    fn allocate_local(&mut self, _: String, _: hir::Type) -> CombineResult<hir::VariableId> {
        Err(CombineError::FailedToAllocateTemporary)
    }
    fn register_local(&mut self, _: hir::VariableRef) -> CombineResult<hir::VariableRef> {
        Err(CombineError::FailedToRegisterLocal)
    }
}

pub struct ScopeCombineContext {
    new_locals: HashMap<hir::VariableId, (String, hir::Type)>,
    next_id: u32,
}

impl ScopeCombineContext {
    pub fn new() -> ScopeCombineContext {
        ScopeCombineContext {
            new_locals: HashMap::new(),
            next_id: 0,
        }
    }
    pub fn finalize(self, block: CombinedBlock) -> hir::ScopeBlock {
        let mut statements = block.statements;
        match block.last_expression {
            Some(expr) => statements.push(hir::Statement::Expression(expr)),
            None => {}
        }
        let decls = hir::ScopedDeclarations { variables: self.new_locals };
        hir::ScopeBlock(statements, decls)
    }
}

impl CombineContext for ScopeCombineContext {
    fn allocate_local(&mut self, name: String, ty: hir::Type) -> CombineResult<hir::VariableId> {
        let id = hir::VariableId(self.next_id);
        self.next_id = self.next_id + 1;
        match self.new_locals.insert(id, (name, ty)) {
            Some(_) => panic!("duplicate id in ScopeCombineContext"),
            None => Ok(id),
        }
    }
    fn register_local(&mut self, var_ref: hir::VariableRef) -> CombineResult<hir::VariableRef> {
        let id = var_ref.0;
        let scope = var_ref.1;
        Ok(hir::VariableRef(id, hir::ScopeRef(scope.0 + 1)))
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct CombinedBlock {
    statements: Vec<hir::Statement>,
    last_expression: Option<hir::Expression>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum CombinedExpression {
    Single(hir::Expression),
    Multi(CombinedBlock),
}

pub fn hir_relocate(expr: hir::Expression,
                    map: &HashMap<hir::VariableRef, hir::VariableRef>)
                    -> hir::Expression {
    use slp_lang_hir::Expression;
    let relocate = |expr: Box<hir::Expression>| Box::new(hir_relocate(*expr, map));
    let relocater = |expr: hir::Expression| hir_relocate(expr, map);
    match expr {
        Expression::Literal(lit) => Expression::Literal(lit),
        Expression::Variable(var_ref) => {
            match map.get(&var_ref) {
                Some(var) => Expression::Variable(var.clone()),
                None => panic!("invalid remap table"),
            }
        }
        Expression::Global(id) => Expression::Global(id),
        Expression::ConstantVariable(id, name) => Expression::ConstantVariable(id, name),
        Expression::TernaryConditional(cond, left, right) => {
            let cond = relocate(cond);
            let left = relocate(left);
            let right = relocate(right);
            Expression::TernaryConditional(cond, left, right)
        }
        Expression::Swizzle(vec, swizzle) => {
            let vec = relocate(vec);
            Expression::Swizzle(vec, swizzle)
        }
        Expression::ArraySubscript(arr, index) => {
            let arr = relocate(arr);
            let index = relocate(index);
            Expression::ArraySubscript(arr, index)
        }
        Expression::Member(expr, name) => {
            let expr = relocate(expr);
            Expression::Member(expr, name)
        }
        Expression::Call(id, exprs) => {
            let args = exprs.into_iter().map(|e| relocater(e)).collect();
            Expression::Call(id, args)
        }
        Expression::NumericConstructor(dtyl, elements) => {
            let fun = |mut e: hir::ConstructorSlot| {
                e.expr = relocater(e.expr);
                e
            };
            let cons = elements.into_iter()
                               .map(fun)
                               .collect();
            Expression::NumericConstructor(dtyl, cons)
        }
        Expression::Cast(ty, expr) => {
            let expr = relocate(expr);
            Expression::Cast(ty, expr)
        }
        Expression::Intrinsic0(i) => Expression::Intrinsic0(i),
        Expression::Intrinsic1(i, e1) => {
            let e1 = relocate(e1);
            Expression::Intrinsic1(i, e1)
        }
        Expression::Intrinsic2(i, e1, e2) => {
            let e1 = relocate(e1);
            let e2 = relocate(e2);
            Expression::Intrinsic2(i, e1, e2)
        }
        Expression::Intrinsic3(i, e1, e2, e3) => {
            let e1 = relocate(e1);
            let e2 = relocate(e2);
            let e3 = relocate(e3);
            Expression::Intrinsic3(i, e1, e2, e3)
        }
    }
}

fn combine_group(seq: Sequence) -> Sequence {
    let last = seq.last;
    let binds = seq.binds;

    let mut grouped_binds = Vec::with_capacity(binds.len());

    fn consume_last_bind(id: BindId, binds: &mut Vec<Bind>) -> Option<Bind> {
        assert!(binds.len() >= 1, "bad sequence");
        let matches = {
            let peeked = binds.last().unwrap();
            if peeked.id == id {
                match peeked.value {
                    Command::Trivial(_, _) => true,
                    Command::Variable(_) => true,
                    Command::Global(_) => true,
                    _ => false,
                }
            } else {
                false
            }
        };
        if matches {
            Some(binds.pop().unwrap())
        } else {
            None
        }
    }

    fn trivialize_bind(bind: Bind) -> hir::Expression {
        match bind.value {
            Command::Trivial(expr, _) => expr,
            Command::Variable(var) => hir::Expression::Variable(var),
            Command::Global(id) => hir::Expression::Global(id),
            _ => panic!("invalid node to trivialize"),
        }
    }

    fn consume_lastn(ids: &[BindId], binds: &mut Vec<Bind>) -> Option<Vec<hir::Expression>> {
        assert!(binds.len() >= ids.len());
        let mut removed_binds = Vec::with_capacity(ids.len());
        for id in ids.iter().rev() {
            let bind_opt = consume_last_bind(id.clone(), binds);
            match bind_opt {
                Some(bind) => removed_binds.push(bind),
                None => break,
            }
        }
        if removed_binds.len() != ids.len() {
            for bind in removed_binds.into_iter().rev() {
                binds.push(bind);
            }
            None
        } else {
            let exprs = removed_binds.into_iter().rev().map(|bind| trivialize_bind(bind)).collect();
            Some(exprs)
        }
    }

    fn consume_last1(id: BindId, binds: &mut Vec<Bind>) -> Option<hir::Expression> {
        match consume_lastn(&[id], binds) {
            Some(mut vec) => {
                assert!(vec.len() == 1);
                Some(vec.pop().unwrap())
            }
            None => None,
        }
    }

    fn consume_last2(id1: BindId,
                     id2: BindId,
                     binds: &mut Vec<Bind>)
                     -> Option<(hir::Expression, hir::Expression)> {
        match consume_lastn(&[id1, id2], binds) {
            Some(mut vec) => {
                assert!(vec.len() == 2);
                let e2 = vec.pop().unwrap();
                let e1 = vec.pop().unwrap();
                Some((e1, e2))
            }
            None => None,
        }
    }

    fn consume_last3(id1: BindId,
                     id2: BindId,
                     id3: BindId,
                     binds: &mut Vec<Bind>)
                     -> Option<(hir::Expression, hir::Expression, hir::Expression)> {
        match consume_lastn(&[id1, id2, id3], binds) {
            Some(mut vec) => {
                assert!(vec.len() == 3);
                let e3 = vec.pop().unwrap();
                let e2 = vec.pop().unwrap();
                let e1 = vec.pop().unwrap();
                Some((e1, e2, e3))
            }
            None => None,
        }
    }

    fn process_command(bind: Bind, ty: &hir::Type, grouped_binds: &mut Vec<Bind>) -> Command {
        match bind.value {
            Command::Literal(lit) => {
                assert!(bind.required_input == InputModifier::In);
                let expr = hir::Expression::Literal(lit);
                Command::Trivial(expr, MutationParam::Const)
            }
            Command::Variable(var) => {
                if bind.required_input == InputModifier::In {
                    let expr = hir::Expression::Variable(var);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Variable(var)
                }
            }
            Command::Global(id) => {
                if bind.required_input == InputModifier::In {
                    let expr = hir::Expression::Global(id);
                    let mp = match ty.0 {
                        hir::TypeLayout::Object(_) => MutationParam::Const,
                        _ => MutationParam::Mutable,
                    };
                    Command::Trivial(expr, mp)
                } else {
                    Command::Global(id)
                }
            }
            Command::ConstantVariable(id, name) => {
                assert!(bind.required_input == InputModifier::In);
                let expr = hir::Expression::ConstantVariable(id, name);
                Command::Trivial(expr, MutationParam::Const)
            }
            Command::TernaryConditional(cond, lhs, rhs) => {
                assert!(bind.required_input == InputModifier::In);
                let cond_val = match consume_last1(cond, grouped_binds) {
                    Some(cond_val) => cond_val,
                    None => return Command::TernaryConditional(cond, lhs, rhs),
                };
                let lhs = combine_group(*lhs);
                let rhs = combine_group(*rhs);
                let lhs_val = match combine_trivial(lhs.clone()) {
                    TrivialResult::Trivial(expr) => expr,
                    _ => {
                        let lhs = Box::new(lhs);
                        let rhs = Box::new(rhs);
                        return Command::TernaryConditional(cond, lhs, rhs);
                    }
                };
                let rhs_val = match combine_trivial(rhs.clone()) {
                    TrivialResult::Trivial(expr) => expr,
                    _ => {
                        let lhs = Box::new(lhs);
                        let rhs = Box::new(rhs);
                        return Command::TernaryConditional(cond, lhs, rhs);
                    }
                };
                let cond_box = Box::new(cond_val);
                let lhs_val = Box::new(lhs_val);
                let rhs_val = Box::new(rhs_val);
                let expr = hir::Expression::TernaryConditional(cond_box, lhs_val, rhs_val);
                Command::Trivial(expr, MutationParam::Mutable)
            }
            Command::Swizzle(val, swizzle) => {
                if bind.required_input == InputModifier::In {
                    let val = match consume_last1(val, grouped_binds) {
                        Some(val) => val,
                        None => return Command::Swizzle(val, swizzle),
                    };
                    let val_box = Box::new(val);
                    let expr = hir::Expression::Swizzle(val_box, swizzle);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Swizzle(val, swizzle)
                }
            }
            Command::ArraySubscript(arr, ind) => {
                if bind.required_input == InputModifier::In {
                    let (arr_val, ind_val) = match consume_last2(arr, ind, grouped_binds) {
                        Some((arr_val, ind_val)) => (arr_val, ind_val),
                        None => return Command::ArraySubscript(arr, ind),
                    };
                    let arr_box = Box::new(arr_val);
                    let ind_box = Box::new(ind_val);
                    let expr = hir::Expression::ArraySubscript(arr_box, ind_box);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::ArraySubscript(arr, ind)
                }
            }
            Command::Texture2DIndex(dty, tex, index) => Command::Texture2DIndex(dty, tex, index),
            Command::RWTexture2DIndex(dty, tex, index) => {
                Command::RWTexture2DIndex(dty, tex, index)
            }
            Command::Member(val, name) => {
                if bind.required_input == InputModifier::In {
                    let val = match consume_last1(val, grouped_binds) {
                        Some(val) => val,
                        None => return Command::Member(val, name),
                    };
                    let val_box = Box::new(val);
                    let expr = hir::Expression::Member(val_box, name);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Member(val, name)
                }
            }
            Command::Call(id, params) => {
                assert_eq!(bind.required_input, InputModifier::In);
                let vals = match consume_lastn(&params[..], grouped_binds) {
                    Some(vals) => vals,
                    None => return Command::Call(id, params),
                };
                let expr = hir::Expression::Call(id, vals);
                Command::Trivial(expr, MutationParam::Mutable)
            }
            Command::NumericConstructor(dtyl, cons_slots) => {
                assert_eq!(bind.required_input, InputModifier::In);
                let mut params = Vec::with_capacity(cons_slots.len());
                let mut arity = Vec::with_capacity(cons_slots.len());
                for con_slot in &cons_slots {
                    arity.push(con_slot.arity.clone());
                    params.push(con_slot.expr.clone());
                }
                let vals = match consume_lastn(&params[..], grouped_binds) {
                    Some(vals) => vals,
                    None => return Command::NumericConstructor(dtyl, cons_slots),
                };
                assert_eq!(vals.len(), arity.len());
                let cons = arity.into_iter()
                                .zip(vals)
                                .map(|(a, e)| {
                                    hir::ConstructorSlot {
                                        arity: a,
                                        expr: e,
                                    }
                                })
                                .collect();
                let expr = hir::Expression::NumericConstructor(dtyl, cons);
                Command::Trivial(expr, MutationParam::Mutable)
            }
            Command::Cast(ty, val) => {
                if bind.required_input == InputModifier::In {
                    let val = match consume_last1(val, grouped_binds) {
                        Some(val) => val,
                        None => return Command::Cast(ty, val),
                    };
                    let val_box = Box::new(val);
                    let expr = hir::Expression::Cast(ty, val_box);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Cast(ty, val)
                }
            }
            Command::Intrinsic0(i) => {
                if bind.required_input == InputModifier::In {
                    let expr = hir::Expression::Intrinsic0(i);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Intrinsic0(i)
                }
            }
            Command::Intrinsic1(i, p1) => {
                if bind.required_input == InputModifier::In {
                    let e1 = match consume_last1(p1, grouped_binds) {
                        Some(e1) => e1,
                        None => return Command::Intrinsic1(i, p1),
                    };
                    let e1b = Box::new(e1);
                    let expr = hir::Expression::Intrinsic1(i, e1b);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Intrinsic1(i, p1)
                }
            }
            Command::Intrinsic2(i, p1, p2) => {
                if bind.required_input == InputModifier::In {
                    let (e1, e2) = match consume_last2(p1, p2, grouped_binds) {
                        Some(es) => es,
                        None => return Command::Intrinsic2(i, p1, p2),
                    };
                    let e1b = Box::new(e1);
                    let e2b = Box::new(e2);
                    let expr = hir::Expression::Intrinsic2(i, e1b, e2b);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Intrinsic2(i, p1, p2)
                }
            }
            Command::Intrinsic3(i, p1, p2, p3) => {
                if bind.required_input == InputModifier::In {
                    let (e1, e2, e3) = match consume_last3(p1, p2, p3, grouped_binds) {
                        Some(es) => es,
                        None => return Command::Intrinsic3(i, p1, p2, p3),
                    };
                    let e1b = Box::new(e1);
                    let e2b = Box::new(e2);
                    let e3b = Box::new(e3);
                    let expr = hir::Expression::Intrinsic3(i, e1b, e2b, e3b);
                    Command::Trivial(expr, MutationParam::Mutable)
                } else {
                    Command::Intrinsic3(i, p1, p2, p3)
                }
            }
            Command::Trivial(expr, mp) => Command::Trivial(expr, mp),
        }
    }

    for bind in binds.into_iter() {
        let id = bind.id.clone();
        let required_input = bind.required_input.clone();
        let ty = bind.ty.clone();
        let command = process_command(bind, &ty, &mut grouped_binds);
        grouped_binds.push(Bind {
            id: id,
            value: command,
            required_input: required_input,
            ty: ty,
        });
    }

    Sequence {
        binds: grouped_binds,
        last: last,
    }
}

fn combine_complex(seq: Sequence, context: &mut CombineContext) -> CombineResult<CombinedBlock> {
    // Build map of all required local variables
    let used = seq.find_used_locals();
    let mut required_locals = HashMap::with_capacity(used.len());
    for var in used {
        required_locals.insert(var, try!(context.register_local(var)));
    }

    /// Processed state for bind slot
    struct ProcessedBind {
        /// Expression to use the result of the bind
        reference: Option<hir::Expression>,
        /// Expression to write output parameter
        write_back: WriteBack,
        /// Trigger write back for previous bind
        recursive_write_back: Option<BindId>,
        /// Copy of the type of the bind
        ty: hir::Type,
    }

    #[derive(PartialEq, Debug, Clone)]
    enum WriteBack {
        Invoke(hir::Expression),
        Nothing,
        Fail,
    }

    impl ProcessedBind {
        fn enqueue_write_back_ensure(&self,
                                     sts: &mut Vec<hir::Statement>,
                                     processed: &HashMap<BindId, ProcessedBind>) {
            match self.write_back {
                WriteBack::Invoke(ref wb) => sts.push(hir::Statement::Expression(wb.clone())),
                WriteBack::Nothing => {}
                WriteBack::Fail => panic!("out param used for expression without write back"),
            }
            match self.recursive_write_back {
                Some(ref id) => {
                    match processed.get(id) {
                        Some(ref p) => p.enqueue_write_back(sts, processed),
                        None => panic!("recursive_write_back depends on non-existant id"),
                    }
                }
                None => {}
            }
        }

        fn enqueue_write_back(&self,
                              sts: &mut Vec<hir::Statement>,
                              processed: &HashMap<BindId, ProcessedBind>) {
            if self.write_back != WriteBack::Fail {
                self.enqueue_write_back_ensure(sts, processed)
            }
        }
    }

    fn allocate_local(name: &str,
                      ty: hir::Type,
                      context: &mut CombineContext)
                      -> CombineResult<hir::VariableId> {
        // Ensure type is allocatable as a variable
        if ty.is_array() || ty.is_void() {
            return Err(CombineError::FailedToResolveMultiPartExpression);
        }
        context.allocate_local(name.to_string(), ty)
    }

    fn build_command(command: Command,
                     im: InputModifier,
                     ty: hir::Type,
                     locals: &HashMap<hir::VariableRef, hir::VariableRef>,
                     processed: &HashMap<BindId, ProcessedBind>,
                     context: &mut CombineContext)
                     -> CombineResult<(Vec<hir::Statement>, ProcessedBind)> {
        match command {
            Command::Literal(_) => panic!("combine_complex: literal encountered"),
            Command::Variable(var_prev_scope) => {
                // In should be turned into Trivial node before here
                assert!(im != InputModifier::In);
                // Remap variable into combined scope
                let var = {
                    // Ref to input variable
                    let var_ref = match locals.get(&var_prev_scope) {
                        Some(var_ref) => var_ref,
                        None => panic!("command uses variable that's not in required locals"),
                    };
                    hir::Expression::Variable(var_ref.clone())
                };
                // If we're an out node then order of ops should only matter
                // at out/inout-bind point, so we can just set the reference
                // param and have it substitued in
                let p = ProcessedBind {
                    reference: Some(var),
                    write_back: WriteBack::Nothing,
                    recursive_write_back: None,
                    ty: ty,
                };
                Ok((vec![], p))
            }
            Command::Global(id) => {
                let p = ProcessedBind {
                    reference: Some(hir::Expression::Global(id.clone())),
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                Ok((vec![], p))
            }
            Command::Swizzle(val, swizzle) => {
                // Create temporary local
                let id = try!(allocate_local("swizzle", ty.clone(), context));
                let val_p = match processed.get(&val) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Swizzle)"),
                };
                let val_expr = val_p.reference.clone().expect("void input (swizzle)");
                // Push the temporary local into a hir node
                let tmp_var = {
                    let tmp_var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                    hir::Expression::Variable(tmp_var_ref)
                };
                let val_ref = hir::Expression::Swizzle(Box::new(val_expr.clone()), swizzle);
                let statement = {
                    let init = if im != InputModifier::Out {
                        Some(hir::Initializer::Expression(val_ref.clone()))
                    } else {
                        None
                    };
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                        init: init,
                    };
                    hir::Statement::Var(vd)
                };
                let write_back = if im != InputModifier::In {
                    let assign = hir::Intrinsic2::Assignment(val_p.ty.clone());
                    let write_to = Box::new(val_ref);
                    let write_from = Box::new(tmp_var.clone());
                    WriteBack::Invoke(hir::Expression::Intrinsic2(assign, write_to, write_from))
                } else {
                    WriteBack::Fail
                };
                // Finish
                let p = ProcessedBind {
                    reference: Some(tmp_var),
                    write_back: write_back,
                    recursive_write_back: Some(val),
                    ty: ty,
                };
                Ok((vec![statement], p))
            }
            Command::RWTexture2DIndex(dty, tex, index) => {
                let tex_ref = match processed.get(&tex) {
                    Some(p) => p.reference.clone().expect("void input (rwtex tex)"),
                    None => panic!("reference local bind does not exist (RWTexture2DIndex: tex)"),
                };
                let index_ref = match processed.get(&index) {
                    Some(p) => p.reference.clone().expect("void input (rwtex index)"),
                    None => panic!("reference local bind does not exist (RWTexture2DIndex: index)"),
                };
                let load = {
                    let li = hir::Intrinsic2::RWTexture2DLoad(dty.clone());
                    let texture = Box::new(tex_ref.clone());
                    let index = Box::new(index_ref.clone());
                    hir::Expression::Intrinsic2(li, texture, index)
                };
                let (statements, reference, write_back) = if im == InputModifier::In {
                    (vec![], load, WriteBack::Fail)
                } else {
                    let init = if im == InputModifier::InOut {
                        Some(hir::Initializer::Expression(load))
                    } else {
                        None
                    };
                    let id = try!(allocate_local("tex", ty.clone(), context));
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                        init: init,
                    };
                    let var_ref = hir::Expression::Variable(hir::VariableRef(id, hir::ScopeRef(0)));
                    let store = {
                        let si = hir::Intrinsic3::RWTexture2DStore(dty);
                        let texture = Box::new(tex_ref);
                        let index = Box::new(index_ref);
                        let value = Box::new(var_ref.clone());
                        hir::Expression::Intrinsic3(si, texture, index, value)
                    };
                    (vec![hir::Statement::Var(vd)],
                     var_ref,
                     WriteBack::Invoke(store))
                };
                let p = ProcessedBind {
                    reference: Some(reference),
                    write_back: write_back,
                    recursive_write_back: None,
                    ty: ty,
                };
                Ok((statements, p))
            }
            Command::Call(func_if, params) => {
                assert_eq!(im, InputModifier::In);
                let mut hir_params = vec![];
                let mut write_backs = vec![];
                for param in &params {
                    let p = match processed.get(param) {
                        Some(p) => p,
                        None => panic!("reference local bind does not exist"),
                    };
                    let expr = p.reference.clone().expect("void input (call)");
                    hir_params.push(expr);
                    p.enqueue_write_back(&mut write_backs, processed);
                }
                let init = hir::Expression::Call(func_if, hir_params);
                let (mut statements, reference) = if ty.is_void() {
                    let statements = vec![hir::Statement::Expression(init)];
                    (statements, None)
                } else {
                    let id = try!(allocate_local("call", ty.clone(), context));
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                        init: Some(hir::Initializer::Expression(init)),
                    };
                    let statements = vec![hir::Statement::Var(vd)];
                    let var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                    (statements, Some(hir::Expression::Variable(var_ref)))
                };
                let p = ProcessedBind {
                    reference: reference,
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                statements.append(&mut write_backs);
                Ok((statements, p))
            }
            Command::NumericConstructor(dtyl, cons) => {
                let id = try!(allocate_local("cons", ty.clone(), context));
                let mut hir_cons = vec![];
                for con in cons {
                    let expr = match processed.get(&con.expr) {
                        Some(p) => p.reference.clone().expect("void input (cons)"),
                        None => panic!("reference local bind does not exist"),
                    };
                    let hir_con = hir::ConstructorSlot {
                        arity: con.arity,
                        expr: expr,
                    };
                    hir_cons.push(hir_con);
                }
                let init = hir::Expression::NumericConstructor(dtyl, hir_cons);
                let vd = hir::VarDef {
                    id: id.clone(),
                    local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                    init: Some(hir::Initializer::Expression(init)),
                };
                let statement = hir::Statement::Var(vd);
                let var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                let p = ProcessedBind {
                    reference: Some(hir::Expression::Variable(var_ref)),
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                Ok((vec![statement], p))
            }
            Command::Cast(ty, val) => {
                // Create temporary local
                let val_p = match processed.get(&val) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Cast)"),
                };
                if ty.is_void() {
                    assert_eq!(im, InputModifier::In);
                    let statements = match val_p.reference.clone() {
                        Some(reference) => {
                            let ref_box = Box::new(reference);
                            let casted_expr = hir::Expression::Cast(ty.clone(), ref_box);
                            vec![hir::Statement::Expression(casted_expr)]
                        }
                        None => vec![],
                    };
                    let p = ProcessedBind {
                        reference: None,
                        write_back: WriteBack::Fail,
                        recursive_write_back: None,
                        ty: ty,
                    };
                    Ok((statements, p))
                } else {
                    let val_expr = val_p.reference.clone().expect("void input (cast)");
                    let casted_expr = hir::Expression::Cast(ty.clone(), Box::new(val_expr.clone()));
                    let id = try!(allocate_local("cast", ty.clone(), context));
                    // Push the temporary local into a hir node
                    let tmp_var = {
                        let tmp_var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                        hir::Expression::Variable(tmp_var_ref)
                    };
                    let statement = {
                        let init = if im != InputModifier::Out {
                            Some(hir::Initializer::Expression(casted_expr))
                        } else {
                            None
                        };
                        let vd = hir::VarDef {
                            id: id.clone(),
                            local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                            init: init,
                        };
                        hir::Statement::Var(vd)
                    };
                    let write_back = if im != InputModifier::In {
                        let assign = hir::Intrinsic2::Assignment(val_p.ty.clone());
                        let write_to = Box::new(val_expr);
                        let write_from = {
                            let from_var = Box::new(tmp_var.clone());
                            let cast = hir::Expression::Cast(val_p.ty.clone(), from_var);
                            Box::new(cast)
                        };
                        WriteBack::Invoke(hir::Expression::Intrinsic2(assign, write_to, write_from))
                    } else {
                        WriteBack::Fail
                    };
                    let p = ProcessedBind {
                        reference: Some(tmp_var),
                        write_back: write_back,
                        recursive_write_back: Some(val),
                        ty: ty,
                    };
                    Ok((vec![statement], p))
                }
            }
            Command::Intrinsic2(i, b1, b2) => {
                // Assign intrinsics can technically bind to out slots
                // This is not supported in various ways
                assert_eq!(im, InputModifier::In);
                let p1 = match processed.get(&b1) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic2: b1)"),
                };
                let e1 = p1.reference.clone().expect("void input");
                let im1 = i.get_param1_input_modifier();
                let p2 = match processed.get(&b2) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic2: b2)"),
                };
                let e2 = p2.reference.clone().expect("void input");
                let im2 = i.get_param2_input_modifier();
                let init = hir::Expression::Intrinsic2(i, Box::new(e1), Box::new(e2));
                let (mut statements, reference) = if ty.is_void() {
                    let statements = vec![hir::Statement::Expression(init)];
                    (statements, None)
                } else {
                    let id = try!(allocate_local("i2", ty.clone(), context));
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                        init: Some(hir::Initializer::Expression(init)),
                    };
                    let statements = vec![hir::Statement::Var(vd)];
                    let var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                    (statements, Some(hir::Expression::Variable(var_ref)))
                };
                let p = ProcessedBind {
                    reference: reference,
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                if im1 != InputModifier::In {
                    p1.enqueue_write_back_ensure(&mut statements, processed);
                }
                if im2 != InputModifier::In {
                    p2.enqueue_write_back_ensure(&mut statements, processed);
                }
                Ok((statements, p))
            }
            Command::Intrinsic3(i, b1, b2, b3) => {
                assert_eq!(im, InputModifier::In);
                let p1 = match processed.get(&b1) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic3: b1)"),
                };
                let e1 = p1.reference.clone().expect("void input");
                let im1 = i.get_param1_input_modifier();
                let p2 = match processed.get(&b2) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic3: b2)"),
                };
                let e2 = p2.reference.clone().expect("void input");
                let im2 = i.get_param2_input_modifier();
                let p3 = match processed.get(&b3) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic3: b3)"),
                };
                let e3 = p3.reference.clone().expect("void input");
                let im3 = i.get_param3_input_modifier();
                let init = hir::Expression::Intrinsic3(i, Box::new(e1), Box::new(e2), Box::new(e3));
                let (mut statements, reference) = if ty.is_void() {
                    let statements = vec![hir::Statement::Expression(init)];
                    (statements, None)
                } else {
                    let id = try!(allocate_local("i3", ty.clone(), context));
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                        init: Some(hir::Initializer::Expression(init)),
                    };
                    let statements = vec![hir::Statement::Var(vd)];
                    let var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                    (statements, Some(hir::Expression::Variable(var_ref)))
                };
                let p = ProcessedBind {
                    reference: reference,
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                if im1 != InputModifier::In {
                    p1.enqueue_write_back_ensure(&mut statements, processed);
                }
                if im2 != InputModifier::In {
                    p2.enqueue_write_back_ensure(&mut statements, processed);
                }
                if im3 != InputModifier::In {
                    p3.enqueue_write_back_ensure(&mut statements, processed);
                }
                Ok((statements, p))
            }
            Command::Trivial(expr, MutationParam::Mutable) => {
                let expr = hir_relocate(expr, locals);
                let id = try!(allocate_local("expr", ty.clone(), context));;
                let vd = hir::VarDef {
                    id: id.clone(),
                    local_type: hir::LocalType(ty.clone(), hir::LocalStorage::Local, None),
                    init: Some(hir::Initializer::Expression(expr)),
                };
                let statement = hir::Statement::Var(vd);
                let var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                let p = ProcessedBind {
                    reference: Some(hir::Expression::Variable(var_ref)),
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                Ok((vec![statement], p))
            }
            Command::Trivial(expr, MutationParam::Const) => {
                let expr = hir_relocate(expr, locals);
                let p = ProcessedBind {
                    reference: Some(expr),
                    write_back: WriteBack::Fail,
                    recursive_write_back: None,
                    ty: ty,
                };
                Ok((vec![], p))
            }
            _ => Err(CombineError::FailedToResolveMultiPartExpression),
        }
    }

    let mut processed_binds = HashMap::new();

    let mut statements = Vec::with_capacity(seq.binds.len() + 1);
    for bind in seq.binds {

        let r = try!(build_command(bind.value,
                                   bind.required_input,
                                   bind.ty,
                                   &required_locals,
                                   &processed_binds,
                                   context));

        let (sts, processed) = r;
        processed_binds.insert(bind.id, processed);
        for st in sts {
            statements.push(st);
        }
    }

    let last_expr = match seq.last {
        Some(last) => {
            match processed_binds.get(&last) {
                Some(p) => p.reference.clone(),
                None => return Err(CombineError::LastBindDoesNotExist),
            }
        }
        None => None,
    };

    let combined_block = CombinedBlock {
        statements: statements,
        last_expression: last_expr,
    };

    Ok(combined_block)
}

enum TrivialResult {
    Trivial(hir::Expression),
    Complex(Sequence),
}

fn combine_trivial(seq: Sequence) -> TrivialResult {
    // If the last segment is trivial, then the whole thing must be trivial
    let is_trivial = match seq.binds.last() {
        Some(last) => {
            match seq.last {
                Some(required_last) => assert_eq!(last.id, required_last),
                None => {}
            };
            match last.value {
                Command::Trivial(_, _) => true,
                _ => false,
            }
        }
        None => false,
    };
    if is_trivial {
        let mut seq = seq;
        match seq.binds.pop() {
            Some(Bind { value: Command::Trivial(expr, _), .. }) => TrivialResult::Trivial(expr),
            _ => panic!("bad is_trivial logic"),
        }
    } else {
        TrivialResult::Complex(seq)
    }
}

pub fn combine(seq: Sequence, context: &mut CombineContext) -> CombineResult<CombinedExpression> {
    let seq = combine_group(seq);
    match combine_trivial(seq) {
        TrivialResult::Trivial(expr) => Ok(CombinedExpression::Single(expr)),
        TrivialResult::Complex(seq) => {
            Ok(CombinedExpression::Multi(try!(combine_complex(seq, context))))
        }
    }
}

#[test]
fn test_combine_single_variable() {

    let var_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_seq = Sequence {
        binds: vec![
            Bind::direct(0, Command::Variable(var_ref.clone()), hir::Type::float()),
        ],
        last: Some(BindId(0)),
    };
    let var_expr = combine(var_seq, &mut FakeCombineContext).expect("combine failed");
    let var_0 = hir::Expression::Variable(var_ref);
    assert_eq!(var_expr, CombinedExpression::Single(var_0));
}

#[test]
fn test_combine_binary_operation() {
    let var_0_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_1_ref = hir::VariableRef(hir::VariableId(1), hir::ScopeRef(0));
    let dty = hir::DataType(hir::DataLayout::Scalar(hir::ScalarType::Float),
                            hir::TypeModifier::default());
    let ty = hir::Type::from_data(dty.clone());
    let add = hir::Intrinsic2::Add(dty.clone());
    let bin_seq = Sequence {
        binds: vec![
            Bind::direct(0, Command::Variable(var_0_ref.clone()), ty.clone()),
            Bind::direct(1, Command::Variable(var_1_ref.clone()), ty.clone()),
            Bind::direct(2, Command::Intrinsic2(add.clone(), BindId(0), BindId(1)), ty),
        ],
        last: Some(BindId(2)),
    };
    let bin_expr = combine(bin_seq, &mut FakeCombineContext).expect("combine failed");
    let var_0 = hir::Expression::Variable(var_0_ref);
    let var_1 = hir::Expression::Variable(var_1_ref);
    let add = hir::Expression::Intrinsic2(add, Box::new(var_0), Box::new(var_1));
    assert_eq!(bin_expr, CombinedExpression::Single(add));
}

#[test]
fn test_combine_texture_assignment() {

    let lit_zero = hir::Literal::Int(0);
    let dtyl_index = hir::DataLayout::Vector(hir::ScalarType::Int, 2);

    let dtyl = hir::DataLayout::Vector(hir::ScalarType::Float, 4);
    let dty = hir::DataType(dtyl, hir::TypeModifier::default());
    let ty = hir::Type::from_data(dty.clone());
    let tex_ty = {
        let tex_obj = hir::ObjectType::RWTexture2D(dty.clone());
        hir::Type::from_layout(hir::TypeLayout::Object(tex_obj))
    };
    let tex_0 = hir::GlobalId(0);
    let tex_1 = hir::GlobalId(1);

    let assign = hir::Intrinsic2::Assignment(hir::Type::from_data(dty.clone()));

    let rel = Sequence {
        binds: vec![
            Bind::direct(0, Command::Global(tex_0), tex_ty.clone()),
            Bind::direct(1, Command::Literal(lit_zero.clone()), hir::Type::int()),
            Bind::direct(2, Command::Literal(lit_zero.clone()), hir::Type::int()),
            Bind::direct(3, Command::NumericConstructor(dtyl_index.clone(), vec![ConstructorSlot { arity: 1, expr: BindId(1) }, ConstructorSlot { arity: 1, expr: BindId(2) }]), hir::Type::intn(2)),
            Bind {
                id: BindId(4),
                value: Command::RWTexture2DIndex(dty.clone(), BindId(0), BindId(3)),
                required_input: InputModifier::Out,
                ty: ty.clone(),
            },
            Bind::direct(5, Command::Global(tex_1), tex_ty),
            Bind::direct(6, Command::Literal(lit_zero.clone()), hir::Type::int()),
            Bind::direct(7, Command::Literal(lit_zero.clone()), hir::Type::int()),
            Bind::direct(8, Command::NumericConstructor(dtyl_index, vec![ConstructorSlot { arity: 1, expr: BindId(6) }, ConstructorSlot { arity: 1, expr: BindId(7) }]), hir::Type::intn(2)),
            Bind::direct(9, Command::Intrinsic2(hir::Intrinsic2::Texture2DLoad(dty), BindId(5), BindId(8)), ty.clone()),
            Bind::direct(10, Command::Intrinsic2(assign, BindId(4), BindId(9)), ty),
        ],
        last: Some(BindId(10)),
    };
    let combined = combine(rel, &mut ScopeCombineContext::new()).expect("combine failed");
    let block = CombinedBlock {
        statements: vec![
            hir::Statement::Var(hir::VarDef {
                id: hir::VariableId(0),
                local_type: hir::LocalType(hir::Type::intn(2), hir::LocalStorage::Local, None),
                init: Some(hir::Initializer::Expression(
                    hir::Expression::NumericConstructor(
                        hir::DataLayout::Vector(hir::ScalarType::Int, 2),
                        vec![
                            hir::ConstructorSlot {
                                arity: 1,
                                expr: hir::Expression::Literal(hir::Literal::Int(0))
                            },
                            hir::ConstructorSlot {
                                arity: 1,
                                expr: hir::Expression::Literal(hir::Literal::Int(0))
                            },
                        ]
                    )
                ))
            }),
            hir::Statement::Var(hir::VarDef {
                id: hir::VariableId(1),
                local_type: hir::LocalType(hir::Type::floatn(4), hir::LocalStorage::Local, None),
                init: None
            }),
            hir::Statement::Var(hir::VarDef {
                id: hir::VariableId(2),
                local_type: hir::LocalType(hir::Type::floatn(4), hir::LocalStorage::Local, None),
                init: Some(hir::Initializer::Expression(
                    hir::Expression::Intrinsic2(
                        hir::Intrinsic2::Texture2DLoad(hir::DataType(
                            hir::DataLayout::Vector(hir::ScalarType::Float, 4),
                            hir::TypeModifier::default()
                        )),
                        Box::new(hir::Expression::Global(hir::GlobalId(1))),
                        Box::new(hir::Expression::NumericConstructor(
                            hir::DataLayout::Vector(hir::ScalarType::Int, 2),
                            vec![
                                hir::ConstructorSlot {
                                    arity: 1,
                                    expr: hir::Expression::Literal(hir::Literal::Int(0))
                                },
                                hir::ConstructorSlot {
                                    arity: 1,
                                    expr: hir::Expression::Literal(hir::Literal::Int(0))
                                },
                            ]
                        ))
                    )
                ))
            }),
            hir::Statement::Var(hir::VarDef {
                id: hir::VariableId(3),
                local_type: hir::LocalType(hir::Type::floatn(4), hir::LocalStorage::Local, None),
                init: Some(hir::Initializer::Expression(
                    hir::Expression::Intrinsic2(
                        hir::Intrinsic2::Assignment(hir::Type::floatn(4)),
                        Box::new(hir::Expression::Variable(hir::VariableRef::raw(1, 0))),
                        Box::new(hir::Expression::Variable(hir::VariableRef::raw(2, 0)))
                    )
                ))
            }),
            hir::Statement::Expression(hir::Expression::Intrinsic3(
                hir::Intrinsic3::RWTexture2DStore(hir::DataType(
                    hir::DataLayout::Vector(hir::ScalarType::Float, 4),
                    hir::TypeModifier::default()
                )),
                Box::new(hir::Expression::Global(hir::GlobalId(0))),
                Box::new(hir::Expression::Variable(hir::VariableRef::raw(0, 0))),
                Box::new(hir::Expression::Variable(hir::VariableRef::raw(1, 0)))
            )),
        ],
        last_expression: Some(hir::Expression::Variable(hir::VariableRef::raw(3, 0))),
    };
    assert_eq!(combined, CombinedExpression::Multi(block));
}
