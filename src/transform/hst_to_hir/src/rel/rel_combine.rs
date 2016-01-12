
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
        statements.push(hir::Statement::Expression(block.last_expression));
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
}

#[derive(PartialEq, Debug, Clone)]
pub struct CombinedBlock {
    statements: Vec<hir::Statement>,
    last_expression: hir::Expression,
}

#[derive(PartialEq, Debug, Clone)]
pub enum CombinedExpression {
    Single(hir::Expression),
    Multi(CombinedBlock),
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
            Command::TernaryConditional(ref cond, ref lhs, ref rhs) => {
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
            Command::Texture2DIndex(_, _, _) => None,
            Command::RWTexture2DIndex(_, _, _) => None,
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

    fn build_bind(bind: &Bind, q: BuiltQueue) -> Option<BuiltQueue> {
        match build_command(&bind.value, q) {
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

    if queue.0.len() == 1 && queue.0[0].0 == seq.last {
        Some(queue.0[0].1.clone())
    } else {
        None
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
        reference: hir::Expression,
        /// Expression to write output parameter
        write_back: Option<hir::Expression>,
        /// Trigger write back for previous bind
        recursive_write_back: Option<BindId>,
    }

    impl ProcessedBind {
        fn enqueue_write_back(&self,
                              sts: &mut Vec<hir::Statement>,
                              processed: &HashMap<BindId, ProcessedBind>) {
            match self.write_back {
                Some(ref wb) => sts.push(hir::Statement::Expression(wb.clone())),
                None => panic!("out param used for expression without write back"),
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
            Command::Literal(lit) => {
                // Substitute literals
                let processed = ProcessedBind {
                    reference: hir::Expression::Literal(lit),
                    write_back: None,
                    recursive_write_back: None,
                };
                Ok((vec![], processed))
            }
            Command::Variable(var_prev_scope) => {
                // Build a hir node to the local variable
                let var = {
                    // Ref to input variable
                    let var_ref = match locals.get(&var_prev_scope) {
                        Some(var_ref) => var_ref,
                        None => panic!("command uses variable that's not in required locals"),
                    };
                    hir::Expression::Variable(var_ref.clone())
                };
                // Create temporary local
                let id = try!(allocate_local("var", ty.clone(), context));
                // Push the temporary local into a hir node
                let tmp_var = {
                    let tmp_var_ref = hir::VariableRef(id.clone(), hir::ScopeRef(0));
                    hir::Expression::Variable(tmp_var_ref)
                };
                let statement = {
                    let init = if im != InputModifier::Out {
                        Some(hir::Initializer::Expression(var.clone()))
                    } else {
                        None
                    };
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty, hir::LocalStorage::Local, None),
                        init: init,
                    };
                    hir::Statement::Var(vd)
                };
                let write_back = if im != InputModifier::In {
                    let assign = hir::Intrinsic2::Assignment(hir::Type::void());
                    let write_to = Box::new(var.clone());
                    let write_from = Box::new(tmp_var.clone());
                    Some(hir::Expression::Intrinsic2(assign, write_to, write_from))
                } else {
                    None
                };
                // Finish
                let p = ProcessedBind {
                    reference: tmp_var,
                    write_back: write_back,
                    recursive_write_back: None,
                };
                Ok((vec![statement], p))
            }
            Command::Global(id) => {
                let p = ProcessedBind {
                    reference: hir::Expression::Global(id.clone()),
                    write_back: None,
                    recursive_write_back: None,
                };
                Ok((vec![], p))
            }
            Command::RWTexture2DIndex(dty, tex, index) => {
                let tex_ref = match processed.get(&tex) {
                    Some(p) => p.reference.clone(),
                    None => panic!("reference local bind does not exist (RWTexture2DIndex: tex)"),
                };
                let index_ref = match processed.get(&index) {
                    Some(p) => p.reference.clone(),
                    None => panic!("reference local bind does not exist (RWTexture2DIndex: index)"),
                };
                let load = {
                    let li = hir::Intrinsic2::RWTexture2DLoad(dty.clone());
                    let texture = Box::new(tex_ref.clone());
                    let index = Box::new(index_ref.clone());
                    hir::Expression::Intrinsic2(li, texture, index)
                };
                let (statements, reference, write_back) = if im == InputModifier::In {
                    (vec![], load, None)
                } else {
                    let init = if im == InputModifier::InOut {
                        Some(hir::Initializer::Expression(load))
                    } else {
                        None
                    };
                    let id = try!(allocate_local("tex", ty.clone(), context));
                    let vd = hir::VarDef {
                        id: id.clone(),
                        local_type: hir::LocalType(ty, hir::LocalStorage::Local, None),
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
                    (vec![hir::Statement::Var(vd)], var_ref, Some(store))
                };
                let p = ProcessedBind {
                    reference: reference,
                    write_back: write_back,
                    recursive_write_back: None,
                };
                Ok((statements, p))
            }
            Command::NumericConstructor(dtyl, cons) => {
                let id = try!(allocate_local("cons", ty.clone(), context));
                let mut hir_cons = vec![];
                for con in cons {
                    let expr = match processed.get(&con.expr) {
                        Some(p) => p.reference.clone(),
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
                    local_type: hir::LocalType(ty, hir::LocalStorage::Local, None),
                    init: Some(hir::Initializer::Expression(init)),
                };
                let statement = hir::Statement::Var(vd);
                let p = ProcessedBind {
                    reference: hir::Expression::Variable(hir::VariableRef(id.clone(),
                                                                          hir::ScopeRef(0))),
                    write_back: None,
                    recursive_write_back: None,
                };
                Ok((vec![statement], p))
            }
            Command::Intrinsic2(i, b1, b2) => {
                let id = try!(allocate_local("i2", ty.clone(), context));
                let p1 = match processed.get(&b1) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic2: b1)"),
                };
                let e1 = p1.reference.clone();
                let im1 = i.get_param1_input_modifier();
                let p2 = match processed.get(&b2) {
                    Some(p) => p,
                    None => panic!("reference local bind does not exist (Intrinsic2: b2)"),
                };
                let e2 = p2.reference.clone();
                let im2 = i.get_param2_input_modifier();
                let init = hir::Expression::Intrinsic2(i, Box::new(e1), Box::new(e2));
                let vd = hir::VarDef {
                    id: id.clone(),
                    local_type: hir::LocalType(ty, hir::LocalStorage::Local, None),
                    init: Some(hir::Initializer::Expression(init)),
                };
                let mut statements = vec![hir::Statement::Var(vd)];
                let p = ProcessedBind {
                    reference: hir::Expression::Variable(hir::VariableRef(id.clone(),
                                                                          hir::ScopeRef(0))),
                    write_back: None,
                    recursive_write_back: None,
                };
                if im1 != InputModifier::In {
                    p1.enqueue_write_back(&mut statements, processed);
                }
                if im2 != InputModifier::In {
                    p1.enqueue_write_back(&mut statements, processed);
                }
                Ok((statements, p))
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

    let last_expr = match processed_binds.get(&seq.last) {
        Some(p) => p.reference.clone(),
        None => return Err(CombineError::LastBindDoesNotExist),
    };

    let combined_block = CombinedBlock {
        statements: statements,
        last_expression: last_expr,
    };

    Ok(combined_block)
}

pub fn combine(seq: Sequence, context: &mut CombineContext) -> CombineResult<CombinedExpression> {
    match combine_trivial(&seq) {
        Some(exp) => return Ok(CombinedExpression::Single(exp)),
        None => {}
    }
    Ok(CombinedExpression::Multi(try!(combine_complex(seq, context))))
}

#[test]
fn test_combine_single_variable() {

    let var_ref = hir::VariableRef(hir::VariableId(0), hir::ScopeRef(0));
    let var_seq = Sequence {
        binds: vec![
            Bind::direct(0, Command::Variable(var_ref.clone()), hir::Type::float()),
        ],
        last: BindId(0),
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
        last: BindId(2),
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
        last: BindId(10),
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
                id: hir::VariableId(3),
                local_type: hir::LocalType(hir::Type::floatn(4), hir::LocalStorage::Local, None),
                init: Some(hir::Initializer::Expression(
                    hir::Expression::Intrinsic2(
                        hir::Intrinsic2::Texture2DLoad(hir::DataType(
                            hir::DataLayout::Vector(hir::ScalarType::Float, 4),
                            hir::TypeModifier::default()
                        )),
                        Box::new(hir::Expression::Global(hir::GlobalId(1))),
                        Box::new(hir::Expression::Variable(hir::VariableRef::raw(2, 0)))
                    )
                ))
            }),
            hir::Statement::Var(hir::VarDef {
                id: hir::VariableId(4),
                local_type: hir::LocalType(hir::Type::floatn(4), hir::LocalStorage::Local, None),
                init: Some(hir::Initializer::Expression(
                    hir::Expression::Intrinsic2(
                        hir::Intrinsic2::Assignment(hir::Type::floatn(4)),
                        Box::new(hir::Expression::Variable(hir::VariableRef::raw(1, 0))),
                        Box::new(hir::Expression::Variable(hir::VariableRef::raw(3, 0)))
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
        last_expression: hir::Expression::Variable(hir::VariableRef::raw(4, 0)),
    };
    assert_eq!(combined, CombinedExpression::Multi(block));
}
