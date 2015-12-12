
use std::collections::HashSet;
use std::collections::HashMap;
use super::ir::*;

#[derive(PartialEq, Debug, Clone)]
pub struct FunctionGlobalUsage {
    pub globals: HashSet<GlobalId>,
    pub cbuffers: HashSet<ConstantBufferId>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct GlobalUsage {
    pub functions: HashMap<FunctionId, FunctionGlobalUsage>,
    pub image_reads: HashSet<GlobalId>,
    pub image_writes: HashSet<GlobalId>,
}

#[derive(PartialEq, Debug, Clone)]
struct LocalFunctionGlobalUsage {
    globals: HashSet<GlobalId>,
    cbuffers: HashSet<ConstantBufferId>,
    functions: HashSet<FunctionId>,
    image_reads: HashSet<GlobalId>,
    image_writes: HashSet<GlobalId>,
}

impl GlobalUsage {
    pub fn analyse(root_definitions: &[RootDefinition]) -> GlobalUsage {
        let mut lfgus = HashMap::new();
        for root in root_definitions {
            match *root {
                RootDefinition::Function(ref func) => {
                    lfgus.insert(func.id, LocalFunctionGlobalUsage::analyse(func));
                }
                _ => { }
            }
        }
        let mut image_reads = HashSet::new();
        let mut image_writes = HashSet::new();
        for (_, current_usage) in &lfgus {
            image_reads = image_reads.union(&current_usage.image_reads).cloned().collect::<HashSet<_>>();
            image_writes = image_writes.union(&current_usage.image_writes).cloned().collect::<HashSet<_>>();
        };
        let mut prev = lfgus;
        loop {
            let mut next = prev.clone();
            for (_, current_usage) in next.iter_mut() {
                for (other_id, other_usage) in &prev {
                    if current_usage.functions.contains(&other_id) {
                        current_usage.globals = current_usage.globals.union(&other_usage.globals).cloned().collect::<HashSet<_>>();
                        current_usage.cbuffers = current_usage.cbuffers.union(&other_usage.cbuffers).cloned().collect::<HashSet<_>>();
                    }
                }
            }
            if &prev != &next {
                prev = next;
            } else {
                break;
            }
        }
        let mut usage = GlobalUsage {
            functions: HashMap::new(),
            image_reads: image_reads.clone(),
            image_writes: image_writes.clone(),
        };
        for (id, local_usage) in prev {
            usage.functions.insert(id, FunctionGlobalUsage {
                globals: local_usage.globals.clone(),
                cbuffers: local_usage.cbuffers.clone(),
            });
        }
        usage
    }
}

impl LocalFunctionGlobalUsage {
    fn analyse(function: &FunctionDefinition) -> LocalFunctionGlobalUsage {
        let mut usage = LocalFunctionGlobalUsage {
            globals: HashSet::new(),
            cbuffers: HashSet::new(),
            functions: HashSet::new(),
            image_reads: HashSet::new(),
            image_writes: HashSet::new(),
        };
        search_scope_block(&function.scope_block, &mut usage);
        usage
    }
}

fn search_scope_block(sb: &ScopeBlock, usage: &mut LocalFunctionGlobalUsage) {
    for statement in &sb.0 {
        search_statement(statement, usage);
    };
}

fn search_statement(statement: &Statement, usage: &mut LocalFunctionGlobalUsage) {
    match *statement {
        Statement::Expression(ref expr) => search_expression(expr, usage),
        Statement::Var(ref vd) => search_vardef(vd, usage),
        Statement::Block(ref sb) => search_scope_block(sb, usage),
        Statement::If(ref cond, ref sb) | Statement::While(ref cond, ref sb) => {
            search_expression(cond, usage);
            search_scope_block(sb, usage);
        },
        Statement::For(ref init, ref cond, ref update, ref sb)=> {
            search_initexpression(init, usage);
            search_expression(cond, usage);
            search_expression(update, usage);
            search_scope_block(sb, usage);
        },
        Statement::Return(ref expr) => search_expression(expr, usage),
    }
}

fn search_initexpression(init: &Condition, usage: &mut LocalFunctionGlobalUsage) {
    match *init {
        Condition::Expr(ref expr) => search_expression(expr, usage),
        Condition::Assignment(ref vd) => search_vardef(vd, usage),
    }
}

fn search_vardef(vd: &VarDef, usage: &mut LocalFunctionGlobalUsage) {
    match vd.assignment {
        Some(ref expr) => search_expression(expr, usage),
        None => { },
    }
}

fn search_expression(expression: &Expression, usage: &mut LocalFunctionGlobalUsage) {
    match *expression {
        Expression::Literal(_) | Expression::Variable(_) => { },
        Expression::Global(ref id) => { usage.globals.insert(id.clone()); },
        Expression::ConstantVariable(ref id, _) => { usage.cbuffers.insert(id.clone()); },
        Expression::UnaryOperation(_, ref expr) => search_expression(expr, usage),
        Expression::BinaryOperation(_, ref lhs, ref rhs) => {
            search_expression(lhs, usage);
            search_expression(rhs, usage);
        },
        Expression::TernaryConditional(ref cond, ref left, ref right) => {
            search_expression(cond, usage);
            search_expression(left, usage);
            search_expression(right, usage);
        },
        Expression::ArraySubscript(ref arr, ref index) => {
            search_expression(arr, usage);
            search_expression(index, usage);
        },
        Expression::Member(ref expr, _) => search_expression(expr, usage),
        Expression::Call(ref id, ref exprs) => {
            usage.functions.insert(id.clone());
            for expr in exprs {
                search_expression(expr, usage);
            }
        },
        Expression::Cast(_, ref expr) => search_expression(expr, usage),
        Expression::Intrinsic(ref intrinsic) => search_intrinsic(intrinsic, usage),
    }
}

fn search_intrinsic(intrinsic: &Intrinsic, usage: &mut LocalFunctionGlobalUsage) {
    match *intrinsic {
        Intrinsic::AllMemoryBarrier |
        Intrinsic::AllMemoryBarrierWithGroupSync |
        Intrinsic::DeviceMemoryBarrier |
        Intrinsic::DeviceMemoryBarrierWithGroupSync |
        Intrinsic::GroupMemoryBarrier |
        Intrinsic::GroupMemoryBarrierWithGroupSync => { },

        Intrinsic::AsIntU(ref e1) |
        Intrinsic::AsIntU1(ref e1) |
        Intrinsic::AsIntU2(ref e1) |
        Intrinsic::AsIntU3(ref e1) |
        Intrinsic::AsIntU4(ref e1) |
        Intrinsic::AsIntF(ref e1) |
        Intrinsic::AsIntF1(ref e1) |
        Intrinsic::AsIntF2(ref e1) |
        Intrinsic::AsIntF3(ref e1) |
        Intrinsic::AsIntF4(ref e1) |
        Intrinsic::AsUIntI(ref e1) |
        Intrinsic::AsUIntI1(ref e1) |
        Intrinsic::AsUIntI2(ref e1) |
        Intrinsic::AsUIntI3(ref e1) |
        Intrinsic::AsUIntI4(ref e1) |
        Intrinsic::AsUIntF(ref e1) |
        Intrinsic::AsUIntF1(ref e1) |
        Intrinsic::AsUIntF2(ref e1) |
        Intrinsic::AsUIntF3(ref e1) |
        Intrinsic::AsUIntF4(ref e1) |
        Intrinsic::AsFloatI(ref e1) |
        Intrinsic::AsFloatI1(ref e1) |
        Intrinsic::AsFloatI2(ref e1) |
        Intrinsic::AsFloatI3(ref e1) |
        Intrinsic::AsFloatI4(ref e1) |
        Intrinsic::AsFloatU(ref e1) |
        Intrinsic::AsFloatU1(ref e1) |
        Intrinsic::AsFloatU2(ref e1) |
        Intrinsic::AsFloatU3(ref e1) |
        Intrinsic::AsFloatU4(ref e1) |
        Intrinsic::AsFloatF(ref e1) |
        Intrinsic::AsFloatF1(ref e1) |
        Intrinsic::AsFloatF2(ref e1) |
        Intrinsic::AsFloatF3(ref e1) |
        Intrinsic::AsFloatF4(ref e1) => {
            search_expression(e1, usage);
        }

        Intrinsic::AsDouble(ref e1, ref e2) |
        Intrinsic::Cross(ref e1, ref e2) |
        Intrinsic::Distance1(ref e1, ref e2) |
        Intrinsic::Distance2(ref e1, ref e2) |
        Intrinsic::Distance3(ref e1, ref e2) |
        Intrinsic::Distance4(ref e1, ref e2) |
        Intrinsic::DotI1(ref e1, ref e2) |
        Intrinsic::DotI2(ref e1, ref e2) |
        Intrinsic::DotI3(ref e1, ref e2) |
        Intrinsic::DotI4(ref e1, ref e2) |
        Intrinsic::DotF1(ref e1, ref e2) |
        Intrinsic::DotF2(ref e1, ref e2) |
        Intrinsic::DotF3(ref e1, ref e2) |
        Intrinsic::DotF4(ref e1, ref e2) |
        Intrinsic::Min(ref e1, ref e2) |
        Intrinsic::Max(ref e1, ref e2) |
        Intrinsic::BufferLoad(ref e1, ref e2) |
        Intrinsic::RWBufferLoad(ref e1, ref e2) |
        Intrinsic::StructuredBufferLoad(ref e1, ref e2) |
        Intrinsic::RWStructuredBufferLoad(ref e1, ref e2) => {
            search_expression(e1, usage);
            search_expression(e2, usage);
        }

        Intrinsic::RWTexture2DLoad(ref e1, ref e2) => {

            // Mark textures as used in reading
            // This is not good enough for all HLSL use cases (such as reading
            // from a struct member, but we can't do that in OpenCL anyway)
            match *e1 {
                Expression::Global(ref id) => {
                    usage.image_reads.insert(id.clone());
                },
                _ => unimplemented!(),
            };

            search_expression(e1, usage);
            search_expression(e2, usage);
        }

        Intrinsic::ClampI(ref e1, ref e2, ref e3) |
        Intrinsic::ClampI1(ref e1, ref e2, ref e3) |
        Intrinsic::ClampI2(ref e1, ref e2, ref e3) |
        Intrinsic::ClampI3(ref e1, ref e2, ref e3) |
        Intrinsic::ClampI4(ref e1, ref e2, ref e3) |
        Intrinsic::ClampF(ref e1, ref e2, ref e3) |
        Intrinsic::ClampF1(ref e1, ref e2, ref e3) |
        Intrinsic::ClampF2(ref e1, ref e2, ref e3) |
        Intrinsic::ClampF3(ref e1, ref e2, ref e3) |
        Intrinsic::ClampF4(ref e1, ref e2, ref e3) => {
            search_expression(e1, usage);
            search_expression(e2, usage);
            search_expression(e3, usage);
        }

        Intrinsic::Float4(ref e1, ref e2, ref e3, ref e4) => {
            search_expression(e1, usage);
            search_expression(e2, usage);
            search_expression(e3, usage);
            search_expression(e4, usage);
        }
    }
}
