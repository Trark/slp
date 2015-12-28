
use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use slp_lang_cil as src;
use slp_lang_cst as dst;

#[derive(PartialEq, Debug, Clone)]
pub enum UntyperError {
    LocalVariableNotFound(src::LocalId),
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum GlobalType {
    Function(src::FunctionId),
    Struct(src::StructId),
    Variable(src::GlobalId),
}

struct Context {
    global_name_map: HashMap<GlobalType, dst::Identifier>,
    local_scope: Option<HashMap<src::LocalId, dst::Identifier>>,
}

impl Context {
    fn from_globals(globals: &src::GlobalDeclarations) -> Result<Context, UntyperError> {
        let mut context = Context {
            global_name_map: HashMap::new(),
            local_scope: None,
        };

        // Insert global variables
        {
            for (var_id, var_name) in &globals.globals {
                context.insert_identifier(GlobalType::Variable(var_id.clone()), var_name);
            }
        }

        // Insert functions (try to generate [name]_[overload number] for overloads)
        {
            let mut grouped_functions: HashMap<String, Vec<src::FunctionId>> = HashMap::new();
            for (id, name) in globals.functions.iter() {
                match grouped_functions.entry(name.clone()) {
                    Entry::Occupied(mut occupied) => {
                        occupied.get_mut().push(id.clone());
                    }
                    Entry::Vacant(vacant) => {
                        vacant.insert(vec![id.clone()]);
                    }
                }
            }
            for (key, mut ids) in grouped_functions {
                assert!(ids.len() > 0);
                if ids.len() == 1 {
                    context.insert_identifier(GlobalType::Function(ids[0]), &key);
                } else {
                    ids.sort();
                    for (index, id) in ids.iter().enumerate() {
                        let gen = key.clone() + "_" + &index.to_string();
                        context.insert_identifier(GlobalType::Function(*id), &gen);
                    }
                }
            }
        }

        // Insert structs (name collisions possible with function overloads + globals)
        {
            for (id, struct_name) in globals.structs.iter() {
                context.insert_identifier(GlobalType::Struct(id.clone()), struct_name);
            }
        }

        Ok(context)
    }

    fn is_free(&self, identifier: &str) -> bool {
        for name in self.global_name_map.values() {
            if name == identifier {
                return false;
            }
        }
        if let Some(ref locals) = self.local_scope {
            for name in locals.values() {
                if name == identifier {
                    return false;
                }
            }
        }
        return true;
    }

    fn make_identifier(&self, name: &str) -> String {
        if self.is_free(name) {
            return name.to_string();
        }
        let mut index = 0;
        loop {
            let test_name = name.to_string() + "_" + &index.to_string();
            if self.is_free(&test_name) {
                return test_name;
            }
            index = index + 1;
        }
    }

    fn insert_identifier(&mut self, id: GlobalType, name: &str) {
        let identifier = self.make_identifier(name);
        let r = self.global_name_map.insert(id, identifier);
        assert_eq!(r, None);
    }

    fn push_scope(&mut self, locals: &src::LocalDeclarations) {
        assert_eq!(self.local_scope, None);
        self.local_scope = Some(HashMap::new());
        let mut keys = locals.locals.keys().collect::<Vec<&src::LocalId>>();
        keys.sort();
        for local_id in keys {
            let candidate_name = locals.locals.get(local_id).expect("bad local key");
            let identifier = self.make_identifier(candidate_name);
            match self.local_scope {
                Some(ref mut scope) => scope.insert(local_id.clone(), identifier),
                None => unreachable!(),
            };
        }
    }

    fn pop_scope(&mut self) {
        assert!(!self.local_scope.is_none());
        self.local_scope = None;
    }

    fn get_global_name(&self, id: &src::GlobalId) -> Result<dst::Identifier, UntyperError> {
        Ok(self.global_name_map
               .get(&GlobalType::Variable(id.clone()))
               .expect("untyper: no global")
               .clone())
    }

    fn get_function_name(&self, id: &src::FunctionId) -> Result<dst::Identifier, UntyperError> {
        Ok(self.global_name_map
               .get(&GlobalType::Function(id.clone()))
               .expect("untyper: no function")
               .clone())
    }

    fn get_struct_name(&self, id: &src::StructId) -> Result<dst::Identifier, UntyperError> {
        Ok(self.global_name_map
               .get(&GlobalType::Struct(id.clone()))
               .expect("untyper: no struct")
               .clone())
    }

    fn get_local_name(&self, id: &src::LocalId) -> Result<dst::Identifier, UntyperError> {
        Ok(match self.local_scope {
            Some(ref scope) => scope.get(id).expect("untyper: no local variable").clone(),
            None => return Err(UntyperError::LocalVariableNotFound(id.clone())),
        })
    }
}

impl error::Error for UntyperError {
    fn description(&self) -> &str {
        match *self {
            UntyperError::LocalVariableNotFound(_) => "local variable not found",
        }
    }
}

impl fmt::Display for UntyperError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}


fn result_map<T, G, F>(func: F, inputs: &[T], context: &mut Context) -> Result<Vec<G>, UntyperError>
    where F: Fn(&T, &mut Context) -> Result<G, UntyperError>
{
    inputs.iter().fold(Ok(vec![]), |vec, next| {
        let mut vec = match vec {
            Ok(vec) => vec,
            Err(err) => return Err(err),
        };
        vec.push(try!(func(next, context)));
        Ok(vec)
    })
}

fn untype_type(ty: &src::Type, context: &mut Context) -> Result<dst::Type, UntyperError> {
    Ok(match *ty {
        src::Type::Void => dst::Type::Void,
        src::Type::Bool => dst::Type::Bool,
        src::Type::Scalar(ref scalar) => dst::Type::Scalar(scalar.clone()),
        src::Type::Vector(ref scalar, ref dim) => dst::Type::Vector(scalar.clone(), dim.clone()),
        src::Type::SizeT => dst::Type::SizeT,
        src::Type::PtrDiffT => dst::Type::PtrDiffT,
        src::Type::IntPtrT => dst::Type::IntPtrT,
        src::Type::UIntPtrT => dst::Type::UIntPtrT,
        src::Type::Struct(ref identifier) => {
            dst::Type::Struct(try!(context.get_struct_name(identifier)))
        }
        src::Type::Pointer(ref address_space, ref inner) => {
            dst::Type::Pointer(address_space.clone(),
                               Box::new(try!(untype_type(inner, context))))
        }
        src::Type::Array(ref inner, dim) => {
            dst::Type::Array(Box::new(try!(untype_type(inner, context))), dim)
        }
        src::Type::Image1D(ref address_space) => dst::Type::Image1D(address_space.clone()),
        src::Type::Image1DBuffer(ref address_space) => {
            dst::Type::Image1DBuffer(address_space.clone())
        }
        src::Type::Image1DArray(ref address_space) => {
            dst::Type::Image1DArray(address_space.clone())
        }
        src::Type::Image2D(ref address_space) => dst::Type::Image2D(address_space.clone()),
        src::Type::Image2DArray(ref address_space) => {
            dst::Type::Image2DArray(address_space.clone())
        }
        src::Type::Image2DDepth(ref address_space) => {
            dst::Type::Image2DDepth(address_space.clone())
        }
        src::Type::Image2DArrayDepth(ref address_space) => {
            dst::Type::Image2DArrayDepth(address_space.clone())
        }
        src::Type::Image3D(ref address_space) => dst::Type::Image3D(address_space.clone()),
        src::Type::Image3DArray(ref address_space) => {
            dst::Type::Image3DArray(address_space.clone())
        }
        src::Type::Sampler => dst::Type::Sampler,
        src::Type::Queue => dst::Type::Queue,
        src::Type::NDRange => dst::Type::NDRange,
        src::Type::ClkEvent => dst::Type::ClkEvent,
        src::Type::ReserveId => dst::Type::ReserveId,
        src::Type::Event => dst::Type::Event,
        src::Type::MemFenceFlags => dst::Type::MemFenceFlags,
    })
}

fn untype_constructor(cons: &src::Constructor,
                      context: &mut Context)
                      -> Result<dst::Constructor, UntyperError> {
    Ok(match *cons {
        src::Constructor::UInt3(ref e1, ref e2, ref e3) => {
            let u1 = Box::new(try!(untype_expression(e1, context)));
            let u2 = Box::new(try!(untype_expression(e2, context)));
            let u3 = Box::new(try!(untype_expression(e3, context)));
            dst::Constructor::UInt3(u1, u2, u3)
        }
        src::Constructor::Float4(ref e1, ref e2, ref e3, ref e4) => {
            let u1 = Box::new(try!(untype_expression(e1, context)));
            let u2 = Box::new(try!(untype_expression(e2, context)));
            let u3 = Box::new(try!(untype_expression(e3, context)));
            let u4 = Box::new(try!(untype_expression(e4, context)));
            dst::Constructor::Float4(u1, u2, u3, u4)
        }
    })
}

fn untype_intrinsic(instrinic: &src::Intrinsic,
                    context: &mut Context)
                    -> Result<dst::Intrinsic, UntyperError> {
    Ok(match *instrinic {
        src::Intrinsic::GetGlobalId(ref expr) => {
            dst::Intrinsic::GetGlobalId(Box::new(try!(untype_expression(expr, context))))
        }
    })
}

fn untype_expression(expression: &src::Expression,
                     context: &mut Context)
                     -> Result<dst::Expression, UntyperError> {
    Ok(match *expression {
        src::Expression::Literal(ref literal) => dst::Expression::Literal(literal.clone()),
        src::Expression::Constructor(ref cons) => {
            dst::Expression::Constructor(try!(untype_constructor(cons, context)))
        }
        src::Expression::Local(ref id) => {
            dst::Expression::Variable(try!(context.get_local_name(id)))
        }
        src::Expression::Global(ref id) => {
            dst::Expression::Variable(try!(context.get_global_name(id)))
        }
        src::Expression::UnaryOperation(ref un, ref expr) => {
            dst::Expression::UnaryOperation(un.clone(),
                                            Box::new(try!(untype_expression(expr, context))))
        }
        src::Expression::BinaryOperation(ref bin, ref e1, ref e2) => {
            dst::Expression::BinaryOperation(bin.clone(),
                                             Box::new(try!(untype_expression(e1, context))),
                                             Box::new(try!(untype_expression(e2, context))))
        }
        src::Expression::TernaryConditional(ref e1, ref e2, ref e3) => {
            dst::Expression::TernaryConditional(Box::new(try!(untype_expression(e1, context))),
                                                Box::new(try!(untype_expression(e2, context))),
                                                Box::new(try!(untype_expression(e3, context))))
        }
        src::Expression::Swizzle(ref expr, ref swizzle) => {
            dst::Expression::Swizzle(Box::new(try!(untype_expression(expr, context))),
                                     swizzle.iter()
                                            .map(|swizzle_slot| {
                                                match *swizzle_slot {
                                                    src::SwizzleSlot::X => dst::SwizzleSlot::X,
                                                    src::SwizzleSlot::Y => dst::SwizzleSlot::Y,
                                                    src::SwizzleSlot::Z => dst::SwizzleSlot::Z,
                                                    src::SwizzleSlot::W => dst::SwizzleSlot::W,
                                                }
                                            })
                                            .collect::<Vec<_>>())
        }
        src::Expression::ArraySubscript(ref arr, ref ind) => {
            dst::Expression::ArraySubscript(Box::new(try!(untype_expression(arr, context))),
                                            Box::new(try!(untype_expression(ind, context))))
        }
        src::Expression::Member(ref expr, ref name) => {
            dst::Expression::Member(Box::new(try!(untype_expression(expr, context))),
                                    name.clone())
        }
        src::Expression::Deref(ref expr) => {
            dst::Expression::Deref(Box::new(try!(untype_expression(expr, context))))
        }
        src::Expression::MemberDeref(ref expr, ref name) => {
            dst::Expression::MemberDeref(Box::new(try!(untype_expression(expr, context))),
                                         name.clone())
        }
        src::Expression::AddressOf(ref expr) => {
            dst::Expression::AddressOf(Box::new(try!(untype_expression(expr, context))))
        }
        src::Expression::Call(ref func, ref params) => dst::Expression::Call(
            Box::new(dst::Expression::Variable(try!(context.get_function_name(func)))),
            try!(result_map(untype_expression, params, context))
        ),
        src::Expression::Cast(ref ty, ref expr) => {
            dst::Expression::Cast(try!(untype_type(ty, context)),
                                  Box::new(try!(untype_expression(expr, context))))
        }
        src::Expression::Intrinsic(ref intrinsic) => {
            dst::Expression::Intrinsic(try!(untype_intrinsic(intrinsic, context)))
        }
        src::Expression::UntypedIntrinsic(ref func, ref params) => {
            dst::Expression::Call(Box::new(dst::Expression::Variable(func.clone())),
                                  try!(result_map(untype_expression, params, context)))
        }
        src::Expression::UntypedLiteral(ref name) => dst::Expression::Variable(name.clone()),
    })
}

fn untype_vardef(vd: &src::VarDef, context: &mut Context) -> Result<dst::VarDef, UntyperError> {
    Ok(dst::VarDef {
        name: try!(context.get_local_name(&vd.id)),
        typename: try!(untype_type(&vd.typename, context)),
        assignment: match vd.assignment {
            None => None,
            Some(ref expr) => Some(try!(untype_expression(expr, context))),
        },
    })
}

fn untype_init_expression(member: &src::InitStatement,
                          context: &mut Context)
                          -> Result<dst::InitStatement, UntyperError> {
    Ok(match *member {
        src::InitStatement::Expression(ref expr) => {
            dst::InitStatement::Expression(try!(untype_expression(expr, context)))
        }
        src::InitStatement::Declaration(ref vd) => {
            dst::InitStatement::Declaration(try!(untype_vardef(vd, context)))
        }
    })
}

fn untype_statement(statement: &src::Statement,
                    context: &mut Context)
                    -> Result<dst::Statement, UntyperError> {
    Ok(match *statement {
        src::Statement::Empty => dst::Statement::Empty,
        src::Statement::Expression(ref expr) => {
            dst::Statement::Expression(try!(untype_expression(expr, context)))
        }
        src::Statement::Var(ref vd) => dst::Statement::Var(try!(untype_vardef(vd, context))),
        src::Statement::Block(ref block) => {
            dst::Statement::Block(try!(result_map(untype_statement, block, context)))
        }
        src::Statement::If(ref cond, ref statement) => {
            dst::Statement::If(try!(untype_expression(cond, context)),
                               Box::new(try!(untype_statement(statement, context))))
        }
        src::Statement::For(ref init, ref cond, ref update, ref statement) => {
            dst::Statement::For(try!(untype_init_expression(init, context)),
                                try!(untype_expression(cond, context)),
                                try!(untype_expression(update, context)),
                                Box::new(try!(untype_statement(statement, context))))
        }
        src::Statement::While(ref cond, ref statement) => {
            dst::Statement::While(try!(untype_expression(cond, context)),
                                  Box::new(try!(untype_statement(statement, context))))
        }
        src::Statement::Return(ref expr) => {
            dst::Statement::Return(try!(untype_expression(expr, context)))
        }
    })
}

fn untype_struct_member(member: &src::StructMember,
                        context: &mut Context)
                        -> Result<dst::StructMember, UntyperError> {
    Ok(dst::StructMember {
        name: member.name.clone(),
        typename: try!(untype_type(&member.typename, context)),
    })
}

fn untype_function_param(param: &src::FunctionParam,
                         context: &mut Context)
                         -> Result<dst::FunctionParam, UntyperError> {
    Ok(dst::FunctionParam {
        name: try!(context.get_local_name(&param.id)),
        typename: try!(untype_type(&param.typename, context)),
    })
}

fn untype_kernel_param(param: &src::KernelParam,
                       context: &mut Context)
                       -> Result<dst::KernelParam, UntyperError> {
    Ok(dst::KernelParam {
        name: try!(context.get_local_name(&param.id)),
        typename: try!(untype_type(&param.typename, context)),
    })
}

fn untype_root_definition(root: &src::RootDefinition,
                          context: &mut Context)
                          -> Result<dst::RootDefinition, UntyperError> {
    Ok(match *root {
        src::RootDefinition::GlobalVariable(ref gv) => {
            dst::RootDefinition::GlobalVariable(dst::GlobalVariable {
                name: try!(context.get_global_name(&gv.id)),
                ty: try!(untype_type(&gv.ty, context)),
                address_space: gv.address_space.clone(),
                init: match gv.init {
                    None => None,
                    Some(ref expr) => Some(try!(untype_expression(expr, context))),
                },
            })
        }
        src::RootDefinition::Struct(ref sd) => {
            dst::RootDefinition::Struct(dst::StructDefinition {
                name: try!(context.get_struct_name(&sd.id)),
                members: try!(result_map(untype_struct_member, &sd.members, context)),
            })
        }
        src::RootDefinition::Function(ref fd) => {
            context.push_scope(&fd.local_declarations);
            let f = dst::RootDefinition::Function(dst::FunctionDefinition {
                name: try!(context.get_function_name(&fd.id)),
                returntype: try!(untype_type(&fd.returntype, context)),
                params: try!(result_map(untype_function_param, &fd.params, context)),
                body: try!(result_map(untype_statement, &fd.body, context)),
            });
            context.pop_scope();
            f
        }
        src::RootDefinition::Kernel(ref kernel) => {
            context.push_scope(&kernel.local_declarations);
            let k = dst::RootDefinition::Kernel(dst::Kernel {
                params: try!(result_map(untype_kernel_param, &kernel.params, context)),
                body: try!(result_map(untype_statement, &kernel.body, context)),
                group_dimensions: kernel.group_dimensions.clone(),
            });
            context.pop_scope();
            k
        }
    })
}

pub fn untype_module(module: &src::Module) -> Result<dst::Module, UntyperError> {

    let mut context = try!(Context::from_globals(&module.global_declarations));

    let mut final_defs = vec![];
    let mut fragment_list = module.fragments.keys().collect::<Vec<_>>();
    fragment_list.sort();
    for fragment in fragment_list {
        let id = module.fragments.get(fragment).expect("bad fragment key");
        let name = try!(context.get_function_name(id));
        let gen = fragment.generate(&name);
        final_defs.push(dst::RootDefinition::Function(gen));
    }

    let mut defs = try!(result_map(untype_root_definition,
                                   &module.root_definitions,
                                   &mut context));
    final_defs.append(&mut defs);

    Ok(dst::Module {
        root_definitions: final_defs,
        binds: module.binds.clone(),
    })
}
