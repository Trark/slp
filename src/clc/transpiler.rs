use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use BindMap;
use super::cir as dst;
use super::super::hlsl::ir as src;
use super::super::hlsl::globals_analysis;

#[derive(PartialEq, Debug, Clone)]
pub enum TranspileError {
    Unknown,

    TypeIsNotAllowedAsGlobal(src::GlobalType),
    CouldNotGetEquivalentType(src::TypeLayout),

    GlobalFoundThatIsntInKernelParams(src::GlobalVariable),

    UnknownFunctionId(src::FunctionId),
    UnknownStructId(src::StructId),
    UnknownConstantBufferId(src::ConstantBufferId),
    InvalidVariableRef,
    UnknownVariableId,

    BoolVectorsNotSupported,

    IntrinsicUnimplemented,
}

impl error::Error for TranspileError {
    fn description(&self) -> &str {
        match *self {
            TranspileError::Unknown => "unknown transpiler error",
            TranspileError::TypeIsNotAllowedAsGlobal(_) => "global variable has unsupported type",
            TranspileError::CouldNotGetEquivalentType(_) => "could not find equivalent clc type",
            TranspileError::GlobalFoundThatIsntInKernelParams(_) => "non-parameter global found",
            TranspileError::UnknownFunctionId(_) => "unknown function id",
            TranspileError::UnknownStructId(_) => "unknown struct id",
            TranspileError::UnknownConstantBufferId(_) => "unknown cbuffer id",
            TranspileError::InvalidVariableRef => "invalid variable ref",
            TranspileError::UnknownVariableId => "unknown variable id",
            TranspileError::BoolVectorsNotSupported => "bool vectors not supported",
            TranspileError::IntrinsicUnimplemented => "intrinsic function is not implemented",
        }
    }
}

impl fmt::Display for TranspileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

type KernelParams = Vec<dst::KernelParam>;

/// Tells us if a parameter is converted as is or as a
/// pointer to a value
#[derive(PartialEq, Debug, Clone)]
enum ParamType {
    Normal,
    Pointer,
}

/// A variable name paired with its type
#[derive(PartialEq, Debug, Clone)]
struct VariableDecl(String, ParamType);

impl VariableDecl {
    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum GlobalId {
    Function(src::FunctionId),
    Struct(src::StructId),
    ConstantBufferType(src::ConstantBufferId),
    ConstantBufferInstance(src::ConstantBufferId),
    Variable(src::GlobalId),
}

struct GlobalNameContext {
    pub global_name_map: HashMap<GlobalId, String>,
}

impl GlobalNameContext {
    fn from_globals(globals: &src::GlobalDeclarations) -> Result<GlobalNameContext, TranspileError> {
        let mut context = GlobalNameContext {
            global_name_map: HashMap::new(),
        };

        // Insert global variables
        {
            for (var_id, var_name) in &globals.globals {
                context.insert_identifier(GlobalId::Variable(var_id.clone()), var_name);
            }
        }

        // Insert functions (try to generate [name]_[overload number] for overloads)
        {
            let mut grouped_functions: HashMap<String, Vec<src::FunctionId>> = HashMap::new();
            for (id, name) in globals.functions.iter() {
                match grouped_functions.entry(name.clone()) {
                    Entry::Occupied(mut occupied) => { occupied.get_mut().push(id.clone()); },
                    Entry::Vacant(vacant) => { vacant.insert(vec![id.clone()]); },
                }
            };
            for (key, mut ids) in grouped_functions {
                assert!(ids.len() > 0);
                if ids.len() == 1 {
                    context.insert_identifier(GlobalId::Function(ids[0]), &key);
                } else {
                    ids.sort();
                    for (index, id) in ids.iter().enumerate() {
                        let gen = key.clone() + "_" + &index.to_string();
                        context.insert_identifier(GlobalId::Function(*id), &gen);
                    }
                }
            };
        }


        // Insert structs (name collisions possible with function overloads + globals)
        {
            for (id, struct_name) in globals.structs.iter() {
                context.insert_identifier(GlobalId::Struct(id.clone()), struct_name);
            };
        }

        // Insert cbuffers
        {
            for (id, cbuffer_name) in globals.constants.iter() {
                context.insert_identifier(GlobalId::ConstantBufferInstance(id.clone()), cbuffer_name);
                context.insert_identifier(GlobalId::ConstantBufferType(id.clone()), &(cbuffer_name.clone() + "_t"));
            };
        }

        Ok(context)
    }

    fn is_free(&self, identifier: &str) -> bool {
        for name in self.global_name_map.values() {
            if name == identifier {
                return false;
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

    fn insert_identifier(&mut self, id: GlobalId, name: &str) {
        let identifier = self.make_identifier(name);
        let r = self.global_name_map.insert(id, identifier);
        assert_eq!(r, None);
    }
}

struct Context {
    kernel_params: KernelParams,
    global_names: GlobalNameContext,
    global_lifted_vars: HashMap<src::GlobalId, bool>,
    function_decl_map: HashMap<src::FunctionId, (Vec<ParamType>, Vec<dst::FunctionParam>, Vec<dst::Expression>)>,
    variable_scopes: Vec<HashMap<src::VariableId, VariableDecl>>,
    type_context: src::TypeContext,
}

impl Context {

    fn from_globals(table: &src::GlobalTable, globals: &src::GlobalDeclarations, root_defs: &[src::RootDefinition]) -> Result<(Context, BindMap), TranspileError> {
        let mut context = Context {
            kernel_params: vec![],
            global_names: try!(GlobalNameContext::from_globals(globals)),
            global_lifted_vars: HashMap::new(),
            function_decl_map: HashMap::new(),
            variable_scopes: vec![],
            type_context: match src::TypeContext::from_roots(root_defs) {
                Ok(rt) => rt,
                Err(()) => return Err(TranspileError::Unknown),
            },
        };

        for rootdef in root_defs {
            match rootdef {
                &src::RootDefinition::GlobalVariable(ref gv) => {
                    let &src::GlobalType(src::Type(_, ref modifier), ref gs, _) = &gv.global_type;
                    let static_const = modifier.is_const && *gs == src::GlobalStorage::Static;
                    context.global_lifted_vars.insert(gv.id, !static_const);
                },
                _ => { },
            }
        }

        let usage = globals_analysis::GlobalUsage::analyse(root_defs);

        let mut binds = BindMap::new();

        // Create list of kernel parameters
        let mut kernel_param_ids = vec![];
        enum GlobalKey { Global(src::GlobalId), Constant(src::ConstantBufferId) };
        {
            // Ensure a stable order (for easier testing + repeatability)
            let mut c_keys = table.constants.keys().collect::<Vec<&u32>>();
            c_keys.sort();
            for slot in c_keys {
                let id = table.constants.get(slot).unwrap_or_else(|| panic!("bad key"));
                let cl_type = dst::Type::Pointer(
                    dst::AddressSpace::Constant,
                    Box::new(dst::Type::Struct(try!(context.get_cbuffer_struct_name(id))))
                );
                let cl_var = try!(context.get_cbuffer_instance_name(id));
                let param = dst::KernelParam {
                    name: cl_var,
                    typename: cl_type,
                };
                let entry = binds.cbuffer_map.insert(*slot, context.kernel_params.len() as u32);
                assert!(entry.is_none());
                context.kernel_params.push(param);
                kernel_param_ids.push(GlobalKey::Constant(id.clone()));
            }
            let mut r_keys = table.r_resources.keys().collect::<Vec<&u32>>();
            r_keys.sort();
            for slot in r_keys {
                let global_entry = table.r_resources.get(slot).unwrap_or_else(|| panic!("bad key"));
                let &src::Type(ref tyl, _) = &global_entry.ty.0;
                let cl_type = match tyl {
                    &src::TypeLayout::Object(src::ObjectType::Buffer(ref data_type)) => {
                        dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_datatype(data_type, &context))))
                    }
                    &src::TypeLayout::Object(src::ObjectType::StructuredBuffer(ref structured_type)) => {
                        dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_structuredtype(structured_type, &context))))
                    }
                    _ => return Err(TranspileError::TypeIsNotAllowedAsGlobal(global_entry.ty.clone())),
                };
                let param = dst::KernelParam {
                    name: try!(context.get_global_name(&global_entry.id)).0,
                    typename: cl_type,
                };
                let entry = binds.read_map.insert(*slot, context.kernel_params.len() as u32);
                assert!(entry.is_none());
                context.kernel_params.push(param);
                kernel_param_ids.push(GlobalKey::Global(global_entry.id));
            }
            let mut rw_keys = table.rw_resources.keys().collect::<Vec<&u32>>();
            rw_keys.sort();
            for slot in rw_keys {
                let global_entry = table.rw_resources.get(slot).unwrap_or_else(|| panic!("bad key"));
                let &src::Type(ref tyl, _) = &global_entry.ty.0;
                let cl_type = match tyl {
                    &src::TypeLayout::Object(src::ObjectType::RWBuffer(ref data_type)) => {
                        dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_datatype(data_type, &context))))
                    }
                    &src::TypeLayout::Object(src::ObjectType::RWStructuredBuffer(ref structured_type)) => {
                        dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_structuredtype(structured_type, &context))))
                    }
                    &src::TypeLayout::Object(src::ObjectType::RWTexture2D(_)) => {
                        let read = usage.image_reads.contains(&global_entry.id);
                        let write = usage.image_writes.contains(&global_entry.id);
                        let access = match (read, write) {
                            (false, false) | (true, false) => dst::AccessModifier::ReadOnly,
                            (false, true) => dst::AccessModifier::WriteOnly,
                            (true, true) => dst::AccessModifier::ReadWrite, // OpenCL 2.0 only
                        };
                        dst::Type::Image2D(access)
                    }
                    _ => return Err(TranspileError::TypeIsNotAllowedAsGlobal(global_entry.ty.clone())),
                };
                let param = dst::KernelParam {
                    name: try!(context.get_global_name(&global_entry.id)).0,
                    typename: cl_type,
                };
                let entry = binds.write_map.insert(*slot, context.kernel_params.len() as u32);
                assert!(entry.is_none());
                context.kernel_params.push(param);
                kernel_param_ids.push(GlobalKey::Global(global_entry.id));
            }
        }

        for rootdef in root_defs {
            match rootdef {
                &src::RootDefinition::Function(ref func) => {
                    let param_types = func.params.iter().fold(vec![],
                        |mut param_types, param| {
                            match param.param_type.1 {
                                src::InputModifier::InOut | src::InputModifier::Out => param_types.push(ParamType::Pointer),
                                src::InputModifier::In => param_types.push(ParamType::Normal),
                            };
                            param_types
                        }
                    );
                    let function_global_usage = usage.functions.get(&func.id).unwrap_or_else(|| panic!("analysis missing function"));

                    let mut global_params = vec![];
                    let mut global_args = vec![];
                    for (ref key, ref kernel_param) in kernel_param_ids.iter().zip(context.kernel_params.iter()) {
                        let used = match *key {
                            &GlobalKey::Global(ref global) => function_global_usage.globals.contains(&global),
                            &GlobalKey::Constant(ref cb) => function_global_usage.cbuffers.contains(&cb)
                        };
                        if used {
                            global_params.push(dst::FunctionParam { name: kernel_param.name.clone(), typename: kernel_param.typename.clone() });
                            global_args.push(dst::Expression::Variable(kernel_param.name.clone()));
                        }
                    };

                    let ret = context.function_decl_map.insert(func.id.clone(), (param_types, global_params, global_args));
                    assert_eq!(ret, None);
                },
                _ => { },
            }
        }

        Ok((context, binds))
    }

    fn get_function(&self, id: &src::FunctionId) -> Result<(dst::Expression, Vec<ParamType>, Vec<dst::FunctionParam>, Vec<dst::Expression>), TranspileError> {
        let name = try!(self.get_function_name(id));
        match self.function_decl_map.get(id) {
            Some(&(ref decl, ref params, ref args)) => Ok((dst::Expression::Variable(name), decl.clone(), params.clone(), args.clone())),
            None => panic!("Function not defined"), // Name exists but decl doesn't
        }
    }

    fn get_function_name(&self, id: &src::FunctionId) -> Result<String, TranspileError> {
        match self.global_names.global_name_map.get(&GlobalId::Function(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownFunctionId(id.clone())),
        }
    }

    /// Get the expression to access an in scope variable
    fn get_variable_ref(&self, var_ref: &src::VariableRef) -> Result<dst::Expression, TranspileError> {
        let scopes_up = (var_ref.1).0 as usize;
        if scopes_up >= self.variable_scopes.len() {
            return Err(TranspileError::UnknownVariableId)
        } else {
            let scope = self.variable_scopes.len() - scopes_up - 1;
            match self.variable_scopes[scope].get(&var_ref.0) {
                Some(&VariableDecl(ref s, ref pt)) => Ok(match *pt {
                    ParamType::Normal => dst::Expression::Variable(s.clone()),
                    ParamType::Pointer => dst::Expression::Deref(Box::new(dst::Expression::Variable(s.clone())))
                }),
                None => Err(TranspileError::UnknownVariableId),
            }
        }
    }

    /// Get the name of a variable declared in the current block
    fn get_variable_id(&self, id: &src::VariableId) -> Result<String, TranspileError> {
        assert!(self.variable_scopes.len() > 0);
        match self.variable_scopes[self.variable_scopes.len() - 1].get(id) {
            Some(v) => Ok(v.as_str().to_string()),
            None => Err(TranspileError::UnknownVariableId),
        }
    }

    /// Get the name of a variable declared in the current block
    fn get_global_var(&self, id: &src::GlobalId) -> Result<dst::Expression, TranspileError> {
        let (global_name, _) = try!(self.get_global_name(id));
        Ok(dst::Expression::Variable(global_name))
    }

    /// Get the name of a variable declared in the current block
    fn get_global_name(&self, id: &src::GlobalId) -> Result<(String, bool), TranspileError> {
        match self.global_names.global_name_map.get(&GlobalId::Variable(*id)) {
            Some(v) => match self.global_lifted_vars.get(id) {
                Some(lifted) => Ok((v.to_string(), *lifted)),
                None => panic!(),
            },
            None => Err(TranspileError::UnknownVariableId),
        }
    }

    /// Get the name of a struct
    fn get_struct_name(&self, id: &src::StructId) -> Result<String, TranspileError> {
        match self.global_names.global_name_map.get(&GlobalId::Struct(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownStructId(id.clone())),
        }
    }

    /// Get the name of the struct used for a cbuffer
    fn get_cbuffer_struct_name(&self, id: &src::ConstantBufferId) -> Result<String, TranspileError> {
        match self.global_names.global_name_map.get(&GlobalId::ConstantBufferType(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
        }
    }

    /// Get the name of the cbuffer instance
    fn get_cbuffer_instance_name(&self, id: &src::ConstantBufferId) -> Result<String, TranspileError> {
        match self.global_names.global_name_map.get(&GlobalId::ConstantBufferInstance(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
        }
    }

    /// Get the expression to find the given constant
    fn get_constant(&self, id: &src::ConstantBufferId, name: String) -> Result<dst::Expression, TranspileError> {
        Ok(dst::Expression::MemberDeref(
            Box::new(dst::Expression::Variable(
                try!(self.get_cbuffer_instance_name(id))
            )),
            name
        ))
    }

    fn is_free(&self, identifier: &str) -> bool {
        if !self.global_names.is_free(identifier) {
            return false;
        }
        for scope in &self.variable_scopes {
            for var in scope.values() {
                if var.as_str() == identifier {
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

    fn push_scope(&mut self, scope_block: &src::ScopeBlock) {
        self.push_scope_with_pointer_overrides(scope_block, &[])
    }

    fn push_scope_with_pointer_overrides(&mut self, scope_block: &src::ScopeBlock, pointers: &[src::VariableId]) {
        self.variable_scopes.push(HashMap::new());
        for (var_id, var_name) in &scope_block.1.variables {
            let identifier = self.make_identifier(&var_name);
            let map = self.variable_scopes.last_mut().unwrap();
            map.insert(var_id.clone(), VariableDecl(identifier, if pointers.iter().any(|pp| pp == var_id) { ParamType::Pointer } else { ParamType::Normal }));
        };
        self.type_context.push_scope(scope_block);
    }

    fn pop_scope(&mut self) {
        assert!(self.variable_scopes.len() > 0);
        self.variable_scopes.pop();
        self.type_context.pop_scope();
    }
}

fn transpile_scalartype(scalartype: &src::ScalarType) -> Result<dst::Scalar, TranspileError> {
    match scalartype {
        &src::ScalarType::Bool => Err(TranspileError::BoolVectorsNotSupported),
        &src::ScalarType::Int => Ok(dst::Scalar::Int),
        &src::ScalarType::UInt => Ok(dst::Scalar::UInt),
        &src::ScalarType::Half => Ok(dst::Scalar::Half),
        &src::ScalarType::Float => Ok(dst::Scalar::Float),
        &src::ScalarType::Double => Ok(dst::Scalar::Double),
        &src::ScalarType::UntypedInt => panic!(),
    }
}

fn transpile_datatype(datatype: &src::DataType, context: &Context) -> Result<dst::Type, TranspileError> {
    transpile_type(&src::Type::from(datatype.clone()), context)
}

fn transpile_structuredtype(structured_type: &src::StructuredType, context: &Context) -> Result<dst::Type, TranspileError> {
    transpile_type(&src::Type::from(structured_type.clone()), context)
}

fn transpile_typelayout(ty: &src::TypeLayout, context: &Context) -> Result<dst::Type, TranspileError> {
    match ty {
        &src::TypeLayout::Void => Ok(dst::Type::Void),
        &src::TypeLayout::Scalar(src::ScalarType::Bool) => Ok(dst::Type::Bool),
        &src::TypeLayout::Scalar(ref scalar) => Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar)))),
        &src::TypeLayout::Vector(ref scalar, 1) => Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar)))),
        &src::TypeLayout::Vector(ref scalar, 2) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Two)),
        &src::TypeLayout::Vector(ref scalar, 3) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Three)),
        &src::TypeLayout::Vector(ref scalar, 4) => Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)), dst::VectorDimension::Four)),
        &src::TypeLayout::Struct(ref id) => {
            let struct_name = try!(context.get_struct_name(id));
            Ok(dst::Type::Struct(struct_name))
        },
        &src::TypeLayout::Array(ref element, ref dim) => {
            let inner_ty = try!(transpile_typelayout(element, context));
            Ok(dst::Type::Array(Box::new(inner_ty), *dim))
        },
        _ => return Err(TranspileError::CouldNotGetEquivalentType(ty.clone())),
    }
}

fn transpile_type(hlsltype: &src::Type, context: &Context) -> Result<dst::Type, TranspileError> {
    let &src::Type(ref ty, _) = hlsltype;
    transpile_typelayout(ty, context)
}

fn transpile_localtype(local_type: &src::LocalType, context: &Context) -> Result<dst::Type, TranspileError> {
    let &src::LocalType(ref ty, ref ls, ref modifiers) = local_type;
    match *modifiers {
        Some(_) => return Err(TranspileError::Unknown),
        None => { },
    };
    match *ls {
        src::LocalStorage::Local => transpile_type(ty, context),
        src::LocalStorage::Static => Err(TranspileError::Unknown),
    }
}

fn transpile_unaryop(unaryop: &src::UnaryOp) -> Result<dst::UnaryOp, TranspileError> {
    match *unaryop {
        src::UnaryOp::PrefixIncrement => Ok(dst::UnaryOp::PrefixIncrement),
        src::UnaryOp::PrefixDecrement => Ok(dst::UnaryOp::PrefixDecrement),
        src::UnaryOp::PostfixIncrement => Ok(dst::UnaryOp::PostfixIncrement),
        src::UnaryOp::PostfixDecrement => Ok(dst::UnaryOp::PostfixDecrement),
        src::UnaryOp::Plus => Ok(dst::UnaryOp::Plus),
        src::UnaryOp::Minus => Ok(dst::UnaryOp::Minus),
        src::UnaryOp::LogicalNot => Ok(dst::UnaryOp::LogicalNot),
        src::UnaryOp::BitwiseNot => Ok(dst::UnaryOp::BitwiseNot),
    }
}

fn transpile_binop(binop: &src::BinOp) -> Result<dst::BinOp, TranspileError> {
    match *binop {
        src::BinOp::Add => Ok(dst::BinOp::Add),
        src::BinOp::Subtract => Ok(dst::BinOp::Subtract),
        src::BinOp::Multiply => Ok(dst::BinOp::Multiply),
        src::BinOp::Divide => Ok(dst::BinOp::Divide),
        src::BinOp::Modulus => Ok(dst::BinOp::Modulus),
        src::BinOp::LeftShift => Ok(dst::BinOp::LeftShift),
        src::BinOp::RightShift => Ok(dst::BinOp::RightShift),
        src::BinOp::LessThan => Ok(dst::BinOp::LessThan),
        src::BinOp::LessEqual => Ok(dst::BinOp::LessEqual),
        src::BinOp::GreaterThan => Ok(dst::BinOp::GreaterThan),
        src::BinOp::GreaterEqual => Ok(dst::BinOp::GreaterEqual),
        src::BinOp::Equality => Ok(dst::BinOp::Equality),
        src::BinOp::Inequality => Ok(dst::BinOp::Inequality),
        src::BinOp::Assignment => Ok(dst::BinOp::Assignment),
    }
}

fn transpile_literal(lit: &src::Literal) -> Result<dst::Literal, TranspileError> {
    match lit {
        &src::Literal::Bool(b) => Ok(dst::Literal::Bool(b)),
        &src::Literal::UntypedInt(i) => Ok(dst::Literal::Int(i)),
        &src::Literal::Int(i) => Ok(dst::Literal::Int(i)),
        &src::Literal::UInt(i) => Ok(dst::Literal::UInt(i)),
        &src::Literal::Long(i) => Ok(dst::Literal::Long(i)),
        &src::Literal::Half(f) => Ok(dst::Literal::Half(f)),
        &src::Literal::Float(f) => Ok(dst::Literal::Float(f)),
        &src::Literal::Double(f) => Ok(dst::Literal::Double(f)),
    }
}

fn write_func(name: &'static str, args: &[&src::Expression], context: &Context) -> Result<dst::Expression, TranspileError> {
    Ok(dst::Expression::Call(
        Box::new(dst::Expression::Variable(name.to_string())),
        try!(args.iter().fold(Ok(vec![]), |result, exp| {
            let mut vec = try!(result);
            vec.push(try!(transpile_expression(exp, context)));
            Ok(vec)
        }))
    ))
}

fn transpile_intrinsic(intrinsic: &src::Intrinsic, context: &Context) -> Result<dst::Expression, TranspileError> {
    match *intrinsic {
        src::Intrinsic::AllMemoryBarrier | src::Intrinsic::AllMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::Call(
                Box::new(dst::Expression::Variable("barrier".to_string())),
                vec![dst::Expression::BinaryOperation(dst::BinOp::BitwiseOr,
                    Box::new(dst::Expression::Variable("CLK_LOCAL_MEM_FENCE".to_string())),
                    Box::new(dst::Expression::Variable("CLK_GLOBAL_MEM_FENCE".to_string()))
                )]
            ))
        },
        src::Intrinsic::DeviceMemoryBarrier | src::Intrinsic::DeviceMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::Call(
                Box::new(dst::Expression::Variable("barrier".to_string())),
                vec![dst::Expression::Variable("CLK_GLOBAL_MEM_FENCE".to_string())]
            ))
        },
        src::Intrinsic::GroupMemoryBarrier | src::Intrinsic::GroupMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::Call(
                Box::new(dst::Expression::Variable("barrier".to_string())),
                vec![dst::Expression::Variable("CLK_LOCAL_MEM_FENCE".to_string())]
            ))
        },
        src::Intrinsic::AsIntU(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntU1(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntU2(ref e) => write_func("as_int2", &[e], context),
        src::Intrinsic::AsIntU3(ref e) => write_func("as_int3", &[e], context),
        src::Intrinsic::AsIntU4(ref e) => write_func("as_int4", &[e], context),
        src::Intrinsic::AsIntF(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntF1(ref e) => write_func("as_int", &[e], context),
        src::Intrinsic::AsIntF2(ref e) => write_func("as_int2", &[e], context),
        src::Intrinsic::AsIntF3(ref e) => write_func("as_int3", &[e], context),
        src::Intrinsic::AsIntF4(ref e) => write_func("as_int4", &[e], context),
        src::Intrinsic::AsUIntI(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntI1(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntI2(ref e) => write_func("as_uint2", &[e], context),
        src::Intrinsic::AsUIntI3(ref e) => write_func("as_uint3", &[e], context),
        src::Intrinsic::AsUIntI4(ref e) => write_func("as_uint4", &[e], context),
        src::Intrinsic::AsUIntF(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntF1(ref e) => write_func("as_uint", &[e], context),
        src::Intrinsic::AsUIntF2(ref e) => write_func("as_uint2", &[e], context),
        src::Intrinsic::AsUIntF3(ref e) => write_func("as_uint3", &[e], context),
        src::Intrinsic::AsUIntF4(ref e) => write_func("as_uint4", &[e], context),
        src::Intrinsic::AsFloatI(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatI1(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatI2(ref e) => write_func("as_float2", &[e], context),
        src::Intrinsic::AsFloatI3(ref e) => write_func("as_float3", &[e], context),
        src::Intrinsic::AsFloatI4(ref e) => write_func("as_float4", &[e], context),
        src::Intrinsic::AsFloatU(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatU1(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatU2(ref e) => write_func("as_float2", &[e], context),
        src::Intrinsic::AsFloatU3(ref e) => write_func("as_float3", &[e], context),
        src::Intrinsic::AsFloatU4(ref e) => write_func("as_float4", &[e], context),
        src::Intrinsic::AsFloatF(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatF1(ref e) => write_func("as_float", &[e], context),
        src::Intrinsic::AsFloatF2(ref e) => write_func("as_float2", &[e], context),
        src::Intrinsic::AsFloatF3(ref e) => write_func("as_float3", &[e], context),
        src::Intrinsic::AsFloatF4(ref e) => write_func("as_float4", &[e], context),
        src::Intrinsic::ClampI(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI1(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI2(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI3(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampI4(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF1(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF2(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF3(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::ClampF4(ref x, ref min, ref max) => write_func("clamp", &[x, min, max], context),
        src::Intrinsic::Cross(ref x, ref y) => write_func("cross", &[x, y], context),
        src::Intrinsic::Distance1(ref x, ref y) |
        src::Intrinsic::Distance2(ref x, ref y) |
        src::Intrinsic::Distance3(ref x, ref y) |
        src::Intrinsic::Distance4(ref x, ref y) => {
            Ok(dst::Expression::Call(
                Box::new(dst::Expression::Variable("length".to_string())),
                vec![dst::Expression::BinaryOperation(dst::BinOp::Subtract,
                    Box::new(try!(transpile_expression(x, context))),
                    Box::new(try!(transpile_expression(y, context)))
                )]
            ))
        },
        src::Intrinsic::DotI1(_, _) => Err(TranspileError::IntrinsicUnimplemented),
        src::Intrinsic::DotI2(_, _) => Err(TranspileError::IntrinsicUnimplemented),
        src::Intrinsic::DotI3(_, _) => Err(TranspileError::IntrinsicUnimplemented),
        src::Intrinsic::DotI4(_, _) => Err(TranspileError::IntrinsicUnimplemented),
        src::Intrinsic::DotF1(ref x, ref y) => write_func("dot", &[x, y], context),
        src::Intrinsic::DotF2(ref x, ref y) => write_func("dot", &[x, y], context),
        src::Intrinsic::DotF3(ref x, ref y) => write_func("dot", &[x, y], context),
        src::Intrinsic::DotF4(ref x, ref y) => write_func("dot", &[x, y], context),
        src::Intrinsic::Float4(ref x, ref y, ref z, ref w) => {
            Ok(dst::Expression::Constructor(dst::Constructor::Float4(
                Box::new(try!(transpile_expression(x, context))),
                Box::new(try!(transpile_expression(y, context))),
                Box::new(try!(transpile_expression(z, context))),
                Box::new(try!(transpile_expression(w, context)))
            )))
        },
        src::Intrinsic::BufferLoad(ref buffer, ref loc) |
        src::Intrinsic::RWBufferLoad(ref buffer, ref loc) |
        src::Intrinsic::StructuredBufferLoad(ref buffer, ref loc) |
        src::Intrinsic::RWStructuredBufferLoad(ref buffer, ref loc) => {
            let cl_buffer = Box::new(try!(transpile_expression(buffer, context)));
            let cl_loc = Box::new(try!(transpile_expression(loc, context)));
            Ok(dst::Expression::ArraySubscript(cl_buffer, cl_loc))
        },
        src::Intrinsic::RWTexture2DLoad(ref tex, ref loc) => {
            let cl_tex = try!(transpile_expression(tex, context));
            let cl_loc = try!(transpile_expression(loc, context));
            let ty = match context.type_context.get_expression_type(tex) {
                Ok(ty) => ty,
                Err(()) => return Err(TranspileError::Unknown),
            };
            let func_name = match ty {
                src::ExpressionType(src::Type(src::TypeLayout::Object(src::ObjectType::RWTexture2D(ref data_type)), _), _) => {
                    match data_type.0 {
                        src::DataLayout::Scalar(ref scalar) | src::DataLayout::Vector(ref scalar, _) => {
                            match *scalar {
                                src::ScalarType::Int => "read_imagei",
                                src::ScalarType::UInt => "read_imageui",
                                src::ScalarType::Float => "read_imagef",
                                _ => return Err(TranspileError::Unknown),
                            }
                        },
                        src::DataLayout::Matrix(_, _, _) => return Err(TranspileError::Unknown),
                    }
                },
                _ => return Err(TranspileError::Unknown),
            };
            // Todo: Swizzle down
            Ok(dst::Expression::Call(
                Box::new(dst::Expression::Variable(func_name.to_string())),
                vec![cl_tex, cl_loc]
            ))
        },
        _ => unimplemented!(),
    }
}

fn transpile_expression(expression: &src::Expression, context: &Context) -> Result<dst::Expression, TranspileError> {
    match expression {
        &src::Expression::Literal(ref lit) => Ok(dst::Expression::Literal(try!(transpile_literal(lit)))),
        &src::Expression::Variable(ref var_ref) => context.get_variable_ref(var_ref),
        &src::Expression::Global(ref id) => context.get_global_var(id),
        &src::Expression::ConstantVariable(ref id, ref name) => context.get_constant(id, name.clone()),
        &src::Expression::UnaryOperation(ref unaryop, ref expr) => {
            let cl_unaryop = try!(transpile_unaryop(unaryop));
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::UnaryOperation(cl_unaryop, cl_expr))
        }
        &src::Expression::BinaryOperation(ref binop, ref lhs, ref rhs) => {
            let cl_binop = try!(transpile_binop(binop));
            let cl_lhs = Box::new(try!(transpile_expression(lhs, context)));
            let cl_rhs = Box::new(try!(transpile_expression(rhs, context)));
            Ok(dst::Expression::BinaryOperation(cl_binop, cl_lhs, cl_rhs))
        }
        &src::Expression::TernaryConditional(ref cond, ref lhs, ref rhs) => {
            let cl_cond = Box::new(try!(transpile_expression(cond, context)));
            let cl_lhs = Box::new(try!(transpile_expression(lhs, context)));
            let cl_rhs = Box::new(try!(transpile_expression(rhs, context)));
            Ok(dst::Expression::TernaryConditional(cl_cond, cl_lhs, cl_rhs))
        }
        &src::Expression::ArraySubscript(ref expr, ref sub) => {
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            let cl_sub = Box::new(try!(transpile_expression(sub, context)));
            Ok(dst::Expression::ArraySubscript(cl_expr, cl_sub))
        },
        &src::Expression::Member(ref expr, ref member_name) => {
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::Member(cl_expr, member_name.clone()))
        },
        &src::Expression::Call(ref func_id, ref params) => {
            let (func_expr, pts, _, args) = try!(context.get_function(func_id));
            assert_eq!(params.len(), pts.len());
            let mut params_exprs: Vec<dst::Expression> = vec![];
            for (param, pt) in params.iter().zip(pts) {
                let param_expr = try!(transpile_expression(param, context));
                let param_expr = match pt {
                    ParamType::Normal => param_expr,
                    ParamType::Pointer => dst::Expression::AddressOf(Box::new(param_expr)),
                };
                params_exprs.push(param_expr);
            };
            let mut final_arguments = args;
            final_arguments.append(&mut params_exprs);
            Ok(dst::Expression::Call(Box::new(func_expr), final_arguments))
        },
        &src::Expression::Cast(ref cast_type, ref expr) => {
            let cl_type = try!(transpile_type(cast_type, context));
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::Cast(cl_type, cl_expr))
        },
        &src::Expression::Intrinsic(ref intrinsic) => transpile_intrinsic(intrinsic, context),
    }
}

fn transpile_vardef(vardef: &src::VarDef, context: &Context) -> Result<dst::VarDef, TranspileError> {
    Ok(dst::VarDef {
        name: try!(context.get_variable_id(&vardef.id)),
        typename: try!(transpile_localtype(&vardef.local_type, context)),
        assignment: match &vardef.assignment { &None => None, &Some(ref expr) => Some(try!(transpile_expression(expr, context))) },
    })
}

#[allow(dead_code)]
fn transpile_condition(cond: &src::Condition, context: &Context) -> Result<dst::Condition, TranspileError> {
    match *cond {
        src::Condition::Expr(ref expr) => {
            let expr_ir = try!(transpile_expression(expr, &context));
            Ok(dst::Condition::Expr(expr_ir))
        },
        src::Condition::Assignment(ref vd) => {
            let cl_vardef = try!(transpile_vardef(vd, &context));
            Ok(dst::Condition::Assignment(cl_vardef))
        },
    }
}

fn transpile_statement(statement: &src::Statement, context: &mut Context) -> Result<dst::Statement, TranspileError> {
    match statement {
        &src::Statement::Expression(ref expr) => Ok(dst::Statement::Expression(try!(transpile_expression(expr, context)))),
        &src::Statement::Var(ref vd) => Ok(dst::Statement::Var(try!(transpile_vardef(vd, context)))),
        &src::Statement::Block(ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::Block(cl_statements))
        },
        &src::Statement::If(ref cond, ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::If(cl_cond, Box::new(dst::Statement::Block(cl_statements))))
        },
        &src::Statement::For(ref init, ref cond, ref update, ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_init = try!(transpile_condition(init, context));
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_update = try!(transpile_expression(update, context));
            let cl_statements= try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::For(cl_init, cl_cond, cl_update, Box::new(dst::Statement::Block(cl_statements))))
        },
        &src::Statement::While(ref cond, ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(dst::Statement::While(cl_cond, Box::new(dst::Statement::Block(cl_statements))))
        },
        &src::Statement::Return(_) => unimplemented!(),
    }
}

fn transpile_statements(statements: &[src::Statement], context: &mut Context) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_statements = vec![];
    for statement in statements {
        cl_statements.push(try!(transpile_statement(statement, context)));
    }
    Ok(cl_statements)
}

fn transpile_param(param: &src::FunctionParam, context: &Context) -> Result<dst::FunctionParam, TranspileError> {
    let &src::ParamType(ref ty_ast, ref it, ref interp) = &param.param_type;
    let ty = match *it {
        src::InputModifier::In => try!(transpile_type(ty_ast, context)),
        src::InputModifier::Out | src::InputModifier::InOut => {
            // Only allow out params to work on Private address space
            // as we don't support generating multiple function for each
            // address space (and can't use __generic as it's 2.0)
            dst::Type::Pointer(dst::AddressSpace::Private, Box::new(try!(transpile_type(ty_ast, context))))
        },
    };
    match *interp {
        Some(_) => return Err(TranspileError::Unknown),
        None => { },
    };
    Ok(dst::FunctionParam {
        name: try!(context.get_variable_id(&param.id)),
        typename: ty,
    })
}

fn transpile_params(params: &[src::FunctionParam], context: &Context) -> Result<Vec<dst::FunctionParam>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_param(param, context)));
    }
    Ok(cl_params)
}

fn transpile_kernel_input_semantic(param: &src::KernelParam, context: &Context) -> Result<dst::Statement, TranspileError> {
    match &param.1 {
        &src::KernelSemantic::DispatchThreadId => {
            let typename = try!(transpile_type(&param.1.get_type(), context));
            let assign = match &param.1 {
                &src::KernelSemantic::DispatchThreadId => {
                    let x = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(Box::new(dst::Expression::Literal(dst::Literal::UInt(0)))));
                    let y = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(Box::new(dst::Expression::Literal(dst::Literal::UInt(1)))));
                    let z = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(Box::new(dst::Expression::Literal(dst::Literal::UInt(2)))));
                    dst::Expression::Constructor(dst::Constructor::UInt3(Box::new(x), Box::new(y), Box::new(z)))
                },
                _ => unimplemented!(),
            };
            Ok(dst::Statement::Var(dst::VarDef {
                name: try!(context.get_variable_id(&param.0)),
                typename: typename,
                assignment: Some(assign),
            }))
        },
        &src::KernelSemantic::GroupId => unimplemented!(),
        &src::KernelSemantic::GroupIndex => unimplemented!(),
        &src::KernelSemantic::GroupThreadId => unimplemented!(),
    }
}

fn transpile_kernel_input_semantics(params: &[src::KernelParam], context: &Context) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_kernel_input_semantic(param, context)));
    }
    Ok(cl_params)
}

fn transpile_structdefinition(structdefinition: &src::StructDefinition, context: &mut Context) -> Result<dst::RootDefinition, TranspileError> {
    Ok(dst::RootDefinition::Struct(dst::StructDefinition {
        name: try!(context.get_struct_name(&structdefinition.id)),
        members: try!(structdefinition.members.iter().fold(
            Ok(vec![]),
            |result, member| {
                let mut vec = try!(result);
                let dst_member = dst::StructMember {
                    name: member.name.clone(),
                    typename: try!(transpile_type(&member.typename, context)),
                };
                vec.push(dst_member);
                Ok(vec)
            }
        )),
    }))
}

fn transpile_cbuffer(cb: &src::ConstantBuffer, context: &mut Context) -> Result<dst::RootDefinition, TranspileError> {
    let mut members = vec![];
    for member in &cb.members {
        let var_name = member.name.clone();
        let var_type = try!(transpile_type(&member.typename, context));
        members.push(dst::StructMember {
            name: var_name,
            typename: var_type,
        });
    };
    Ok(dst::RootDefinition::Struct(dst::StructDefinition {
        name: try!(context.get_cbuffer_struct_name(&cb.id)),
        members: members,
    }))
}

fn transpile_globalvariable(gv: &src::GlobalVariable, context: &mut Context) -> Result<Option<dst::RootDefinition>, TranspileError> {
    let (global_name, lifted) = try!(context.get_global_name(&gv.id));
    assert_eq!(lifted, context.kernel_params.iter().any(|gp| { gp.name == global_name }));
    if lifted {
        return Ok(None)
    } else {
        let &src::GlobalType(src::Type(ref ty, ref modifiers), ref gs, _) = &gv.global_type;
        if *gs == src::GlobalStorage::Static && modifiers.is_const {
            let cl_type = try!(transpile_type(&src::Type(ty.clone(), modifiers.clone()), &context));
            let cl_init = match &gv.assignment {
                &Some(ref expr) => Some(try!(transpile_expression(expr, &context))),
                &None => None,
            };
            Ok(Some(dst::RootDefinition::GlobalVariable(dst::GlobalVariable {
                name: global_name,
                ty: cl_type,
                address_space: dst::AddressSpace::Constant,
                init: cl_init,
            })))
        } else {
            return Err(TranspileError::GlobalFoundThatIsntInKernelParams(gv.clone()))
        }
    }
}

fn transpile_functiondefinition(func: &src::FunctionDefinition, context: &mut Context) -> Result<dst::RootDefinition, TranspileError> {
    // Find the parameters that need to be turned into pointers
    // so the context can deref them
    let out_params = func.params.iter().fold(vec![],
        |mut out_params, param| {
            match param.param_type.1 {
                src::InputModifier::InOut | src::InputModifier::Out => {
                    out_params.push(param.id);
                },
                src::InputModifier::In => { },
            };
            out_params
        }
    );
    context.push_scope_with_pointer_overrides(&func.scope_block, &out_params);
    let (_, _, global_params, _) = try!(context.get_function(&func.id));
    let mut params = global_params;
    params.append(&mut try!(transpile_params(&func.params, context)));
    let body = try!(transpile_statements(&func.scope_block.0, context));
    context.pop_scope();
    let cl_func = dst::FunctionDefinition {
        name: try!(context.get_function_name(&func.id)),
        returntype: try!(transpile_type(&func.returntype, context)),
        params: params,
        body: body,
    };
    Ok(dst::RootDefinition::Function(cl_func))
}

fn transpile_kernel(kernel: &src::Kernel, context: &mut Context) -> Result<dst::RootDefinition, TranspileError> {
    context.push_scope(&kernel.scope_block);
    let mut body = try!(transpile_kernel_input_semantics(&kernel.params, context));
    let mut main_body = try!(transpile_statements(&kernel.scope_block.0, context));
    context.pop_scope();
    body.append(&mut main_body);
    let cl_kernel = dst::Kernel {
        params: context.kernel_params.clone(),
        body: body,
        group_dimensions: dst::Dimension(kernel.group_dimensions.0, kernel.group_dimensions.1, kernel.group_dimensions.2),
    };
    Ok(dst::RootDefinition::Kernel(cl_kernel))
}

fn transpile_roots(root_defs: &[src::RootDefinition], context: &mut Context) -> Result<Vec<dst::RootDefinition>, TranspileError> {
    let mut cl_defs = vec![];

    for rootdef in root_defs {
        match *rootdef {
            src::RootDefinition::Struct(ref structdefinition) => {
                cl_defs.push(try!(transpile_structdefinition(structdefinition, context)));
            },
            src::RootDefinition::ConstantBuffer(ref cb) => {
                cl_defs.push(try!(transpile_cbuffer(cb, context)));
            },
            src::RootDefinition::GlobalVariable(ref gv) => {
                match try!(transpile_globalvariable(gv, context)) {
                    Some(root) => cl_defs.push(root),
                    None => { },
                }
            },
            src::RootDefinition::SamplerState => unimplemented!(),
            _ => { },
        };
    }

    for rootdef in root_defs {
        match *rootdef {
            src::RootDefinition::Function(ref func) => {
                cl_defs.push(try!(transpile_functiondefinition(func, context)));
            },
            src::RootDefinition::Kernel(ref kernel) => {
                cl_defs.push(try!(transpile_kernel(kernel, context)));
            },
            _ => { },
        };
    }

    Ok(cl_defs)
}

pub fn transpile(module: &src::Module) -> Result<dst::Module, TranspileError> {

    let (mut context, binds) = try!(Context::from_globals(&module.global_table, &module.global_declarations, &module.root_definitions));

    let cl_defs = try!(transpile_roots(&module.root_definitions, &mut context));

    let cl_module = dst::Module {
        root_definitions: cl_defs,
        binds: binds,
    };

    Ok(cl_module)
}

#[test]
fn test_transpile() {

    use super::super::hlsl;
    use super::super::hlsl::typer::typeparse;

    let module = hlsl::ast::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            hlsl::ast::RootDefinition::GlobalVariable(hlsl::ast::GlobalVariable {
                name: "g_myInBuffer".to_string(),
                global_type: hlsl::ast::Type::from_object(hlsl::ast::ObjectType::Buffer(
                    hlsl::ast::DataType(hlsl::ast::DataLayout::Scalar(hlsl::ast::ScalarType::Int), hlsl::ast::TypeModifier::default())
                )).into(),
                slot: Some(hlsl::ast::GlobalSlot::ReadSlot(0)),
                assignment: None,
            }),
            hlsl::ast::RootDefinition::Function(hlsl::ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: hlsl::ast::Type::void(),
                params: vec![hlsl::ast::FunctionParam { name: "x".to_string(), param_type: hlsl::ast::Type::uint().into(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            hlsl::ast::RootDefinition::Function(hlsl::ast::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: hlsl::ast::Type::void(),
                params: vec![hlsl::ast::FunctionParam { name: "x".to_string(), param_type: hlsl::ast::Type::float().into(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            hlsl::ast::RootDefinition::Function(hlsl::ast::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: hlsl::ast::Type::void(),
                params: vec![],
                body: vec![
                    hlsl::ast::Statement::Empty,
                    hlsl::ast::Statement::Var(hlsl::ast::VarDef { name: "a".to_string(), local_type: hlsl::ast::Type::uint().into(), assignment: None }),
                    hlsl::ast::Statement::Var(hlsl::ast::VarDef { name: "b".to_string(), local_type: hlsl::ast::Type::uint().into(), assignment: None }),
                    hlsl::ast::Statement::Expression(
                        hlsl::ast::Expression::BinaryOperation(hlsl::ast::BinOp::Assignment,
                            Box::new(hlsl::ast::Expression::Variable("a".to_string())),
                            Box::new(hlsl::ast::Expression::Variable("b".to_string()))
                        )
                    ),
                    hlsl::ast::Statement::If(
                        hlsl::ast::Expression::Variable("b".to_string()),
                        Box::new(hlsl::ast::Statement::Empty),
                    ),
                    hlsl::ast::Statement::Expression(
                        hlsl::ast::Expression::BinaryOperation(hlsl::ast::BinOp::Assignment,
                            Box::new(hlsl::ast::Expression::ArraySubscript(
                                Box::new(hlsl::ast::Expression::Variable("g_myInBuffer".to_string())),
                                Box::new(hlsl::ast::Expression::Literal(hlsl::ast::Literal::Int(0)))
                            )),
                            Box::new(hlsl::ast::Expression::Literal(hlsl::ast::Literal::Int(4)))
                        ),
                    ),
                    hlsl::ast::Statement::Expression(
                        hlsl::ast::Expression::Call(
                            Box::new(hlsl::ast::Expression::Variable("myFunc".to_string())),
                            vec![
                                hlsl::ast::Expression::Variable("b".to_string())
                            ]
                        ),
                    ),
                ],
                attributes: vec![hlsl::ast::FunctionAttribute::NumThreads(8, 8, 1)],
            }),
        ],
    };
    let res = typeparse(&module);
    assert!(res.is_ok(), "{:?}", res);

    let clc_res = transpile(&res.unwrap());
    assert!(clc_res.is_ok(), "{:?}", clc_res);
}