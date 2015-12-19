use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::collections::HashSet;
use BindMap;
use super::cil as dst;
use super::fragments::Fragment;
use super::super::hlsl::ir as src;
use super::super::hlsl::globals_analysis;

#[derive(PartialEq, Debug, Clone)]
pub enum TranspileError {
    Unknown,
    Internal(String),

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
            TranspileError::Internal(_) => "unknown transpiler error",
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
struct VariableDecl(dst::LocalId, ParamType);

impl VariableDecl {
    fn as_local(&self) -> dst::LocalId {
        self.0.clone()
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum StructSource {
    Struct(src::StructId),
    ConstantBuffer(src::ConstantBufferId),
}

struct GlobalIdAllocator {
    global_id_map: HashMap<src::GlobalId, dst::GlobalId>,
    function_id_map: HashMap<src::FunctionId, dst::FunctionId>,
    struct_id_map: HashMap<StructSource, dst::StructId>,
    fragments: HashMap<Fragment, dst::FunctionId>,
    global_name_map: HashMap<dst::GlobalId, String>,
    function_name_map: HashMap<dst::FunctionId, String>,
    struct_name_map: HashMap<dst::StructId, String>,
    last_global_id: dst::GlobalId,
    last_function_id: dst::FunctionId,
    last_struct_id: dst::StructId,
}

impl GlobalIdAllocator {
    fn from_globals(globals: &src::GlobalDeclarations, lifted: &HashSet<src::GlobalId>) -> Result<GlobalIdAllocator, TranspileError> {
        let mut context = GlobalIdAllocator {
            global_id_map: HashMap::new(),
            function_id_map: HashMap::new(),
            struct_id_map: HashMap::new(),
            fragments: HashMap::new(),
            global_name_map: HashMap::new(),
            function_name_map: HashMap::new(),
            struct_name_map: HashMap::new(),
            last_global_id: dst::GlobalId(0),
            last_function_id: dst::FunctionId(0),
            last_struct_id: dst::StructId(0),
        };

        // Insert global variables
        {
            let mut keys = globals.globals.keys().collect::<Vec<&src::GlobalId>>();
            keys.sort();
            for var_id in keys {
                if !lifted.contains(var_id) {
                    let var_name = globals.globals.get(var_id).expect("global variables from keys");
                    let dst_id = context.insert_identifier_global(var_id.clone());
                    context.global_name_map.insert(dst_id, var_name.clone());
                }
            }
        }

        // Insert functions
        {
            let mut keys = globals.functions.keys().collect::<Vec<&src::FunctionId>>();
            keys.sort();
            for func_id in keys {
                let func_name = globals.functions.get(func_id).expect("functions from keys");
                let dst_id = context.insert_identifier_function(func_id.clone());
                context.function_name_map.insert(dst_id, func_name.clone());
            }
        }


        // Insert structs
        {
            let mut keys = globals.structs.keys().collect::<Vec<&src::StructId>>();
            keys.sort();
            for id in keys {
                let struct_name = globals.structs.get(id).expect("structs from keys");
                let dst_id = context.insert_identifier_struct(StructSource::Struct(id.clone()));
                context.struct_name_map.insert(dst_id, struct_name.clone());
            };
        }

        // Insert cbuffers
        {
            let mut keys = globals.constants.keys().collect::<Vec<&src::ConstantBufferId>>();
            keys.sort();
            for id in keys {
                let cbuffer_name = globals.constants.get(id).expect("cbuffers from keys");
                let dst_id = context.insert_identifier_struct(StructSource::ConstantBuffer(id.clone()));
                context.struct_name_map.insert(dst_id, (cbuffer_name.clone() + "_t"));
            };
        }

        Ok(context)
    }

    fn insert_identifier_global(&mut self, id: src::GlobalId) -> dst::GlobalId {
        let identifier = self.last_global_id.clone();
        self.last_global_id = dst::GlobalId(self.last_global_id.0 + 1);
        let r = self.global_id_map.insert(id, identifier);
        assert_eq!(r, None);
        identifier
    }

    fn insert_identifier_function(&mut self, id: src::FunctionId) -> dst::FunctionId {
        let identifier = self.last_function_id.clone();
        self.last_function_id = dst::FunctionId(self.last_function_id.0 + 1);
        let r = self.function_id_map.insert(id, identifier);
        assert_eq!(r, None);
        identifier
    }

    fn fetch_fragment(&mut self, fragment: Fragment) -> dst::FunctionId {
        match self.fragments.entry(fragment.clone()) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let identifier = self.last_function_id.clone();
                self.last_function_id = dst::FunctionId(self.last_function_id.0 + 1);
                vacant.insert(identifier);
                self.function_name_map.insert(identifier, fragment.get_candidate_name());
                identifier
            },
        }
    }

    fn insert_identifier_struct(&mut self, id: StructSource) -> dst::StructId {
        let identifier = self.last_struct_id.clone();
        self.last_struct_id = dst::StructId(self.last_struct_id.0 + 1);
        let r = self.struct_id_map.insert(id, identifier);
        assert_eq!(r, None);
        identifier
    }
}

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone, Copy)]
enum GlobalArgument {
    Global(src::GlobalId),
    ConstantBuffer(src::ConstantBufferId),
}

#[derive(PartialEq, Debug, Clone)]
struct FunctionDecl {
    param_types: Vec<ParamType>,
    additional_arguments: Vec<GlobalArgument>,
}

struct Context {
    global_ids: GlobalIdAllocator,
    lifted_global_names: HashMap<src::GlobalId, String>,
    lifted_cbuffer_names: HashMap<src::ConstantBufferId, String>,
    function_decl_map: HashMap<src::FunctionId, FunctionDecl>,
    variable_scopes: Vec<HashMap<src::VariableId, VariableDecl>>,
    lifted_arguments: Option<HashMap<GlobalArgument, dst::LocalId>>,
    next_local_id: Option<dst::LocalId>,
    local_names: Option<HashMap<dst::LocalId, String>>,
    type_context: src::TypeContext,
    global_type_map: HashMap<src::GlobalId, dst::Type>,
    kernel_arguments: Vec<GlobalArgument>,
}

impl Context {

    fn from_globals(table: &src::GlobalTable, globals: &src::GlobalDeclarations, root_defs: &[src::RootDefinition]) -> Result<(Context, BindMap), TranspileError> {

        let mut lifted: HashSet<src::GlobalId> = HashSet::new();
        for rootdef in root_defs {
            match *rootdef {
                src::RootDefinition::GlobalVariable(ref gv) => {
                    if is_global_lifted(gv) {
                        lifted.insert(gv.id);
                    }
                },
                _ => { },
            }
        };

        let mut context = Context {
            global_ids: try!(GlobalIdAllocator::from_globals(globals, &lifted)),
            lifted_global_names: HashMap::new(),
            lifted_cbuffer_names: HashMap::new(),
            function_decl_map: HashMap::new(),
            variable_scopes: vec![],
            lifted_arguments: None,
            next_local_id: None,
            local_names: None,
            type_context: match src::TypeContext::from_roots(root_defs) {
                Ok(rt) => rt,
                Err(()) => return Err(TranspileError::Unknown),
            },
            global_type_map: HashMap::new(),
            kernel_arguments: vec![],
        };

        let usage = globals_analysis::GlobalUsage::analyse(root_defs);

        let mut global_type_map: HashMap<src::GlobalId, src::GlobalType> = HashMap::new();
        for rootdef in root_defs {
            match *rootdef {
                src::RootDefinition::GlobalVariable(ref gv) => {
                    if is_global_lifted(gv) {
                        global_type_map.insert(gv.id, gv.global_type.clone());
                        let cl_ty = try!(get_cl_global_type(&gv.id, &gv.global_type, &usage, &context));
                        context.global_type_map.insert(gv.id.clone(), cl_ty);
                        context.lifted_global_names.insert(gv.id.clone(), globals.globals.get(&gv.id).expect("global does not exist").clone());
                    }
                },
                src::RootDefinition::ConstantBuffer(ref cb) => {
                    context.lifted_cbuffer_names.insert(cb.id.clone(), globals.constants.get(&cb.id).expect("cbuffer does not exist").clone());
                },
                _ => { },
            }
        };

        let mut binds = BindMap::new();

        {
            let mut current = 0;
            let mut c_keys = table.constants.keys().collect::<Vec<&u32>>();
            c_keys.sort();
            for cbuffer_key in c_keys {
                let cbuffer_id = table.constants.get(cbuffer_key).expect("bad cbuffer key");
                context.kernel_arguments.push(GlobalArgument::ConstantBuffer(cbuffer_id.clone()));
                binds.cbuffer_map.insert(*cbuffer_key, current);
                current = current + 1;
            }
            let mut r_keys = table.r_resources.keys().collect::<Vec<&u32>>();
            r_keys.sort();
            for global_entry_key in r_keys {
                let global_entry = table.r_resources.get(global_entry_key).expect("bad r key key");
                context.kernel_arguments.push(GlobalArgument::Global(global_entry.id.clone()));
                binds.read_map.insert(*global_entry_key, current);
                current = current + 1;
            }
            let mut rw_keys = table.rw_resources.keys().collect::<Vec<&u32>>();
            rw_keys.sort();
            for global_entry_key in rw_keys {
                let global_entry = table.rw_resources.get(global_entry_key).expect("bad rw key key");
                context.kernel_arguments.push(GlobalArgument::Global(global_entry.id.clone()));
                binds.write_map.insert(*global_entry_key, current);
                current = current + 1;
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

                    let mut global_args: Vec<GlobalArgument>  = vec![];

                    let mut c_keys = function_global_usage.cbuffers.iter().collect::<Vec<&src::ConstantBufferId>>();
                    c_keys.sort();
                    for id in c_keys {
                        global_args.push(GlobalArgument::ConstantBuffer(id.clone()));
                    }

                    let mut g_keys = function_global_usage.globals.iter().collect::<Vec<&src::GlobalId>>();
                    g_keys.sort();
                    for id in g_keys {
                        global_args.push(GlobalArgument::Global(id.clone()));
                    }

                    let decl = FunctionDecl {
                        param_types: param_types,
                        additional_arguments: global_args,
                    };

                    let ret = context.function_decl_map.insert(func.id.clone(), decl);
                    assert_eq!(ret, None);
                },
                _ => { },
            }
        }

        Ok((context, binds))
    }

    fn get_function(&self, id: &src::FunctionId) -> Result<(dst::FunctionId, FunctionDecl), TranspileError> {
        let function_id = try!(self.get_function_name(id));
        match self.function_decl_map.get(id) {
            Some(ref decl) => Ok((function_id, (*decl).clone())),
            None => panic!("Function not defined"), // Name exists but decl doesn't
        }
    }

    fn get_function_name(&self, id: &src::FunctionId) -> Result<dst::FunctionId, TranspileError> {
        match self.global_ids.function_id_map.get(id) {
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
                    ParamType::Normal => dst::Expression::Local(s.clone()),
                    ParamType::Pointer => dst::Expression::Deref(Box::new(dst::Expression::Local(s.clone())))
                }),
                None => Err(TranspileError::UnknownVariableId),
            }
        }
    }

    /// Get the name of a variable declared in the current block
    fn get_variable_id(&self, id: &src::VariableId) -> Result<dst::LocalId, TranspileError> {
        assert!(self.variable_scopes.len() > 0);
        match self.variable_scopes[self.variable_scopes.len() - 1].get(id) {
            Some(v) => Ok(v.as_local()),
            None => Err(TranspileError::UnknownVariableId),
        }
    }

    fn get_global_lifted_id(&self, id: &src::GlobalId) -> Result<dst::LocalId, TranspileError> {
        match self.lifted_arguments {
            Some(ref map) => {
                match map.get(&GlobalArgument::Global(id.clone())) {
                    Some(local_id) => Ok(local_id.clone()),
                    None => Err(TranspileError::Internal("no such lifted variable".to_string())),
                }
            },
            None => Err(TranspileError::Internal("not inside a function".to_string())),
        }
    }

    fn get_global_static_id(&self, id: &src::GlobalId) -> Result<dst::GlobalId, TranspileError> {
        match self.global_ids.global_id_map.get(id) {
            Some(global_id) => Ok(global_id.clone()),
            None => Err(TranspileError::UnknownVariableId),
        }
    }

    fn get_global_var(&self, id: &src::GlobalId) -> Result<dst::Expression, TranspileError> {
        match self.get_global_lifted_id(id) {
            Ok(val) => return Ok(dst::Expression::Local(val)),
            Err(_) => { },
        };
        match self.get_global_static_id(id) {
            Ok(global_id) => Ok(dst::Expression::Global(global_id.clone())),
            Err(err) => Err(err),
        }
    }

    /// Get the name of a struct
    fn get_struct_name(&self, id: &src::StructId) -> Result<dst::StructId, TranspileError> {
        match self.global_ids.struct_id_map.get(&StructSource::Struct(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownStructId(id.clone())),
        }
    }

    /// Get the name of the struct used for a cbuffer
    fn get_cbuffer_struct_name(&self, id: &src::ConstantBufferId) -> Result<dst::StructId, TranspileError> {
        match self.global_ids.struct_id_map.get(&StructSource::ConstantBuffer(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
        }
    }

    /// Get the name of the cbuffer instance
    fn get_cbuffer_instance_id(&self, id: &src::ConstantBufferId) -> Result<dst::LocalId, TranspileError> {
        match self.lifted_arguments {
            Some(ref args) => match args.get(&GlobalArgument::ConstantBuffer(*id)) {
                Some(ref v) => Ok(*v.clone()),
                None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
            },
            None => Err(TranspileError::Internal("not in scope".to_string())),
        }
    }

    /// Get the expression to find the given constant
    fn get_constant(&self, id: &src::ConstantBufferId, name: String) -> Result<dst::Expression, TranspileError> {
        Ok(dst::Expression::MemberDeref(
            Box::new(dst::Expression::Local(
                try!(self.get_cbuffer_instance_id(id))
            )),
            name
        ))
    }

    fn make_identifier(&mut self) -> dst::LocalId {
        let current_id = self.next_local_id.expect("no local id").clone();
        self.next_local_id = Some(dst::LocalId(current_id.0 + 1));
        current_id
    }

    fn fetch_fragment(&mut self, fragment: Fragment) -> dst::FunctionId {
        self.global_ids.fetch_fragment(fragment)
    }

    fn push_scope(&mut self, scope_block: &src::ScopeBlock) {
        self.push_scope_with_pointer_overrides(scope_block, &[])
    }

    fn push_scope_for_function(&mut self, scope_block: &src::ScopeBlock, pointers: &[src::VariableId], id: &src::FunctionId) {
        let additional_arguments = self.function_decl_map.get(id).expect("function does not exist").additional_arguments.clone();
        self.push_scope_with_additional_args(scope_block, pointers, &additional_arguments)
    }

    fn push_scope_for_kernel(&mut self, scope_block: &src::ScopeBlock) {
        let additional_arguments = self.kernel_arguments.clone();
        self.push_scope_with_additional_args(scope_block, &[], &additional_arguments)
    }

    fn push_scope_with_additional_args(&mut self, scope_block: &src::ScopeBlock, pointers: &[src::VariableId], additional_arguments: &[GlobalArgument]) {
        assert_eq!(self.lifted_arguments, None);
        assert_eq!(self.next_local_id, None);
        assert_eq!(self.local_names, None);
        self.next_local_id = Some(dst::LocalId(0));
        self.local_names = Some(HashMap::new());
        let mut map = HashMap::new();
        for arg in additional_arguments {
            let identifier = self.make_identifier();
            map.insert(arg.clone(), identifier);
            match self.local_names {
                Some(ref mut names) => {
                    let name = match *arg {
                        GlobalArgument::Global(ref id) => {
                            self.lifted_global_names.get(id).expect("global name doesn't exist").clone()
                        },
                        GlobalArgument::ConstantBuffer(ref id)=> {
                            self.lifted_cbuffer_names.get(id).expect("cbuffer name doesn't exist").clone()
                        },
                    };
                    names.insert(identifier, name.to_string())
                },
                None => panic!("not inside function"),
            };
        }
        self.lifted_arguments = Some(map);
        self.push_scope_with_pointer_overrides(scope_block, pointers);
    }

    fn push_scope_with_pointer_overrides(&mut self, scope_block: &src::ScopeBlock, pointers: &[src::VariableId]) {
        self.variable_scopes.push(HashMap::new());
        for (var_id, var_name) in &scope_block.1.variables {
            let identifier = self.make_identifier();
            let map = self.variable_scopes.last_mut().expect("no scopes after pushing scope");
            map.insert(var_id.clone(), VariableDecl(identifier, if pointers.iter().any(|pp| pp == var_id) { ParamType::Pointer } else { ParamType::Normal }));
            match self.local_names {
                Some(ref mut names) => names.insert(identifier, var_name.clone()),
                None => panic!("not inside function"),
            };
        };
        self.type_context.push_scope(scope_block);
    }

    fn pop_scope(&mut self) {
        assert!(self.variable_scopes.len() > 1);
        self.variable_scopes.pop();
        self.type_context.pop_scope();
    }

    fn pop_scope_for_function(&mut self) -> dst::LocalDeclarations {
        assert_eq!(self.variable_scopes.len(), 1);
        let decls = match self.local_names {
            Some(ref locals) => dst::LocalDeclarations { locals: locals.clone() },
            None => panic!("not inside function"),
        };
        self.lifted_arguments = None;
        self.next_local_id = None;
        self.local_names = None;
        self.variable_scopes.pop();
        self.type_context.pop_scope();
        decls
    }

    fn destruct(self) -> (dst::GlobalDeclarations, HashMap<Fragment, dst::FunctionId>) {
        let mut decls = dst::GlobalDeclarations {
            globals: HashMap::new(),
            structs: HashMap::new(),
            functions: HashMap::new()
        };
        for (id, name) in self.global_ids.global_name_map {
            decls.globals.insert(id, name);
        };
        for (id, name) in self.global_ids.struct_name_map {
            decls.structs.insert(id, name);
        };
        for (id, name) in self.global_ids.function_name_map {
            decls.functions.insert(id, name);
        };
        (decls, self.global_ids.fragments)
    }
}

fn is_global_lifted(gv: &src::GlobalVariable) -> bool {
    let &src::GlobalType(src::Type(_, ref modifier), ref gs, _) = &gv.global_type;
    let static_const = modifier.is_const && *gs == src::GlobalStorage::Static;
    !static_const
}

fn get_cl_global_type(id: &src::GlobalId, ty: &src::GlobalType, usage: &globals_analysis::GlobalUsage, context: &Context) -> Result<dst::Type, TranspileError> {
    let tyl = &(ty.0).0;
    Ok(match *tyl {
        src::TypeLayout::Object(src::ObjectType::Buffer(ref data_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_datatype(data_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::StructuredBuffer(ref structured_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_structuredtype(structured_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::RWBuffer(ref data_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_datatype(data_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::RWStructuredBuffer(ref structured_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global, Box::new(try!(transpile_structuredtype(structured_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::RWTexture2D(_)) => {
            let read = usage.image_reads.contains(id);
            let write = usage.image_writes.contains(id);
            let access = match (read, write) {
                (false, false) | (true, false) => dst::AccessModifier::ReadOnly,
                (false, true) => dst::AccessModifier::WriteOnly,
                (true, true) => dst::AccessModifier::ReadWrite, // OpenCL 2.0 only
            };
            dst::Type::Image2D(access)
        }
        src::TypeLayout::Object(src::ObjectType::ByteAddressBuffer) => {
            dst::Type::Pointer(dst::AddressSpace::Global, Box::new(dst::Type::Scalar(dst::Scalar::UChar)))
        }
        src::TypeLayout::Object(src::ObjectType::RWByteAddressBuffer) => {
            dst::Type::Pointer(dst::AddressSpace::Global, Box::new(dst::Type::Scalar(dst::Scalar::UChar)))
        }
        _ => return Err(TranspileError::TypeIsNotAllowedAsGlobal(ty.clone())),
    })
}

fn get_cl_cbuffer_type(id: &src::ConstantBufferId, context: &Context) -> Result<dst::Type, TranspileError> {
    Ok(dst::Type::Pointer(
        dst::AddressSpace::Constant,
        Box::new(dst::Type::Struct(try!(context.get_cbuffer_struct_name(id))))
    ))
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

fn write_unary(op: dst::UnaryOp, expr: &src::Expression, context: &mut Context) -> Result<dst::Expression, TranspileError> {
    Ok(dst::Expression::UnaryOperation(op, Box::new(try!(transpile_expression(expr, context)))))
}

fn write_func(name: &'static str, args: &[&src::Expression], context: &mut Context) -> Result<dst::Expression, TranspileError> {
    Ok(dst::Expression::UntypedIntrinsic(
        name.to_string(),
        try!(args.iter().fold(Ok(vec![]), |result, exp| {
            let mut vec = try!(result);
            vec.push(try!(transpile_expression(exp, context)));
            Ok(vec)
        }))
    ))
}

fn transpile_intrinsic(intrinsic: &src::Intrinsic, context: &mut Context) -> Result<dst::Expression, TranspileError> {
    match *intrinsic {
        src::Intrinsic::PrefixIncrement(_, ref expr) => write_unary(dst::UnaryOp::PrefixIncrement, expr, context),
        src::Intrinsic::PrefixDecrement(_, ref expr) => write_unary(dst::UnaryOp::PrefixDecrement, expr, context),
        src::Intrinsic::PostfixIncrement(_, ref expr) => write_unary(dst::UnaryOp::PostfixIncrement, expr, context),
        src::Intrinsic::PostfixDecrement(_, ref expr) => write_unary(dst::UnaryOp::PostfixDecrement, expr, context),
        src::Intrinsic::Plus(_, ref expr) => write_unary(dst::UnaryOp::Plus, expr, context),
        src::Intrinsic::Minus(_, ref expr) => write_unary(dst::UnaryOp::Minus, expr, context),
        src::Intrinsic::LogicalNot(_, ref expr) => write_unary(dst::UnaryOp::LogicalNot, expr, context),
        src::Intrinsic::BitwiseNot(_, ref expr) => write_unary(dst::UnaryOp::BitwiseNot, expr, context),
        src::Intrinsic::AllMemoryBarrier | src::Intrinsic::AllMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::UntypedIntrinsic(
                "barrier".to_string(),
                vec![dst::Expression::BinaryOperation(dst::BinOp::BitwiseOr,
                    Box::new(dst::Expression::UntypedLiteral("CLK_LOCAL_MEM_FENCE".to_string())),
                    Box::new(dst::Expression::UntypedLiteral("CLK_GLOBAL_MEM_FENCE".to_string()))
                )]
            ))
        },
        src::Intrinsic::DeviceMemoryBarrier | src::Intrinsic::DeviceMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::UntypedIntrinsic(
                "barrier".to_string(),
                vec![dst::Expression::UntypedLiteral("CLK_GLOBAL_MEM_FENCE".to_string())]
            ))
        },
        src::Intrinsic::GroupMemoryBarrier | src::Intrinsic::GroupMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::UntypedIntrinsic(
                "barrier".to_string(),
                vec![dst::Expression::UntypedLiteral("CLK_LOCAL_MEM_FENCE".to_string())]
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
        src::Intrinsic::AsDouble(_, _) => Err(TranspileError::IntrinsicUnimplemented),
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
            Ok(dst::Expression::UntypedIntrinsic(
                "length".to_string(),
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
        src::Intrinsic::Min(_, _) => Err(TranspileError::IntrinsicUnimplemented),
        src::Intrinsic::Max(_, _) => Err(TranspileError::IntrinsicUnimplemented),
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
            Ok(dst::Expression::UntypedIntrinsic(
                func_name.to_string(),
                vec![cl_tex, cl_loc]
            ))
        },
        src::Intrinsic::ByteAddressBufferLoad(ref buffer, ref loc) |
        src::Intrinsic::RWByteAddressBufferLoad(ref buffer, ref loc) |
        src::Intrinsic::ByteAddressBufferLoad2(ref buffer, ref loc) |
        src::Intrinsic::RWByteAddressBufferLoad2(ref buffer, ref loc) |
        src::Intrinsic::ByteAddressBufferLoad3(ref buffer, ref loc) |
        src::Intrinsic::RWByteAddressBufferLoad3(ref buffer, ref loc) |
        src::Intrinsic::ByteAddressBufferLoad4(ref buffer, ref loc) |
        src::Intrinsic::RWByteAddressBufferLoad4(ref buffer, ref loc) => {
            let ty = Box::new(match *intrinsic {
                src::Intrinsic::ByteAddressBufferLoad(_, _) |
                src::Intrinsic::RWByteAddressBufferLoad(_, _) => {
                    dst::Type::Scalar(dst::Scalar::UInt)
                }
                src::Intrinsic::ByteAddressBufferLoad2(_, _) |
                src::Intrinsic::RWByteAddressBufferLoad2(_, _) => {
                    dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Two)
                }
                src::Intrinsic::ByteAddressBufferLoad3(_, _)|
                src::Intrinsic::RWByteAddressBufferLoad3(_, _) => {
                    dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Three)
                }
                src::Intrinsic::ByteAddressBufferLoad4(_, _) |
                src::Intrinsic::RWByteAddressBufferLoad4(_, _) => {
                    dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Four)
                }
                _ => unreachable!(),
            });
            let cl_buffer = Box::new(try!(transpile_expression(buffer, context)));
            let cl_loc = Box::new(try!(transpile_expression(loc, context)));
            Ok(dst::Expression::Deref(Box::new(dst::Expression::Cast(
                dst::Type::Pointer(dst::AddressSpace::Global, ty),
                Box::new(dst::Expression::BinaryOperation(dst::BinOp::Add, cl_buffer, cl_loc))
            ))))
        },
        src::Intrinsic::RWByteAddressBufferStore(ref buffer, ref loc, ref value) |
        src::Intrinsic::RWByteAddressBufferStore2(ref buffer, ref loc, ref value) |
        src::Intrinsic::RWByteAddressBufferStore3(ref buffer, ref loc, ref value) |
        src::Intrinsic::RWByteAddressBufferStore4(ref buffer, ref loc, ref value) => {
            let ty = Box::new(match *intrinsic {
                src::Intrinsic::RWByteAddressBufferStore(_, _, _) => dst::Type::Scalar(dst::Scalar::UInt),
                src::Intrinsic::RWByteAddressBufferStore2(_, _, _) => dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Two),
                src::Intrinsic::RWByteAddressBufferStore3(_, _, _) => dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Three),
                src::Intrinsic::RWByteAddressBufferStore4(_, _, _) => dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Four),
                _ => unreachable!(),
            });
            let cl_buffer = Box::new(try!(transpile_expression(buffer, context)));
            let cl_loc = Box::new(try!(transpile_expression(loc, context)));
            let cl_value = Box::new(try!(transpile_expression(value, context)));
            Ok(dst::Expression::BinaryOperation(
                dst::BinOp::Assignment,
                Box::new(dst::Expression::Deref(Box::new(dst::Expression::Cast(
                    dst::Type::Pointer(dst::AddressSpace::Global, ty),
                    Box::new(dst::Expression::BinaryOperation(dst::BinOp::Add, cl_buffer, cl_loc))
                )))),
                cl_value
            ))
        },
    }
}

fn transpile_expression(expression: &src::Expression, context: &mut Context) -> Result<dst::Expression, TranspileError> {
    match expression {
        &src::Expression::Literal(ref lit) => Ok(dst::Expression::Literal(try!(transpile_literal(lit)))),
        &src::Expression::Variable(ref var_ref) => context.get_variable_ref(var_ref),
        &src::Expression::Global(ref id) => context.get_global_var(id),
        &src::Expression::ConstantVariable(ref id, ref name) => context.get_constant(id, name.clone()),
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
        &src::Expression::Swizzle(ref vec, ref swizzle) => {
            Ok(dst::Expression::Swizzle(
                Box::new(try!(transpile_expression(vec, context))),
                swizzle.iter().map(|swizzle_slot| match *swizzle_slot {
                    src::SwizzleSlot::X => dst::SwizzleSlot::X,
                    src::SwizzleSlot::Y => dst::SwizzleSlot::Y,
                    src::SwizzleSlot::Z => dst::SwizzleSlot::Z,
                    src::SwizzleSlot::W => dst::SwizzleSlot::W,
                }).collect::<Vec<_>>()
            ))
        },
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
            let (func_expr, decl) = try!(context.get_function(func_id));
            assert_eq!(params.len(), decl.param_types.len());
            let mut params_exprs: Vec<dst::Expression> = vec![];
            for (param, pt) in params.iter().zip(decl.param_types) {
                let param_expr = try!(transpile_expression(param, context));
                let param_expr = match pt {
                    ParamType::Normal => param_expr,
                    ParamType::Pointer => dst::Expression::AddressOf(Box::new(param_expr)),
                };
                params_exprs.push(param_expr);
            };
            let mut final_arguments = vec![];
            for global in &decl.additional_arguments {
                final_arguments.push(dst::Expression::Local(match *global {
                    GlobalArgument::Global(ref id) => try!(context.get_global_lifted_id(id)),
                    GlobalArgument::ConstantBuffer(ref id) => try!(context.get_cbuffer_instance_id(id)),
                }));
            };
            final_arguments.append(&mut params_exprs);
            Ok(dst::Expression::Call(func_expr, final_arguments))
        },
        &src::Expression::Cast(ref cast_type, ref expr) => {
            let cl_type = try!(transpile_type(cast_type, context));
            let cl_expr = try!(transpile_expression(expr, context));
            Ok(match cl_type {
                dst::Type::Bool | dst::Type::Scalar(_) => dst::Expression::Cast(cl_type, Box::new(cl_expr)),
                dst::Type::Vector(ref scalar, ref dim) => {
                    let from_exp_type = match context.type_context.get_expression_type(expr) {
                        Ok(ty) => ty,
                        Err(()) => return Err(TranspileError::Internal("could not calculate type for cast source".to_string())),
                    };
                    let from_cl_type = try!(transpile_type(&from_exp_type.0, context));
                    let (from_scalar_type, from_dim) = match from_cl_type {
                        dst::Type::Vector(scalar, dim) => (scalar, dim),
                        _ => return Err(TranspileError::Internal("source of vector cast is not a vector".to_string())),
                    };
                    if from_dim != *dim {
                        return Err(TranspileError::Internal("vector cast between different dimensions".to_string()))
                    };
                    let cast_func_id = context.fetch_fragment(Fragment::VectorCast(from_scalar_type, scalar.clone(), from_dim));
                    dst::Expression::Call(cast_func_id, vec![cl_expr])
                },
                _ => return Err(TranspileError::Internal(format!("don't know how to cast to this type ({:?})", cl_type))),
            })
        },
        &src::Expression::Intrinsic(ref intrinsic) => transpile_intrinsic(intrinsic, context),
    }
}

fn transpile_vardef(vardef: &src::VarDef, context: &mut Context) -> Result<dst::VarDef, TranspileError> {
    Ok(dst::VarDef {
        id: try!(context.get_variable_id(&vardef.id)),
        typename: try!(transpile_localtype(&vardef.local_type, context)),
        assignment: match &vardef.assignment { &None => None, &Some(ref expr) => Some(try!(transpile_expression(expr, context))) },
    })
}

#[allow(dead_code)]
fn transpile_condition(cond: &src::Condition, context: &mut Context) -> Result<dst::Condition, TranspileError> {
    match *cond {
        src::Condition::Expr(ref expr) => {
            let expr_ir = try!(transpile_expression(expr, context));
            Ok(dst::Condition::Expr(expr_ir))
        },
        src::Condition::Assignment(ref vd) => {
            let cl_vardef = try!(transpile_vardef(vd, context));
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

fn transpile_param(param: &src::FunctionParam, context: &mut Context) -> Result<dst::FunctionParam, TranspileError> {
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
        id: try!(context.get_variable_id(&param.id)),
        typename: ty,
    })
}

fn transpile_params(params: &[src::FunctionParam], context: &mut Context) -> Result<Vec<dst::FunctionParam>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_param(param, context)));
    }
    Ok(cl_params)
}

fn transpile_kernel_input_semantic(param: &src::KernelParam, context: &mut Context) -> Result<dst::Statement, TranspileError> {
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
                id: try!(context.get_variable_id(&param.0)),
                typename: typename,
                assignment: Some(assign),
            }))
        },
        &src::KernelSemantic::GroupId => unimplemented!(),
        &src::KernelSemantic::GroupIndex => unimplemented!(),
        &src::KernelSemantic::GroupThreadId => unimplemented!(),
    }
}

fn transpile_kernel_input_semantics(params: &[src::KernelParam], context: &mut Context) -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_kernel_input_semantic(param, context)));
    }
    Ok(cl_params)
}

fn transpile_structdefinition(structdefinition: &src::StructDefinition, context: &mut Context) -> Result<dst::RootDefinition, TranspileError> {
    Ok(dst::RootDefinition::Struct(dst::StructDefinition {
        id: try!(context.get_struct_name(&structdefinition.id)),
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
        id: try!(context.get_cbuffer_struct_name(&cb.id)),
        members: members,
    }))
}

fn transpile_globalvariable(gv: &src::GlobalVariable, context: &mut Context) -> Result<Option<dst::RootDefinition>, TranspileError> {
    let &src::GlobalType(src::Type(ref ty, ref modifiers), _, _) = &gv.global_type;
    let lifted = is_global_lifted(gv);
    if lifted {
        return Ok(None)
    } else {
        let global_id = try!(context.get_global_static_id(&gv.id));
        let cl_type = try!(transpile_type(&src::Type(ty.clone(), modifiers.clone()), &context));
        let cl_init = match &gv.assignment {
            &Some(ref expr) => Some(try!(transpile_expression(expr, context))),
            &None => None,
        };
        Ok(Some(dst::RootDefinition::GlobalVariable(dst::GlobalVariable {
            id: global_id,
            ty: cl_type,
            address_space: dst::AddressSpace::Constant,
            init: cl_init,
        })))
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
    context.push_scope_for_function(&func.scope_block, &out_params, &func.id);
    let (_, decl) = try!(context.get_function(&func.id));
    let mut params = vec![];
    for global in &decl.additional_arguments {
        params.push(match *global {
            GlobalArgument::Global(ref id) => dst::FunctionParam {
                id: try!(context.get_global_lifted_id(id)),
                typename: context.global_type_map.get(id).unwrap().clone(),
            },
            GlobalArgument::ConstantBuffer(ref id) => dst::FunctionParam {
                id: try!(context.get_cbuffer_instance_id(id)),
                typename: try!(get_cl_cbuffer_type(id, context))
            },
        });
    };
    params.append(&mut try!(transpile_params(&func.params, context)));
    let body = try!(transpile_statements(&func.scope_block.0, context));
    let decls = context.pop_scope_for_function();
    let cl_func = dst::FunctionDefinition {
        id: try!(context.get_function_name(&func.id)),
        returntype: try!(transpile_type(&func.returntype, context)),
        params: params,
        body: body,
        local_declarations: decls,
    };
    Ok(dst::RootDefinition::Function(cl_func))
}

fn transpile_kernel(kernel: &src::Kernel, context: &mut Context) -> Result<dst::RootDefinition, TranspileError> {
    context.push_scope_for_kernel(&kernel.scope_block);
    let mut params = vec![];
    for global in &context.kernel_arguments {
        params.push(match *global {
            GlobalArgument::Global(ref id) => dst::KernelParam {
                id: try!(context.get_global_lifted_id(id)),
                typename: match context.global_type_map.get(id) {
                    Some(ty) => ty.clone(),
                    None => panic!("global type unknown {:?}", id.clone()),
                }
            },
            GlobalArgument::ConstantBuffer(ref id) => dst::KernelParam {
                id: try!(context.get_cbuffer_instance_id(id)),
                typename: try!(get_cl_cbuffer_type(id, context))
            },
        });
    };
    let mut body = try!(transpile_kernel_input_semantics(&kernel.params, context));
    let mut main_body = try!(transpile_statements(&kernel.scope_block.0, context));
    let decls = context.pop_scope_for_function();
    body.append(&mut main_body);
    let cl_kernel = dst::Kernel {
        params: params,
        body: body,
        group_dimensions: dst::Dimension(kernel.group_dimensions.0, kernel.group_dimensions.1, kernel.group_dimensions.2),
        local_declarations: decls,
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

    let (decls, fragments) = context.destruct();

    let cl_module = dst::Module {
        root_definitions: cl_defs,
        binds: binds,
        global_declarations: decls,
        fragments: fragments,
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