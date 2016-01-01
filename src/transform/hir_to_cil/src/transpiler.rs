use std::error;
use std::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::collections::HashSet;
use slp_shared::BindMap;
use slp_lang_cil as dst;
use slp_lang_cst::fragments::Fragment;
use slp_lang_hir as src;
use slp_lang_hir::globals_analysis;

#[derive(PartialEq, Debug, Clone)]
pub enum TranspileError {
    Unknown,
    Internal(String),
    InternalTypeError(src::TypeError),

    TypeIsNotAllowedAsGlobal(src::GlobalType),
    CouldNotGetEquivalentType(src::TypeLayout),

    GlobalFoundThatIsntInKernelParams(src::GlobalVariable),

    UnknownFunctionId(src::FunctionId),
    UnknownStructId(src::StructId),
    UnknownConstantBufferId(src::ConstantBufferId),
    InvalidVariableRef,
    UnknownVariableId,

    BoolVectorsNotSupported,
    IntsMustBeTyped,

    Intrinsic1Unimplemented(src::Intrinsic1),
    Intrinsic2Unimplemented(src::Intrinsic2),
}

impl error::Error for TranspileError {
    fn description(&self) -> &str {
        match *self {
            TranspileError::Unknown => "unknown transpiler error",
            TranspileError::Internal(_) => "unknown transpiler error",
            TranspileError::InternalTypeError(_) => "internal type error",
            TranspileError::TypeIsNotAllowedAsGlobal(_) => "global variable has unsupported type",
            TranspileError::CouldNotGetEquivalentType(_) => "could not find equivalent clc type",
            TranspileError::GlobalFoundThatIsntInKernelParams(_) => "non-parameter global found",
            TranspileError::UnknownFunctionId(_) => "unknown function id",
            TranspileError::UnknownStructId(_) => "unknown struct id",
            TranspileError::UnknownConstantBufferId(_) => "unknown cbuffer id",
            TranspileError::InvalidVariableRef => "invalid variable ref",
            TranspileError::UnknownVariableId => "unknown variable id",
            TranspileError::BoolVectorsNotSupported => "bool vectors not supported",
            TranspileError::IntsMustBeTyped => "internal error: untyped int ended up in tree",
            TranspileError::Intrinsic1Unimplemented(_) |
            TranspileError::Intrinsic2Unimplemented(_) => "intrinsic function is not implemented",
        }
    }
}

impl fmt::Display for TranspileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

impl From<src::TypeError> for TranspileError {
    fn from(err: src::TypeError) -> TranspileError {
        TranspileError::InternalTypeError(err)
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
    fn from_globals(globals: &src::GlobalDeclarations,
                    lifted: &HashSet<src::GlobalId>)
                    -> Result<GlobalIdAllocator, TranspileError> {
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
            }
        }

        // Insert cbuffers
        {
            let mut keys = globals.constants.keys().collect::<Vec<&src::ConstantBufferId>>();
            keys.sort();
            for id in keys {
                let cbuffer_name = globals.constants.get(id).expect("cbuffers from keys");
                let dst_id =
                    context.insert_identifier_struct(StructSource::ConstantBuffer(id.clone()));
                context.struct_name_map.insert(dst_id, (cbuffer_name.clone() + "_t"));
            }
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
            }
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
    type_context: src::TypeState,
    global_type_map: HashMap<src::GlobalId, dst::Type>,
    kernel_arguments: Vec<GlobalArgument>,
    required_extensions: HashSet<dst::Extension>,
}

impl Context {
    fn from_globals(table: &src::GlobalTable,
                    globals: &src::GlobalDeclarations,
                    root_defs: &[src::RootDefinition])
                    -> Result<(Context, BindMap), TranspileError> {

        let mut lifted: HashSet<src::GlobalId> = HashSet::new();
        for rootdef in root_defs {
            match *rootdef {
                src::RootDefinition::GlobalVariable(ref gv) => {
                    if is_global_lifted(gv) {
                        lifted.insert(gv.id);
                    }
                }
                _ => {}
            }
        }

        let mut context = Context {
            global_ids: try!(GlobalIdAllocator::from_globals(globals, &lifted)),
            lifted_global_names: HashMap::new(),
            lifted_cbuffer_names: HashMap::new(),
            function_decl_map: HashMap::new(),
            variable_scopes: vec![],
            lifted_arguments: None,
            next_local_id: None,
            local_names: None,
            type_context: match src::TypeState::from_roots(root_defs) {
                Ok(rt) => rt,
                Err(()) => return Err(TranspileError::Unknown),
            },
            global_type_map: HashMap::new(),
            kernel_arguments: vec![],
            required_extensions: HashSet::new(),
        };

        let usage = globals_analysis::GlobalUsage::analyse(root_defs);

        let mut global_type_map: HashMap<src::GlobalId, src::GlobalType> = HashMap::new();
        let mut lifted_globals: HashSet<src::GlobalId> = HashSet::new();
        for rootdef in root_defs {
            match *rootdef {
                src::RootDefinition::GlobalVariable(ref gv) => {
                    if is_global_lifted(gv) {
                        lifted_globals.insert(gv.id);
                        global_type_map.insert(gv.id, gv.global_type.clone());
                        let cl_ty = try!(get_cl_global_type(&gv.id,
                                                            &gv.global_type,
                                                            &usage,
                                                            &context));
                        context.global_type_map.insert(gv.id.clone(), cl_ty);
                        context.lifted_global_names.insert(gv.id.clone(),
                                                           globals.globals
                                                                  .get(&gv.id)
                                                                  .expect("global does not exist")
                                                                  .clone());
                    }
                }
                src::RootDefinition::ConstantBuffer(ref cb) => {
                    context.lifted_cbuffer_names.insert(cb.id.clone(),
                                                        globals.constants
                                                               .get(&cb.id)
                                                               .expect("cbuffer does not exist")
                                                               .clone());
                }
                _ => {}
            }
        }

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
                let global_entry = table.rw_resources
                                        .get(global_entry_key)
                                        .expect("bad rw key key");
                context.kernel_arguments.push(GlobalArgument::Global(global_entry.id.clone()));
                binds.write_map.insert(*global_entry_key, current);
                current = current + 1;
            }
        }

        for rootdef in root_defs {
            match rootdef {
                &src::RootDefinition::Function(ref func) => {
                    let param_types = func.params.iter().fold(vec![], |mut param_types, param| {
                        match param.param_type.1 {
                            src::InputModifier::InOut | src::InputModifier::Out => {
                                param_types.push(ParamType::Pointer)
                            }
                            src::InputModifier::In => param_types.push(ParamType::Normal),
                        };
                        param_types
                    });
                    let function_global_usage = usage.functions.get(&func.id).unwrap_or_else(|| {
                        panic!("analysis missing function")
                    });

                    let mut global_args: Vec<GlobalArgument> = vec![];

                    let mut c_keys = function_global_usage.cbuffers
                                                          .iter()
                                                          .collect::<Vec<&src::ConstantBufferId>>();
                    c_keys.sort();
                    for id in c_keys {
                        global_args.push(GlobalArgument::ConstantBuffer(id.clone()));
                    }

                    let mut g_keys = function_global_usage.globals
                                                          .iter()
                                                          .collect::<Vec<&src::GlobalId>>();
                    g_keys.sort();
                    for id in g_keys {
                        if lifted_globals.contains(id) {
                            global_args.push(GlobalArgument::Global(id.clone()));
                        }
                    }

                    let decl = FunctionDecl {
                        param_types: param_types,
                        additional_arguments: global_args,
                    };

                    let ret = context.function_decl_map.insert(func.id.clone(), decl);
                    assert_eq!(ret, None);
                }
                _ => {}
            }
        }

        Ok((context, binds))
    }

    fn get_function(&self,
                    id: &src::FunctionId)
                    -> Result<(dst::FunctionId, FunctionDecl), TranspileError> {
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
    fn get_variable_ref(&self,
                        var_ref: &src::VariableRef)
                        -> Result<dst::Expression, TranspileError> {
        let scopes_up = (var_ref.1).0 as usize;
        if scopes_up >= self.variable_scopes.len() {
            return Err(TranspileError::UnknownVariableId);
        } else {
            let scope = self.variable_scopes.len() - scopes_up - 1;
            match self.variable_scopes[scope].get(&var_ref.0) {
                Some(&VariableDecl(ref s, ref pt)) => {
                    Ok(match *pt {
                        ParamType::Normal => dst::Expression::Local(s.clone()),
                        ParamType::Pointer => {
                            dst::Expression::Deref(Box::new(dst::Expression::Local(s.clone())))
                        }
                    })
                }
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
            }
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
            Err(_) => {}
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
    fn get_cbuffer_struct_name(&self,
                               id: &src::ConstantBufferId)
                               -> Result<dst::StructId, TranspileError> {
        match self.global_ids.struct_id_map.get(&StructSource::ConstantBuffer(*id)) {
            Some(v) => Ok(v.clone()),
            None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
        }
    }

    /// Get the name of the cbuffer instance
    fn get_cbuffer_instance_id(&self,
                               id: &src::ConstantBufferId)
                               -> Result<dst::LocalId, TranspileError> {
        match self.lifted_arguments {
            Some(ref args) => {
                match args.get(&GlobalArgument::ConstantBuffer(*id)) {
                    Some(ref v) => Ok(*v.clone()),
                    None => Err(TranspileError::UnknownConstantBufferId(id.clone())),
                }
            }
            None => Err(TranspileError::Internal("not in scope".to_string())),
        }
    }

    /// Get the expression to find the given constant
    fn get_constant(&self,
                    id: &src::ConstantBufferId,
                    name: String)
                    -> Result<dst::Expression, TranspileError> {
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

    fn push_scope_for_function(&mut self,
                               scope_block: &src::ScopeBlock,
                               pointers: &[src::VariableId],
                               id: &src::FunctionId) {
        let additional_arguments = self.function_decl_map
                                       .get(id)
                                       .expect("function does not exist")
                                       .additional_arguments
                                       .clone();
        self.push_scope_with_additional_args(scope_block, pointers, &additional_arguments)
    }

    fn push_scope_for_kernel(&mut self, scope_block: &src::ScopeBlock) {
        let additional_arguments = self.kernel_arguments.clone();
        self.push_scope_with_additional_args(scope_block, &[], &additional_arguments)
    }

    fn push_scope_with_additional_args(&mut self,
                                       scope_block: &src::ScopeBlock,
                                       pointers: &[src::VariableId],
                                       additional_arguments: &[GlobalArgument]) {
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
                            self.lifted_global_names
                                .get(id)
                                .expect("global name doesn't exist")
                                .clone()
                        }
                        GlobalArgument::ConstantBuffer(ref id) => {
                            self.lifted_cbuffer_names
                                .get(id)
                                .expect("cbuffer name doesn't exist")
                                .clone()
                        }
                    };
                    names.insert(identifier, name.to_string())
                }
                None => panic!("not inside function"),
            };
        }
        self.lifted_arguments = Some(map);
        self.push_scope_with_pointer_overrides(scope_block, pointers);
    }

    fn push_scope_with_pointer_overrides(&mut self,
                                         scope_block: &src::ScopeBlock,
                                         pointers: &[src::VariableId]) {
        self.variable_scopes.push(HashMap::new());
        for (var_id, &(ref var_name, _)) in &scope_block.1.variables {
            let identifier = self.make_identifier();
            let map = self.variable_scopes.last_mut().expect("no scopes after pushing scope");
            map.insert(var_id.clone(),
                       VariableDecl(identifier,
                                    if pointers.iter().any(|pp| pp == var_id) {
                                        ParamType::Pointer
                                    } else {
                                        ParamType::Normal
                                    }));
            match self.local_names {
                Some(ref mut names) => names.insert(identifier, var_name.clone()),
                None => panic!("not inside function"),
            };
        }
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
            functions: HashMap::new(),
        };
        for (id, name) in self.global_ids.global_name_map {
            decls.globals.insert(id, name);
        }
        for (id, name) in self.global_ids.struct_name_map {
            decls.structs.insert(id, name);
        }
        for (id, name) in self.global_ids.function_name_map {
            decls.functions.insert(id, name);
        }
        (decls, self.global_ids.fragments)
    }
}

fn is_global_lifted(gv: &src::GlobalVariable) -> bool {
    let &src::GlobalType(src::Type(_, ref modifier), ref gs, _) = &gv.global_type;
    let static_const = modifier.is_const && *gs == src::GlobalStorage::Static;
    let groupshared = *gs == src::GlobalStorage::GroupShared;
    !static_const && !groupshared
}

fn get_cl_global_type(id: &src::GlobalId,
                      ty: &src::GlobalType,
                      usage: &globals_analysis::GlobalUsage,
                      context: &Context)
                      -> Result<dst::Type, TranspileError> {
    let tyl = &(ty.0).0;
    Ok(match *tyl {
        src::TypeLayout::Object(src::ObjectType::Buffer(ref data_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global,
                               Box::new(try!(transpile_datatype(data_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::StructuredBuffer(ref structured_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global,
                               Box::new(try!(transpile_structuredtype(structured_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::RWBuffer(ref data_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global,
                               Box::new(try!(transpile_datatype(data_type, &context))))
        }
        src::TypeLayout::Object(src::ObjectType::RWStructuredBuffer(ref structured_type)) => {
            dst::Type::Pointer(dst::AddressSpace::Global,
                               Box::new(try!(transpile_structuredtype(structured_type, &context))))
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
            dst::Type::Pointer(dst::AddressSpace::Global,
                               Box::new(dst::Type::Scalar(dst::Scalar::UChar)))
        }
        src::TypeLayout::Object(src::ObjectType::RWByteAddressBuffer) => {
            dst::Type::Pointer(dst::AddressSpace::Global,
                               Box::new(dst::Type::Scalar(dst::Scalar::UChar)))
        }
        _ => return Err(TranspileError::TypeIsNotAllowedAsGlobal(ty.clone())),
    })
}

fn get_cl_cbuffer_type(id: &src::ConstantBufferId,
                       context: &Context)
                       -> Result<dst::Type, TranspileError> {
    Ok(dst::Type::Pointer(dst::AddressSpace::Constant,
                          Box::new(dst::Type::Struct(try!(context.get_cbuffer_struct_name(id))))))
}

fn transpile_scalartype(scalartype: &src::ScalarType) -> Result<dst::Scalar, TranspileError> {
    match scalartype {
        &src::ScalarType::Bool => Err(TranspileError::BoolVectorsNotSupported),
        &src::ScalarType::Int => Ok(dst::Scalar::Int),
        &src::ScalarType::UInt => Ok(dst::Scalar::UInt),
        &src::ScalarType::Half => Ok(dst::Scalar::Half),
        &src::ScalarType::Float => Ok(dst::Scalar::Float),
        &src::ScalarType::Double => Ok(dst::Scalar::Double),
        &src::ScalarType::UntypedInt => Err(TranspileError::IntsMustBeTyped),
    }
}

fn transpile_datatype(datatype: &src::DataType,
                      context: &Context)
                      -> Result<dst::Type, TranspileError> {
    transpile_type(&src::Type::from(datatype.clone()), context)
}

fn transpile_structuredtype(structured_type: &src::StructuredType,
                            context: &Context)
                            -> Result<dst::Type, TranspileError> {
    transpile_type(&src::Type::from(structured_type.clone()), context)
}

fn transpile_typelayout(ty: &src::TypeLayout,
                        context: &Context)
                        -> Result<dst::Type, TranspileError> {
    match ty {
        &src::TypeLayout::Void => Ok(dst::Type::Void),
        &src::TypeLayout::Scalar(src::ScalarType::Bool) => Ok(dst::Type::Bool),
        &src::TypeLayout::Scalar(ref scalar) => {
            Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar))))
        }
        &src::TypeLayout::Vector(ref scalar, 1) => {
            Ok(dst::Type::Scalar(try!(transpile_scalartype(scalar))))
        }
        &src::TypeLayout::Vector(ref scalar, 2) => {
            Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)),
                                 dst::VectorDimension::Two))
        }
        &src::TypeLayout::Vector(ref scalar, 3) => {
            Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)),
                                 dst::VectorDimension::Three))
        }
        &src::TypeLayout::Vector(ref scalar, 4) => {
            Ok(dst::Type::Vector(try!(transpile_scalartype(scalar)),
                                 dst::VectorDimension::Four))
        }
        &src::TypeLayout::Struct(ref id) => {
            let struct_name = try!(context.get_struct_name(id));
            Ok(dst::Type::Struct(struct_name))
        }
        &src::TypeLayout::Array(ref element, ref dim) => {
            let inner_ty = try!(transpile_typelayout(element, context));
            Ok(dst::Type::Array(Box::new(inner_ty), *dim))
        }
        _ => return Err(TranspileError::CouldNotGetEquivalentType(ty.clone())),
    }
}

fn transpile_type(hlsltype: &src::Type, context: &Context) -> Result<dst::Type, TranspileError> {
    let &src::Type(ref ty, _) = hlsltype;
    transpile_typelayout(ty, context)
}

fn transpile_localtype(local_type: &src::LocalType,
                       context: &Context)
                       -> Result<dst::Type, TranspileError> {
    let &src::LocalType(ref ty, ref ls, ref modifiers) = local_type;
    match *modifiers {
        Some(_) => return Err(TranspileError::Unknown),
        None => {}
    };
    match *ls {
        src::LocalStorage::Local => transpile_type(ty, context),
        src::LocalStorage::Static => Err(TranspileError::Unknown),
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

fn write_unary(op: dst::UnaryOp, expr: dst::Expression) -> Result<dst::Expression, TranspileError> {
    Ok(dst::Expression::UnaryOperation(op, Box::new(expr)))
}

fn write_func(name: &'static str,
              args: &[dst::Expression])
              -> Result<dst::Expression, TranspileError> {
    let args_vec = args.into_iter()
                       .map(|e| e.clone())
                       .collect::<Vec<dst::Expression>>();
    Ok(dst::Expression::UntypedIntrinsic(name.to_string(), args_vec))
}

fn transpile_intrinsic0(intrinsic: &src::Intrinsic0,
                        _: &mut Context)
                        -> Result<dst::Expression, TranspileError> {
    use slp_lang_hir::Intrinsic0 as I;
    match *intrinsic {
        I::AllMemoryBarrier | I::AllMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::UntypedIntrinsic(
                "barrier".to_string(),
                vec![dst::Expression::BinaryOperation(dst::BinOp::BitwiseOr,
                    Box::new(dst::Expression::UntypedLiteral("CLK_LOCAL_MEM_FENCE".to_string())),
                    Box::new(dst::Expression::UntypedLiteral("CLK_GLOBAL_MEM_FENCE".to_string()))
                )]
            ))
        }
        I::DeviceMemoryBarrier | I::DeviceMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::UntypedIntrinsic(
                "barrier".to_string(),
                vec![dst::Expression::UntypedLiteral("CLK_GLOBAL_MEM_FENCE".to_string())]
            ))
        }
        I::GroupMemoryBarrier | I::GroupMemoryBarrierWithGroupSync => {
            Ok(dst::Expression::UntypedIntrinsic(
                "barrier".to_string(),
                vec![dst::Expression::UntypedLiteral("CLK_LOCAL_MEM_FENCE".to_string())]
            ))
        }
    }
}

fn transpile_intrinsic1(intrinsic: &src::Intrinsic1,
                        src_expr_1: &src::Expression,
                        context: &mut Context)
                        -> Result<dst::Expression, TranspileError> {
    use slp_lang_hir::Intrinsic1 as I;
    let e1 = try!(transpile_expression(src_expr_1, context));
    match *intrinsic {
        I::PrefixIncrement(_) => write_unary(dst::UnaryOp::PrefixIncrement, e1),
        I::PrefixDecrement(_) => write_unary(dst::UnaryOp::PrefixDecrement, e1),
        I::PostfixIncrement(_) => write_unary(dst::UnaryOp::PostfixIncrement, e1),
        I::PostfixDecrement(_) => write_unary(dst::UnaryOp::PostfixDecrement, e1),
        I::Plus(_) => write_unary(dst::UnaryOp::Plus, e1),
        I::Minus(_) => write_unary(dst::UnaryOp::Minus, e1),
        I::LogicalNot(_) => write_unary(dst::UnaryOp::LogicalNot, e1),
        I::BitwiseNot(_) => write_unary(dst::UnaryOp::BitwiseNot, e1),
        I::AsIntU => write_func("as_int", &[e1]),
        I::AsIntU2 => write_func("as_int2", &[e1]),
        I::AsIntU3 => write_func("as_int3", &[e1]),
        I::AsIntU4 => write_func("as_int4", &[e1]),
        I::AsIntF => write_func("as_int", &[e1]),
        I::AsIntF2 => write_func("as_int2", &[e1]),
        I::AsIntF3 => write_func("as_int3", &[e1]),
        I::AsIntF4 => write_func("as_int4", &[e1]),
        I::AsUIntI => write_func("as_uint", &[e1]),
        I::AsUIntI2 => write_func("as_uint2", &[e1]),
        I::AsUIntI3 => write_func("as_uint3", &[e1]),
        I::AsUIntI4 => write_func("as_uint4", &[e1]),
        I::AsUIntF => write_func("as_uint", &[e1]),
        I::AsUIntF2 => write_func("as_uint2", &[e1]),
        I::AsUIntF3 => write_func("as_uint3", &[e1]),
        I::AsUIntF4 => write_func("as_uint4", &[e1]),
        I::AsFloatI => write_func("as_float", &[e1]),
        I::AsFloatI2 => write_func("as_float2", &[e1]),
        I::AsFloatI3 => write_func("as_float3", &[e1]),
        I::AsFloatI4 => write_func("as_float4", &[e1]),
        I::AsFloatU => write_func("as_float", &[e1]),
        I::AsFloatU2 => write_func("as_float2", &[e1]),
        I::AsFloatU3 => write_func("as_float3", &[e1]),
        I::AsFloatU4 => write_func("as_float4", &[e1]),
        I::AsFloatF => write_func("as_float", &[e1]),
        I::AsFloatF2 => write_func("as_float2", &[e1]),
        I::AsFloatF3 => write_func("as_float3", &[e1]),
        I::AsFloatF4 => write_func("as_float4", &[e1]),
        I::F16ToF32 => {
            context.required_extensions.insert(dst::Extension::KhrFp16);
            let input_16u = try!(write_cast(dst::Type::Scalar(dst::Scalar::UInt),
                                            dst::Type::Scalar(dst::Scalar::UShort),
                                            e1,
                                            false,
                                            context));
            let as_half = try!(write_func("as_half", &[input_16u]));
            write_cast(dst::Type::Scalar(dst::Scalar::Half),
                       dst::Type::Scalar(dst::Scalar::Float),
                       as_half,
                       false,
                       context)
        }
        I::F32ToF16 => {
            context.required_extensions.insert(dst::Extension::KhrFp16);
            let as_half = try!(write_cast(dst::Type::Scalar(dst::Scalar::Float),
                                          dst::Type::Scalar(dst::Scalar::Half),
                                          e1,
                                          false,
                                          context));
            let as_ushort = try!(write_func("as_ushort", &[as_half]));
            write_cast(dst::Type::Scalar(dst::Scalar::UShort),
                       dst::Type::Scalar(dst::Scalar::UInt),
                       as_ushort,
                       false,
                       context)
        }
        I::Floor |
        I::Floor2 |
        I::Floor3 |
        I::Floor4 => write_func("floor", &[e1]),
        I::IsNaN => {
            // OpenCL isnan returns 1 for true or 0 for false
            // Cast to bool to get boolean result
            write_cast(dst::Type::Scalar(dst::Scalar::Int),
                       dst::Type::Bool,
                       try!(write_func("isnan", &[e1])),
                       false,
                       context)
        }
        I::IsNaN2 => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        I::IsNaN3 => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        I::IsNaN4 => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        I::Length1 |
        I::Length2 |
        I::Length3 |
        I::Length4 => write_func("length", &[e1]),
        I::Normalize1 |
        I::Normalize2 |
        I::Normalize3 |
        I::Normalize4 => write_func("normalize", &[e1]),
        I::SignI => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        I::SignI2 => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        I::SignI3 => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        I::SignI4 => Err(TranspileError::Intrinsic1Unimplemented(intrinsic.clone())),
        // HLSL sign returns int, OpenCL sign returns float (with different results for negative
        // and positive zero)
        // Cast to int to get the same behavior
        I::SignF => {
            write_cast(dst::Type::Scalar(dst::Scalar::Float),
                       dst::Type::Scalar(dst::Scalar::Int),
                       try!(write_func("sign", &[e1])),
                       false,
                       context)
        }
        I::SignF2 => {
            write_cast(dst::Type::Vector(dst::Scalar::Float, dst::VectorDimension::Two),
                       dst::Type::Vector(dst::Scalar::Int, dst::VectorDimension::Two),
                       try!(write_func("sign", &[e1])),
                       false,
                       context)
        }
        I::SignF3 => {
            write_cast(dst::Type::Vector(dst::Scalar::Float, dst::VectorDimension::Three),
                       dst::Type::Vector(dst::Scalar::Int, dst::VectorDimension::Three),
                       try!(write_func("sign", &[e1])),
                       false,
                       context)
        }
        I::SignF4 => {
            write_cast(dst::Type::Vector(dst::Scalar::Float, dst::VectorDimension::Four),
                       dst::Type::Vector(dst::Scalar::Int, dst::VectorDimension::Four),
                       try!(write_func("sign", &[e1])),
                       false,
                       context)
        }
        I::Sqrt |
        I::Sqrt2 |
        I::Sqrt3 |
        I::Sqrt4 => write_func("sqrt", &[e1]),
    }
}

fn write_binop(binop: dst::BinOp,
               lhs: dst::Expression,
               rhs: dst::Expression)
               -> Result<dst::Expression, TranspileError> {
    Ok(dst::Expression::BinaryOperation(binop, Box::new(lhs), Box::new(rhs)))
}

fn transpile_intrinsic2(intrinsic: &src::Intrinsic2,
                        src_expr_1: &src::Expression,
                        src_expr_2: &src::Expression,
                        context: &mut Context)
                        -> Result<dst::Expression, TranspileError> {
    use slp_lang_hir::Intrinsic2 as I;
    let e1 = try!(transpile_expression(src_expr_1, context));
    let e2 = try!(transpile_expression(src_expr_2, context));
    match *intrinsic {
        I::Add(_) => write_binop(dst::BinOp::Add, e1, e2),
        I::Subtract(_) => write_binop(dst::BinOp::Subtract, e1, e2),
        I::Multiply(_) => write_binop(dst::BinOp::Multiply, e1, e2),
        I::Divide(_) => write_binop(dst::BinOp::Divide, e1, e2),
        I::Modulus(_) => write_binop(dst::BinOp::Modulus, e1, e2),
        I::LeftShift(_) => write_binop(dst::BinOp::LeftShift, e1, e2),
        I::RightShift(_) => write_binop(dst::BinOp::RightShift, e1, e2),
        I::BitwiseAnd(_) => write_binop(dst::BinOp::BitwiseAnd, e1, e2),
        I::BitwiseOr(_) => write_binop(dst::BinOp::BitwiseOr, e1, e2),
        I::BitwiseXor(_) => write_binop(dst::BinOp::BitwiseXor, e1, e2),
        I::BooleanAnd(_) => write_binop(dst::BinOp::LogicalAnd, e1, e2),
        I::BooleanOr(_) => write_binop(dst::BinOp::LogicalOr, e1, e2),
        I::LessThan(_) => write_binop(dst::BinOp::LessThan, e1, e2),
        I::LessEqual(_) => write_binop(dst::BinOp::LessEqual, e1, e2),
        I::GreaterThan(_) => write_binop(dst::BinOp::GreaterThan, e1, e2),
        I::GreaterEqual(_) => write_binop(dst::BinOp::GreaterEqual, e1, e2),
        I::Equality(_) => write_binop(dst::BinOp::Equality, e1, e2),
        I::Inequality(_) => write_binop(dst::BinOp::Inequality, e1, e2),
        I::Assignment(_) => write_binop(dst::BinOp::Assignment, e1, e2),
        I::SumAssignment(_) => write_binop(dst::BinOp::SumAssignment, e1, e2),
        I::DifferenceAssignment(_) => write_binop(dst::BinOp::DifferenceAssignment, e1, e2),
        I::AsDouble => Err(TranspileError::Intrinsic2Unimplemented(intrinsic.clone())),
        I::Cross => write_func("cross", &[e1, e2]),
        I::Distance1 |
        I::Distance2 |
        I::Distance3 |
        I::Distance4 => {
            Ok(dst::Expression::UntypedIntrinsic(
                "length".to_string(),
                vec![dst::Expression::BinaryOperation(dst::BinOp::Subtract,
                    Box::new(e1),
                    Box::new(e2)
                )]
            ))
        }
        I::DotI1 => Err(TranspileError::Intrinsic2Unimplemented(intrinsic.clone())),
        I::DotI2 => Err(TranspileError::Intrinsic2Unimplemented(intrinsic.clone())),
        I::DotI3 => Err(TranspileError::Intrinsic2Unimplemented(intrinsic.clone())),
        I::DotI4 => Err(TranspileError::Intrinsic2Unimplemented(intrinsic.clone())),
        I::DotF1 |
        I::DotF2 |
        I::DotF3 |
        I::DotF4 => write_func("dot", &[e1, e2]),
        I::MinI |
        I::MinI2 |
        I::MinI3 |
        I::MinI4 => write_func("min", &[e1, e2]),
        I::MinF |
        I::MinF2 |
        I::MinF3 |
        I::MinF4 => write_func("fmin", &[e1, e2]),
        I::MaxI |
        I::MaxI2 |
        I::MaxI3 |
        I::MaxI4 => write_func("max", &[e1, e2]),
        I::MaxF |
        I::MaxF2 |
        I::MaxF3 |
        I::MaxF4 => write_func("fmax", &[e1, e2]),
        I::BufferLoad(_) |
        I::RWBufferLoad(_) |
        I::StructuredBufferLoad(_) |
        I::RWStructuredBufferLoad(_) => {
            Ok(dst::Expression::ArraySubscript(Box::new(e1), Box::new(e2)))
        }
        I::RWTexture2DLoad(ref data_type) => {
            let (func_name, read_type) = match data_type.0 {
                src::DataLayout::Scalar(ref scalar) | src::DataLayout::Vector(ref scalar, _) => {
                    let dim = dst::VectorDimension::Four;
                    match *scalar {
                        src::ScalarType::Int => {
                            ("read_imagei", dst::Type::Vector(dst::Scalar::Int, dim))
                        }
                        src::ScalarType::UInt => {
                            ("read_imageui", dst::Type::Vector(dst::Scalar::UInt, dim))
                        }
                        src::ScalarType::Float => {
                            ("read_imagef", dst::Type::Vector(dst::Scalar::Float, dim))
                        }
                        _ => return Err(TranspileError::Unknown),
                    }
                }
                src::DataLayout::Matrix(_, _, _) => return Err(TranspileError::Unknown),
            };
            let cast_type = try!(transpile_datatype(data_type, context));
            let expr = dst::Expression::UntypedIntrinsic(func_name.to_string(), vec![e1, e2]);
            write_cast(read_type, cast_type, expr, false, context)
        }
        I::ByteAddressBufferLoad |
        I::RWByteAddressBufferLoad |
        I::ByteAddressBufferLoad2 |
        I::RWByteAddressBufferLoad2 |
        I::ByteAddressBufferLoad3 |
        I::RWByteAddressBufferLoad3 |
        I::ByteAddressBufferLoad4 |
        I::RWByteAddressBufferLoad4 => {
            let ty = Box::new(match *intrinsic {
                I::ByteAddressBufferLoad |
                I::RWByteAddressBufferLoad => dst::Type::Scalar(dst::Scalar::UInt),
                I::ByteAddressBufferLoad2 |
                I::RWByteAddressBufferLoad2 => {
                    dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Two)
                }
                I::ByteAddressBufferLoad3 |
                I::RWByteAddressBufferLoad3 => {
                    dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Three)
                }
                I::ByteAddressBufferLoad4 |
                I::RWByteAddressBufferLoad4 => {
                    dst::Type::Vector(dst::Scalar::UInt, dst::VectorDimension::Four)
                }
                _ => unreachable!(),
            });
            Ok(dst::Expression::Deref(Box::new(dst::Expression::Cast(
                dst::Type::Pointer(dst::AddressSpace::Global, ty),
                Box::new(dst::Expression::BinaryOperation(dst::BinOp::Add, Box::new(e1), Box::new(e2)))
            ))))
        }
    }
}
fn transpile_intrinsic3(intrinsic: &src::Intrinsic3,
                        src_expr_1: &src::Expression,
                        src_expr_2: &src::Expression,
                        src_expr_3: &src::Expression,
                        context: &mut Context)
                        -> Result<dst::Expression, TranspileError> {
    use slp_lang_hir::Intrinsic3 as I;
    let e1 = try!(transpile_expression(src_expr_1, context));
    let e2 = try!(transpile_expression(src_expr_2, context));
    let e3 = try!(transpile_expression(src_expr_3, context));
    match *intrinsic {
        I::ClampI => write_func("clamp", &[e1, e2, e3]),
        I::ClampI2 => write_func("clamp", &[e1, e2, e3]),
        I::ClampI3 => write_func("clamp", &[e1, e2, e3]),
        I::ClampI4 => write_func("clamp", &[e1, e2, e3]),
        I::ClampF => write_func("clamp", &[e1, e2, e3]),
        I::ClampF2 => write_func("clamp", &[e1, e2, e3]),
        I::ClampF3 => write_func("clamp", &[e1, e2, e3]),
        I::ClampF4 => write_func("clamp", &[e1, e2, e3]),
        I::RWByteAddressBufferStore |
        I::RWByteAddressBufferStore2 |
        I::RWByteAddressBufferStore3 |
        I::RWByteAddressBufferStore4 => {
            let st = dst::Scalar::UInt;
            let ty = Box::new(match *intrinsic {
                I::RWByteAddressBufferStore => dst::Type::Scalar(st),
                I::RWByteAddressBufferStore2 => dst::Type::Vector(st, dst::VectorDimension::Two),
                I::RWByteAddressBufferStore3 => dst::Type::Vector(st, dst::VectorDimension::Three),
                I::RWByteAddressBufferStore4 => dst::Type::Vector(st, dst::VectorDimension::Four),
                _ => unreachable!(),
            });
            let binop = dst::Expression::BinaryOperation(dst::BinOp::Add,
                                                         Box::new(e1),
                                                         Box::new(e2));
            Ok(dst::Expression::BinaryOperation(
                dst::BinOp::Assignment,
                Box::new(dst::Expression::Deref(Box::new(dst::Expression::Cast(
                    dst::Type::Pointer(dst::AddressSpace::Global, ty),
                    Box::new(binop)
                )))),
                Box::new(e3)
            ))
        }
    }
}

fn write_cast(source_cl_type: dst::Type,
              dest_cl_type: dst::Type,
              cl_expr: dst::Expression,
              always_cast: bool,
              context: &mut Context)
              -> Result<dst::Expression, TranspileError> {
    if dest_cl_type == source_cl_type && !always_cast {
        // If the cast would cast to the same time, ignore it
        return Ok(cl_expr);
    }
    Ok(match dest_cl_type {
        dst::Type::Bool => {
            match source_cl_type {
                // Vector to bool cast
                dst::Type::Vector(_, _) => {
                    dst::Expression::Cast(dest_cl_type.clone(), Box::new(dst::Expression::Swizzle(Box::new(cl_expr), vec![dst::SwizzleSlot::X])))
                }
                // Scalar to bool cast
                dst::Type::Bool | dst::Type::Scalar(_) => {
                    dst::Expression::Cast(dest_cl_type, Box::new(cl_expr))
                }
                _ => {
                    return Err(TranspileError::Internal("source of bool cast is not a numeric \
                                                         type"
                                                            .to_string()))
                }
            }
        }
        dst::Type::Scalar(ref to_scalar_type) => {
            match source_cl_type {
                // Vector to same type scalar cast, swizzle
                dst::Type::Vector(ref from_scalar_type, _) if *from_scalar_type ==
                                                              *to_scalar_type => {
                    dst::Expression::Swizzle(Box::new(cl_expr), vec![dst::SwizzleSlot::X])
                }
                // Vector to scalar cast, swizzle + cast
                dst::Type::Vector(_, _) => {
                    dst::Expression::Cast(dest_cl_type.clone(), Box::new(dst::Expression::Swizzle(Box::new(cl_expr), vec![dst::SwizzleSlot::X])))
                }
                // Scalar to scalar cast
                dst::Type::Bool | dst::Type::Scalar(_) => {
                    dst::Expression::Cast(dest_cl_type.clone(), Box::new(cl_expr))
                }
                _ => {
                    return Err(TranspileError::Internal("source of scalar cast is not a numeric \
                                                         type"
                                                            .to_string()))
                }
            }
        }
        dst::Type::Vector(ref scalar, ref to_dim) => {
            match source_cl_type {
                // Vector to same type vector cast, swizzle
                dst::Type::Vector(ref from_scalar_type, ref from_dim) if *from_scalar_type ==
                                                                         *scalar => {
                    assert!(to_dim.as_u32() <= from_dim.as_u32(),
                            "{:?} <= {:?}",
                            to_dim,
                            from_dim);
                    let swizzle = match *to_dim {
                        dst::VectorDimension::Two => vec![dst::SwizzleSlot::X, dst::SwizzleSlot::Y],
                        dst::VectorDimension::Three => {
                            vec![dst::SwizzleSlot::X, dst::SwizzleSlot::Y, dst::SwizzleSlot::Z]
                        }
                        dst::VectorDimension::Four => {
                            panic!("4 element vectors can not be downcast from anything")
                        }
                        _ => panic!("casting from {:?} to {:?}", source_cl_type, dest_cl_type),
                    };
                    dst::Expression::Swizzle(Box::new(cl_expr), swizzle)
                }
                // Vector to different type vector cast, make a function to do the work
                dst::Type::Vector(from_scalar_type, from_dim) => {
                    let cast_func_id =
                        context.fetch_fragment(Fragment::VectorCast(from_scalar_type,
                                                                    scalar.clone(),
                                                                    from_dim,
                                                                    to_dim.clone()));
                    dst::Expression::Call(cast_func_id, vec![cl_expr])
                }
                // Scalar to vector cast, make a function to do the work
                dst::Type::Scalar(from_scalar_type) => {
                    let cast_func_id =
                        context.fetch_fragment(Fragment::ScalarToVectorCast(from_scalar_type,
                                                                            scalar.clone(),
                                                                            to_dim.clone()));
                    dst::Expression::Call(cast_func_id, vec![cl_expr])
                }
                dst::Type::Bool => unimplemented!(),
                _ => {
                    return Err(TranspileError::Internal("source of vector cast is not a numeric \
                                                         type"
                                                            .to_string()))
                }
            }
        }
        _ => {
            return Err(TranspileError::Internal(format!("don't know how to cast to this type \
                                                         ({:?})",
                                                        dest_cl_type)))
        }
    })
}

fn transpile_expression(expression: &src::Expression,
                        context: &mut Context)
                        -> Result<dst::Expression, TranspileError> {
    match expression {
        &src::Expression::Literal(ref lit) => {
            Ok(dst::Expression::Literal(try!(transpile_literal(lit))))
        }
        &src::Expression::Variable(ref var_ref) => context.get_variable_ref(var_ref),
        &src::Expression::Global(ref id) => context.get_global_var(id),
        &src::Expression::ConstantVariable(ref id, ref name) => {
            context.get_constant(id, name.clone())
        }
        &src::Expression::TernaryConditional(ref cond, ref lhs, ref rhs) => {
            let cl_cond = Box::new(try!(transpile_expression(cond, context)));
            let cl_lhs = Box::new(try!(transpile_expression(lhs, context)));
            let cl_rhs = Box::new(try!(transpile_expression(rhs, context)));
            Ok(dst::Expression::TernaryConditional(cl_cond, cl_lhs, cl_rhs))
        }
        &src::Expression::Swizzle(ref vec, ref swizzle) => {
            Ok(dst::Expression::Swizzle(Box::new(try!(transpile_expression(vec, context))),
                                        swizzle.iter()
                                               .map(|swizzle_slot| {
                                                   match *swizzle_slot {
                                                       src::SwizzleSlot::X => dst::SwizzleSlot::X,
                                                       src::SwizzleSlot::Y => dst::SwizzleSlot::Y,
                                                       src::SwizzleSlot::Z => dst::SwizzleSlot::Z,
                                                       src::SwizzleSlot::W => dst::SwizzleSlot::W,
                                                   }
                                               })
                                               .collect::<Vec<_>>()))
        }
        &src::Expression::ArraySubscript(ref expr, ref sub) => {
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            let cl_sub = Box::new(try!(transpile_expression(sub, context)));
            Ok(dst::Expression::ArraySubscript(cl_expr, cl_sub))
        }
        &src::Expression::Member(ref expr, ref member_name) => {
            let cl_expr = Box::new(try!(transpile_expression(expr, context)));
            Ok(dst::Expression::Member(cl_expr, member_name.clone()))
        }
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
            }
            let mut final_arguments = vec![];
            for global in &decl.additional_arguments {
                final_arguments.push(dst::Expression::Local(match *global {
                    GlobalArgument::Global(ref id) => try!(context.get_global_lifted_id(id)),
                    GlobalArgument::ConstantBuffer(ref id) => {
                        try!(context.get_cbuffer_instance_id(id))
                    }
                }));
            }
            final_arguments.append(&mut params_exprs);
            Ok(dst::Expression::Call(func_expr, final_arguments))
        }
        &src::Expression::NumericConstructor(ref dtyl, ref slots) => {

            // Generate target constructor type
            let tty = src::TypeLayout::from_data(dtyl.clone());
            let sty = dtyl.to_scalar();
            // Will fail for bools. TODO: bools -> scalars implementation trivial, vectors very hard
            match sty {
                src::ScalarType::Bool => return Err(TranspileError::BoolVectorsNotSupported),
                _ => {}
            };
            let dst_scalar = try!(transpile_scalartype(&sty));
            let dst_dim = match try!(transpile_typelayout(&tty, context)) {
                dst::Type::Scalar(ref scalar) => {
                    assert_eq!(dst_scalar, *scalar);
                    dst::NumericDimension::Scalar
                }
                dst::Type::Vector(ref scalar, ref dim) => {
                    assert_eq!(dst_scalar, *scalar);
                    dst::NumericDimension::Vector(dim.clone())
                }
                _ => panic!("not numeric type created from data type"),
            };

            // Transpile arguments
            let mut arguments: Vec<dst::Expression> = vec![];
            for cons in slots {
                let dst_expr = try!(transpile_expression(&cons.expr, context));
                arguments.push(dst_expr);
            }

            Ok(dst::Expression::NumericConstructor(dst_scalar, dst_dim, arguments))
        }
        &src::Expression::Cast(ref cast_type, ref expr) => {
            let dest_cl_type = try!(transpile_type(cast_type, context));
            let cl_expr = try!(transpile_expression(expr, context));
            let source_ir_type = try!(src::TypeParser::get_expression_type(expr,
                                                                           &context.type_context))
                                     .0;
            let (source_cl_type, untyped) = match transpile_type(&source_ir_type, context) {
                Ok(ty) => (ty, false),
                Err(TranspileError::IntsMustBeTyped) => {
                    // Force untyped int literals to be treated as normal ints
                    (dst::Type::Scalar(dst::Scalar::Int), true)
                }
                Err(err) => return Err(err),
            };
            write_cast(source_cl_type, dest_cl_type, cl_expr, untyped, context)
        }
        &src::Expression::Intrinsic0(ref i) => transpile_intrinsic0(i, context),
        &src::Expression::Intrinsic1(ref i, ref e1) => transpile_intrinsic1(i, e1, context),
        &src::Expression::Intrinsic2(ref i, ref e1, ref e2) => {
            transpile_intrinsic2(i, e1, e2, context)
        }
        &src::Expression::Intrinsic3(ref i, ref e1, ref e2, ref e3) => {
            transpile_intrinsic3(i, e1, e2, e3, context)
        }
    }
}

fn transpile_vardef(vardef: &src::VarDef,
                    context: &mut Context)
                    -> Result<dst::VarDef, TranspileError> {
    Ok(dst::VarDef {
        id: try!(context.get_variable_id(&vardef.id)),
        typename: try!(transpile_localtype(&vardef.local_type, context)),
        assignment: match &vardef.assignment {
            &None => None,
            &Some(ref expr) => Some(try!(transpile_expression(expr, context))),
        },
    })
}

fn transpile_statement(statement: &src::Statement,
                       context: &mut Context)
                       -> Result<Vec<dst::Statement>, TranspileError> {
    match statement {
        &src::Statement::Expression(ref expr) => {
            Ok(vec![dst::Statement::Expression(try!(transpile_expression(expr, context)))])
        }
        &src::Statement::Var(ref vd) => {
            Ok(vec![dst::Statement::Var(try!(transpile_vardef(vd, context)))])
        }
        &src::Statement::Block(ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(vec![dst::Statement::Block(cl_statements)])
        }
        &src::Statement::If(ref cond, ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(vec![dst::Statement::If(cl_cond, Box::new(dst::Statement::Block(cl_statements)))])
        }
        &src::Statement::IfElse(ref cond, ref true_block, ref false_block) => {
            let &src::ScopeBlock(ref true_statements, _) = true_block;
            let &src::ScopeBlock(ref false_statements, _) = false_block;
            // Condition
            let cl_cond = try!(transpile_expression(cond, context));
            // True part
            context.push_scope(true_block);
            let cl_true_statements = try!(transpile_statements(true_statements, context));
            context.pop_scope();
            // False part
            context.push_scope(false_block);
            let cl_false_statements = try!(transpile_statements(false_statements, context));
            context.pop_scope();
            // Combine
            Ok(vec![dst::Statement::IfElse(cl_cond,
                                           Box::new(dst::Statement::Block(cl_true_statements)),
                                           Box::new(dst::Statement::Block(cl_false_statements)))])
        }
        &src::Statement::For(ref init, ref cond, ref update, ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);

            let (cl_init, defs) = match *init {
                src::ForInit::Expression(ref expr) => {
                    let expr_ir = try!(transpile_expression(expr, context));
                    (dst::InitStatement::Expression(expr_ir), vec![])
                }
                src::ForInit::Definitions(ref vds) => {
                    assert!(vds.len() > 0);
                    let mut vardefs = vec![];
                    for vd in vds {
                        vardefs.push(try!(transpile_vardef(vd, context)));
                    }
                    let last_vardef = vardefs.pop().expect("zero variable definitions in for init");
                    (dst::InitStatement::Declaration(last_vardef), vardefs)
                }
            };

            let cl_cond = try!(transpile_expression(cond, context));
            let cl_update = try!(transpile_expression(update, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            let cl_for = dst::Statement::For(cl_init,
                                             cl_cond,
                                             cl_update,
                                             Box::new(dst::Statement::Block(cl_statements)));
            let mut block_contents = defs.into_iter()
                                         .map(|d| dst::Statement::Var(d))
                                         .collect::<Vec<_>>();
            block_contents.push(cl_for);
            Ok(block_contents)
        }
        &src::Statement::While(ref cond, ref scope_block) => {
            let &src::ScopeBlock(ref statements, _) = scope_block;
            context.push_scope(scope_block);
            let cl_cond = try!(transpile_expression(cond, context));
            let cl_statements = try!(transpile_statements(statements, context));
            context.pop_scope();
            Ok(vec![dst::Statement::While(cl_cond, Box::new(dst::Statement::Block(cl_statements)))])
        }
        &src::Statement::Return(ref expr) => {
            Ok(vec![dst::Statement::Return(try!(transpile_expression(expr, context)))])
        }
    }
}

fn transpile_statements(statements: &[src::Statement],
                        context: &mut Context)
                        -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_statements = vec![];
    for statement in statements {
        cl_statements.append(&mut try!(transpile_statement(statement, context)));
    }
    Ok(cl_statements)
}

fn transpile_param(param: &src::FunctionParam,
                   context: &mut Context)
                   -> Result<dst::FunctionParam, TranspileError> {
    let &src::ParamType(ref ty_ast, ref it, ref interp) = &param.param_type;
    let ty = match *it {
        src::InputModifier::In => try!(transpile_type(ty_ast, context)),
        src::InputModifier::Out | src::InputModifier::InOut => {
            // Only allow out params to work on Private address space
            // as we don't support generating multiple function for each
            // address space (and can't use __generic as it's 2.0)
            dst::Type::Pointer(dst::AddressSpace::Private,
                               Box::new(try!(transpile_type(ty_ast, context))))
        }
    };
    match *interp {
        Some(_) => return Err(TranspileError::Unknown),
        None => {}
    };
    Ok(dst::FunctionParam {
        id: try!(context.get_variable_id(&param.id)),
        typename: ty,
    })
}

fn transpile_params(params: &[src::FunctionParam],
                    context: &mut Context)
                    -> Result<Vec<dst::FunctionParam>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_param(param, context)));
    }
    Ok(cl_params)
}

fn transpile_kernel_input_semantic(param: &src::KernelParam,
                                   group_dim: &dst::Dimension,
                                   context: &mut Context)
                                   -> Result<dst::Statement, TranspileError> {
    let typename = try!(transpile_type(&param.1.get_type(), context));
    fn lit(i: u64) -> Box<dst::Expression> {
        Box::new(dst::Expression::Literal(dst::Literal::UInt(i)))
    }
    let assign = match param.1 {
        src::KernelSemantic::DispatchThreadId => {
            let x = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(lit(0)));
            let y = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(lit(1)));
            let z = dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(lit(2)));
            let sty = dst::Scalar::UInt;
            let dim = dst::NumericDimension::Vector(dst::VectorDimension::Three);
            dst::Expression::NumericConstructor(sty, dim, vec![x, y, z])
        }
        src::KernelSemantic::GroupId => {
            let x_g = Box::new(dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(lit(0))));
            let y_g = Box::new(dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(lit(1))));
            let z_g = Box::new(dst::Expression::Intrinsic(dst::Intrinsic::GetGlobalId(lit(2))));
            let x = dst::Expression::BinaryOperation(dst::BinOp::Divide, x_g, lit(group_dim.0));
            let y = dst::Expression::BinaryOperation(dst::BinOp::Divide, y_g, lit(group_dim.1));
            let z = dst::Expression::BinaryOperation(dst::BinOp::Divide, z_g, lit(group_dim.2));
            let sty = dst::Scalar::UInt;
            let dim = dst::NumericDimension::Vector(dst::VectorDimension::Three);
            dst::Expression::NumericConstructor(sty, dim, vec![x, y, z])
        }
        src::KernelSemantic::GroupIndex => {
            let x = dst::Expression::Intrinsic(dst::Intrinsic::GetLocalId(lit(0)));
            let yt = dst::Expression::Intrinsic(dst::Intrinsic::GetLocalId(lit(1)));
            let zt = dst::Expression::Intrinsic(dst::Intrinsic::GetLocalId(lit(2)));
            let dim_x = lit(group_dim.0);
            let dim_xy = lit(group_dim.0 * group_dim.1);
            let y = dst::Expression::BinaryOperation(dst::BinOp::Multiply, Box::new(yt), dim_x);
            let z = dst::Expression::BinaryOperation(dst::BinOp::Multiply, Box::new(zt), dim_xy);
            let y_z = dst::Expression::BinaryOperation(dst::BinOp::Add, Box::new(y), Box::new(z));
            dst::Expression::BinaryOperation(dst::BinOp::Add, Box::new(x), Box::new(y_z))
        }
        src::KernelSemantic::GroupThreadId => {
            let x = dst::Expression::Intrinsic(dst::Intrinsic::GetLocalId(lit(0)));
            let y = dst::Expression::Intrinsic(dst::Intrinsic::GetLocalId(lit(1)));
            let z = dst::Expression::Intrinsic(dst::Intrinsic::GetLocalId(lit(2)));
            let sty = dst::Scalar::UInt;
            let dim = dst::NumericDimension::Vector(dst::VectorDimension::Three);
            dst::Expression::NumericConstructor(sty, dim, vec![x, y, z])
        }
    };
    Ok(dst::Statement::Var(dst::VarDef {
        id: try!(context.get_variable_id(&param.0)),
        typename: typename,
        assignment: Some(assign),
    }))
}

fn transpile_kernel_input_semantics(params: &[src::KernelParam],
                                    group_dim: &dst::Dimension,
                                    context: &mut Context)
                                    -> Result<Vec<dst::Statement>, TranspileError> {
    let mut cl_params = vec![];
    for param in params {
        cl_params.push(try!(transpile_kernel_input_semantic(param, group_dim, context)));
    }
    Ok(cl_params)
}

fn transpile_structdefinition(structdefinition: &src::StructDefinition,
                              context: &mut Context)
                              -> Result<dst::RootDefinition, TranspileError> {
    Ok(dst::RootDefinition::Struct(dst::StructDefinition {
        id: try!(context.get_struct_name(&structdefinition.id)),
        members: try!(structdefinition.members
                                      .iter()
                                      .fold(Ok(vec![]), |result: Result<Vec<dst::StructMember>,
                                                            TranspileError>,
                                             member| {
                                          let mut vec = try!(result);
                                          let dst_member = dst::StructMember {
                                              name: member.name.clone(),
                                              typename: try!(transpile_type(&member.typename,
                                                                            context)),
                                          };
                                          vec.push(dst_member);
                                          Ok(vec)
                                      })),
    }))
}

fn transpile_cbuffer(cb: &src::ConstantBuffer,
                     context: &mut Context)
                     -> Result<dst::RootDefinition, TranspileError> {
    let mut members = vec![];
    for member in &cb.members {
        let var_name = member.name.clone();
        let var_type = try!(transpile_type(&member.typename, context));
        members.push(dst::StructMember {
            name: var_name,
            typename: var_type,
        });
    }
    Ok(dst::RootDefinition::Struct(dst::StructDefinition {
        id: try!(context.get_cbuffer_struct_name(&cb.id)),
        members: members,
    }))
}

fn transpile_globalvariable(gv: &src::GlobalVariable,
                            context: &mut Context)
                            -> Result<Option<dst::RootDefinition>, TranspileError> {
    let &src::GlobalType(src::Type(ref ty, ref modifiers), _, _) = &gv.global_type;
    let lifted = is_global_lifted(gv);
    if lifted {
        return Ok(None);
    } else {
        let global_id = try!(context.get_global_static_id(&gv.id));
        let cl_type = try!(transpile_type(&src::Type(ty.clone(), modifiers.clone()), &context));
        let cl_init = match &gv.assignment {
            &Some(ref expr) => Some(try!(transpile_expression(expr, context))),
            &None => None,
        };
        let address_space = match gv.global_type.1 {
            src::GlobalStorage::Extern => panic!("extern not lifted"),
            src::GlobalStorage::Static => dst::AddressSpace::Constant,
            src::GlobalStorage::GroupShared => dst::AddressSpace::Local,
        };
        Ok(Some(dst::RootDefinition::GlobalVariable(dst::GlobalVariable {
            id: global_id,
            ty: cl_type,
            address_space: address_space,
            init: cl_init,
        })))
    }
}

fn transpile_functiondefinition(func: &src::FunctionDefinition,
                                context: &mut Context)
                                -> Result<dst::RootDefinition, TranspileError> {
    // Find the parameters that need to be turned into pointers
    // so the context can deref them
    let out_params = func.params.iter().fold(vec![], |mut out_params, param| {
        match param.param_type.1 {
            src::InputModifier::InOut | src::InputModifier::Out => {
                out_params.push(param.id);
            }
            src::InputModifier::In => {}
        };
        out_params
    });
    context.push_scope_for_function(&func.scope_block, &out_params, &func.id);
    let (_, decl) = try!(context.get_function(&func.id));
    let mut params = vec![];
    for global in &decl.additional_arguments {
        params.push(match *global {
            GlobalArgument::Global(ref id) => {
                dst::FunctionParam {
                    id: try!(context.get_global_lifted_id(id)),
                    typename: context.global_type_map.get(id).unwrap().clone(),
                }
            }
            GlobalArgument::ConstantBuffer(ref id) => {
                dst::FunctionParam {
                    id: try!(context.get_cbuffer_instance_id(id)),
                    typename: try!(get_cl_cbuffer_type(id, context)),
                }
            }
        });
    }
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

fn transpile_kernel(kernel: &src::Kernel,
                    context: &mut Context)
                    -> Result<dst::RootDefinition, TranspileError> {
    context.push_scope_for_kernel(&kernel.scope_block);
    let mut params = vec![];
    for global in &context.kernel_arguments {
        params.push(match *global {
            GlobalArgument::Global(ref id) => {
                dst::KernelParam {
                    id: try!(context.get_global_lifted_id(id)),
                    typename: match context.global_type_map.get(id) {
                        Some(ty) => ty.clone(),
                        None => panic!("global type unknown {:?}", id.clone()),
                    },
                }
            }
            GlobalArgument::ConstantBuffer(ref id) => {
                dst::KernelParam {
                    id: try!(context.get_cbuffer_instance_id(id)),
                    typename: try!(get_cl_cbuffer_type(id, context)),
                }
            }
        });
    }
    let group_dim = dst::Dimension(kernel.group_dimensions.0,
                                   kernel.group_dimensions.1,
                                   kernel.group_dimensions.2);
    let mut body = try!(transpile_kernel_input_semantics(&kernel.params, &group_dim, context));
    let mut main_body = try!(transpile_statements(&kernel.scope_block.0, context));
    let decls = context.pop_scope_for_function();
    body.append(&mut main_body);
    let cl_kernel = dst::Kernel {
        params: params,
        body: body,
        group_dimensions: group_dim,
        local_declarations: decls,
    };
    Ok(dst::RootDefinition::Kernel(cl_kernel))
}

fn transpile_roots(root_defs: &[src::RootDefinition],
                   context: &mut Context)
                   -> Result<Vec<dst::RootDefinition>, TranspileError> {
    let mut cl_defs = vec![];

    for rootdef in root_defs {
        match *rootdef {
            src::RootDefinition::Struct(ref structdefinition) => {
                cl_defs.push(try!(transpile_structdefinition(structdefinition, context)));
            }
            src::RootDefinition::ConstantBuffer(ref cb) => {
                cl_defs.push(try!(transpile_cbuffer(cb, context)));
            }
            src::RootDefinition::GlobalVariable(ref gv) => {
                match try!(transpile_globalvariable(gv, context)) {
                    Some(root) => cl_defs.push(root),
                    None => {}
                }
            }
            src::RootDefinition::SamplerState => unimplemented!(),
            _ => {}
        };
    }

    for rootdef in root_defs {
        match *rootdef {
            src::RootDefinition::Function(ref func) => {
                cl_defs.push(try!(transpile_functiondefinition(func, context)));
            }
            src::RootDefinition::Kernel(ref kernel) => {
                cl_defs.push(try!(transpile_kernel(kernel, context)));
            }
            _ => {}
        };
    }

    Ok(cl_defs)
}

pub fn transpile(module: &src::Module) -> Result<dst::Module, TranspileError> {

    let (mut context, binds) = try!(Context::from_globals(&module.global_table,
                                                          &module.global_declarations,
                                                          &module.root_definitions));

    let cl_defs = try!(transpile_roots(&module.root_definitions, &mut context));

    let required_extensions = context.required_extensions.clone();
    let (decls, fragments) = context.destruct();

    let cl_module = dst::Module {
        root_definitions: cl_defs,
        binds: binds,
        global_declarations: decls,
        fragments: fragments,
        required_extensions: required_extensions,
    };

    Ok(cl_module)
}

#[test]
fn test_transpile() {

    use slp_lang_hst as hst;
    use slp_transform_hst_to_hir::typeparse;
    use slp_shared::Located;

    let module = hst::Module {
        entry_point: "CSMAIN".to_string(),
        root_definitions: vec![
            hst::RootDefinition::GlobalVariable(hst::GlobalVariable {
                global_type: hst::Type::from_object(hst::ObjectType::Buffer(
                    hst::DataType(hst::DataLayout::Scalar(hst::ScalarType::Int), hst::TypeModifier::default())
                )).into(),
                defs: vec![hst::GlobalVariableName {
                    name: "g_myInBuffer".to_string(),
                    bind: hst::VariableBind::Normal,
                    slot: Some(hst::GlobalSlot::ReadSlot(0)),
                    assignment: None,
                }],
            }),
            hst::RootDefinition::Function(hst::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: hst::Type::void(),
                params: vec![hst::FunctionParam { name: "x".to_string(), param_type: hst::Type::uint().into(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            hst::RootDefinition::Function(hst::FunctionDefinition {
                name: "myFunc".to_string(),
                returntype: hst::Type::void(),
                params: vec![hst::FunctionParam { name: "x".to_string(), param_type: hst::Type::float().into(), semantic: None }],
                body: vec![],
                attributes: vec![],
            }),
            hst::RootDefinition::Function(hst::FunctionDefinition {
                name: "CSMAIN".to_string(),
                returntype: hst::Type::void(),
                params: vec![],
                body: vec![
                    hst::Statement::Empty,
                    hst::Statement::Var(hst::VarDef::new("a".to_string(), hst::Type::uint().into(), None)),
                    hst::Statement::Var(hst::VarDef::new("b".to_string(), hst::Type::uint().into(), None)),
                    hst::Statement::Expression(Located::none(
                        hst::Expression::BinaryOperation(hst::BinOp::Assignment,
                            Box::new(Located::none(hst::Expression::Variable("a".to_string()))),
                            Box::new(Located::none(hst::Expression::Variable("b".to_string())))
                        )
                    )),
                    hst::Statement::If(
                        Located::none(hst::Expression::Variable("b".to_string())),
                        Box::new(hst::Statement::Empty),
                    ),
                    hst::Statement::Expression(Located::none(
                        hst::Expression::BinaryOperation(hst::BinOp::Assignment,
                            Box::new(Located::none(hst::Expression::ArraySubscript(
                                Box::new(Located::none(hst::Expression::Variable("g_myInBuffer".to_string()))),
                                Box::new(Located::none(hst::Expression::Literal(hst::Literal::Int(0))))
                            ))),
                            Box::new(Located::none(hst::Expression::Literal(hst::Literal::Int(4))))
                        )
                    )),
                    hst::Statement::Expression(Located::none(
                        hst::Expression::Call(
                            Box::new(Located::none(hst::Expression::Variable("myFunc".to_string()))),
                            vec![
                                Located::none(hst::Expression::Variable("b".to_string()))
                            ]
                        )
                    )),
                ],
                attributes: vec![hst::FunctionAttribute::NumThreads(8, 8, 1)],
            }),
        ],
    };
    let res = typeparse(&module);
    assert!(res.is_ok(), "{:?}", res);

    let clc_res = transpile(&res.unwrap());
    assert!(clc_res.is_ok(), "{:?}", clc_res);
}
