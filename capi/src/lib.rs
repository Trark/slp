use libc::c_char;
use libc::c_void;
use slp::sequence::hlsl_to_cl::hlsl_to_cl;
use slp::sequence::hlsl_to_cl::Input;
use slp::shared::IncludeHandler;
use slp::shared::KernelParamSlot;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::thread;

#[repr(C)]
pub struct slp_hlsl_to_cl_input {
    pub entry_point: *const libc::c_char,
    pub main_file: *const libc::c_char,
    pub include_handler: extern "C" fn(*mut c_void, *const c_char) -> *const c_char,
    pub user_data: *mut c_void,
    pub kernel_name: *const libc::c_char,
}

#[repr(C)]
pub struct slp_hlsl_to_cl_bind {
    pub hlsl_slot: u16,
    pub cl_slot: u16,
}

#[repr(C)]
pub struct slp_hlsl_to_cl_bind_table {
    pub binds: *mut slp_hlsl_to_cl_bind,
    pub count: u16,
}

impl slp_hlsl_to_cl_bind_table {
    fn empty() -> slp_hlsl_to_cl_bind_table {
        slp_hlsl_to_cl_bind_table {
            binds: std::ptr::null_mut(),
            count: 0,
        }
    }

    unsafe fn new(slots: HashMap<u32, KernelParamSlot>) -> slp_hlsl_to_cl_bind_table {
        let len = slots.len();
        if len == 0 {
            return slp_hlsl_to_cl_bind_table::empty();
        }

        let mut vec = Vec::with_capacity(len);

        for (hlsl_slot, cl_slot) in slots {
            vec.push(slp_hlsl_to_cl_bind {
                hlsl_slot: hlsl_slot as u16,
                cl_slot: cl_slot as u16,
            });
        }

        assert_eq!(vec.capacity(), len);
        assert_eq!(vec.len(), len);
        let res = slp_hlsl_to_cl_bind_table {
            binds: vec.as_mut_ptr(),
            count: len as u16,
        };
        mem::forget(vec);
        res
    }

    unsafe fn free(self) {
        if self.binds != ptr::null_mut() {
            Vec::from_raw_parts(self.binds, self.count as usize, self.count as usize);
        }
    }
}

#[repr(C)]
pub struct slp_hlsl_to_cl_output {
    pub error: *mut libc::c_char,
    pub source: *mut libc::c_char,
    pub kernel_name: *mut libc::c_char,
    pub constant_buffers: slp_hlsl_to_cl_bind_table,
    pub shader_resource_views: slp_hlsl_to_cl_bind_table,
    pub sampler_states: slp_hlsl_to_cl_bind_table,
    pub unordered_access_views: slp_hlsl_to_cl_bind_table,
    pub dim_x: u64,
    pub dim_y: u64,
    pub dim_z: u64,
}

pub struct CIncludeHandler {
    pub include_handler: extern "C" fn(*mut c_void, *const c_char) -> *const c_char,
    pub user_data: *mut c_void,
}

impl IncludeHandler for CIncludeHandler {
    fn load(&mut self, file_name: &str) -> Result<String, ()> {
        let c_name = CString::new(file_name).expect("expect file_name in CIncludeHandler");
        let c_user_data = self.user_data as *mut c_void;
        let data = (self.include_handler)(c_user_data, c_name.as_ptr());
        if data == ptr::null() {
            Err(())
        } else {
            let ret = unsafe { CStr::from_ptr(data).to_string_lossy().into_owned() };
            Ok(ret)
        }
    }
}

unsafe impl Send for CIncludeHandler {}

#[no_mangle]
pub unsafe extern "C" fn slp_hlsl_to_cl(input: slp_hlsl_to_cl_input) -> slp_hlsl_to_cl_output {
    unsafe fn slp_hlsl_to_cl_impl(
        input: slp_hlsl_to_cl_input,
    ) -> Result<slp_hlsl_to_cl_output, String> {
        let entry_point = CStr::from_ptr(input.entry_point)
            .to_string_lossy()
            .into_owned();
        let main_file = CStr::from_ptr(input.main_file)
            .to_string_lossy()
            .into_owned();
        let kernel_name = CStr::from_ptr(input.kernel_name)
            .to_string_lossy()
            .into_owned();

        let include_handler = CIncludeHandler {
            include_handler: input.include_handler,
            user_data: input.user_data,
        };

        let join_handle_res = thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let input = Input {
                    entry_point: entry_point,
                    file_loader: Box::new(include_handler),
                    main_file: main_file,
                    kernel_name: kernel_name,
                };

                hlsl_to_cl(input)
            });

        let join_handle = match join_handle_res {
            Ok(jh) => jh,
            Err(_) => {
                return Err("Failed to start worker thread".to_string());
            }
        };

        let output_res = match join_handle.join() {
            Ok(output) => output,
            Err(_) => {
                return Err("Failed to join worker thread".to_string());
            }
        };

        match output_res {
            Ok(output) => {
                let source_cstring =
                    CString::new(output.code.to_string()).expect("missing null byte");
                let kernel_name_cstring =
                    CString::new(output.kernel_name).expect("missing null byte");

                Ok(slp_hlsl_to_cl_output {
                    error: ptr::null_mut(),
                    source: source_cstring.into_raw(),
                    kernel_name: kernel_name_cstring.into_raw(),
                    constant_buffers: slp_hlsl_to_cl_bind_table::new(output.binds.cbuffer_map),
                    shader_resource_views: slp_hlsl_to_cl_bind_table::new(output.binds.read_map),
                    sampler_states: slp_hlsl_to_cl_bind_table::new(output.binds.sampler_map),
                    unordered_access_views: slp_hlsl_to_cl_bind_table::new(output.binds.write_map),
                    dim_x: output.dimensions.0,
                    dim_y: output.dimensions.1,
                    dim_z: output.dimensions.2,
                })
            }
            Err(err) => Err(format!("{}", err)),
        }
    }

    match slp_hlsl_to_cl_impl(input) {
        Ok(input) => input,
        Err(cstring) => {
            let error_cstring = CString::new(cstring).expect("missing null byte");

            slp_hlsl_to_cl_output {
                error: error_cstring.into_raw(),
                source: ptr::null_mut(),
                kernel_name: ptr::null_mut(),
                constant_buffers: slp_hlsl_to_cl_bind_table::empty(),
                shader_resource_views: slp_hlsl_to_cl_bind_table::empty(),
                sampler_states: slp_hlsl_to_cl_bind_table::empty(),
                unordered_access_views: slp_hlsl_to_cl_bind_table::empty(),
                dim_x: 0,
                dim_y: 0,
                dim_z: 0,
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slp_hlsl_to_cl_free(free: slp_hlsl_to_cl_output) {
    if free.error != ptr::null_mut() {
        CString::from_raw(free.error);
    }
    if free.source != ptr::null_mut() {
        CString::from_raw(free.source);
    }
    if free.kernel_name != ptr::null_mut() {
        CString::from_raw(free.kernel_name);
    }
    free.constant_buffers.free();
    free.shader_resource_views.free();
    free.sampler_states.free();
    free.unordered_access_views.free();
}
