
use std::collections::HashMap;
use slp_sequence_hlsl_to_cl::Input;
use slp_shared::IncludeHandler;
use slp_shared::NullIncludeHandler;
use slp_sequence_hlsl_to_cl::hlsl_to_cl;
use slp_shared::BindMap;

mod cs1;

fn run_input(input: Input, cl: &'static str, binds: BindMap) {
    let code_result = hlsl_to_cl(input);
    assert!(code_result.is_ok(), "{:?}", code_result);
    let output = code_result.unwrap();

    let expected = cl.to_string().replace("\r\n", "\n");
    for (code_line, expected_line) in output.code.to_string().lines().zip(expected.lines()) {
        assert_eq!(&code_line[..], expected_line);
    }
    assert_eq!(&output.code.to_string()[..], expected);

    assert_eq!(output.binds, binds);
}

fn run_full(hlsl: &'static str, cl: &'static str, binds: BindMap) {
    run_input(Input {
                  entry_point: "CSMAIN".to_string(),
                  main_file: hlsl.to_string(),
                  file_loader: Box::new(NullIncludeHandler),
                  kernel_name: "MyKernel".to_string(),
              },
              cl,
              binds)
}


#[test]
fn cs2_full() {
    const HLSL: &'static str = include_str!("cs2.hlsl");
    const CL: &'static str = include_str!("cs2.cl");
    run_full(HLSL,
             CL,
             BindMap {
                 read_map: {
                     let mut map = HashMap::new();
                     map.insert(0, 1);
                     map
                 },
                 write_map: {
                     let mut map = HashMap::new();
                     map.insert(0, 2);
                     map
                 },
                 cbuffer_map: {
                     let mut map = HashMap::new();
                     map.insert(0, 0);
                     map
                 },
                 sampler_map: HashMap::new(),
             });
}

#[test]
fn overload_full() {
    const HLSL: &'static str = include_str!("overload.hlsl");
    const CL: &'static str = include_str!("overload.cl");
    run_full(HLSL, CL, BindMap::new());
}


#[test]
fn intrinsic_full() {
    const HLSL: &'static str = include_str!("intrinsic.hlsl");
    const CL: &'static str = include_str!("intrinsic.cl");
    run_full(HLSL,
             CL,
             BindMap {
                 read_map: {
                     let mut map = HashMap::new();
                     map.insert(0, 1);
                     map.insert(1, 2);
                     map.insert(2, 3);
                     map.insert(3, 4);
                     map.insert(4, 5);
                     map.insert(5, 6);
                     map
                 },
                 write_map: {
                     let mut map = HashMap::new();
                     map.insert(0, 7);
                     map.insert(1, 8);
                     map.insert(2, 9);
                     map.insert(3, 10);
                     map.insert(4, 11);
                     map.insert(5, 12);
                     map
                 },
                 cbuffer_map: HashMap::new(),
                 sampler_map: {
                     let mut map = HashMap::new();
                     map.insert(0, 0);
                     map
                 },
             });
}

#[test]
fn swizzle_full() {
    const HLSL: &'static str = include_str!("swizzle.hlsl");
    const CL: &'static str = include_str!("swizzle.cl");
    run_full(HLSL, CL, BindMap::new());
}

#[test]
fn cons_full() {
    const HLSL: &'static str = include_str!("cons.hlsl");
    const CL: &'static str = include_str!("cons.cl");
    run_full(HLSL, CL, BindMap::new());
}

#[test]
fn ternary_full() {
    const HLSL: &'static str = include_str!("ternary.hlsl");
    const CL: &'static str = include_str!("ternary.cl");
    run_full(HLSL, CL, BindMap::new());
}

#[test]
fn include() {
    const HLSL_MAIN: &'static str = include_str!("include_main.hlsl");
    const CL: &'static str = include_str!("include.cl");

    struct TestFileLoader;
    impl IncludeHandler for TestFileLoader {
        fn load(&mut self, file_name: &str) -> Result<String, ()> {
            match file_name.as_ref() {
                "aux.csh" => Ok(include_str!("include_aux.hlsl").to_string()),
                _ => Err(()),
            }
        }
    }

    run_input(Input {
                  entry_point: "CSMAIN".to_string(),
                  main_file: HLSL_MAIN.to_string(),
                  file_loader: Box::new(TestFileLoader),
                  kernel_name: "Main".to_string(),
              },
              CL,
              BindMap::new())
}
