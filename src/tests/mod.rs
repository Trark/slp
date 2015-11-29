
use std::collections::HashMap;
use hlsl_to_cl;
use BindMap;

fn run_full(hlsl: &'static [u8], cl: &'static str, binds: BindMap) {

    let code_result = hlsl_to_cl(hlsl, "CSMAIN");
    assert!(code_result.is_ok(), "{:?}", code_result);
    let output = code_result.unwrap();

    let expected = cl.to_string().replace("\r\n", "\n");
    for (code_line, expected_line) in output.code.to_string().lines().zip(expected.lines()) {
        assert_eq!(&code_line[..], expected_line);
    }
    assert_eq!(&output.code.to_string()[..], expected);

    assert_eq!(output.binds, binds);
}


#[test]
fn cs2_full() {
    const HLSL: &'static [u8] = include_bytes!("cs2.hlsl");
    const CL: &'static str = include_str!("cs2.cl");
    run_full(HLSL, CL, BindMap {
        read_map: { let mut map = HashMap::new(); map.insert(0, 1); map },
        write_map: { let mut map = HashMap::new(); map.insert(0, 2); map },
        cbuffer_map: { let mut map = HashMap::new(); map.insert(0, 0); map },
        sampler_map: HashMap::new(),
    });
}

#[test]
fn overload_full() {
    const HLSL: &'static [u8] = include_bytes!("overload.hlsl");
    const CL: &'static str = include_str!("overload.cl");
    run_full(HLSL, CL, BindMap::new());
}


#[test]
fn intrinsic_full() {
    const HLSL: &'static [u8] = include_bytes!("intrinsic.hlsl");
    const CL: &'static str = include_str!("intrinsic.cl");
    run_full(HLSL, CL, BindMap::new());
}
