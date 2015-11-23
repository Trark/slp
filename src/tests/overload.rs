
use hlsl_to_cl;

const OVERLOAD_HLSL: &'static [u8] = include_bytes!("overload.hlsl");
const OVERLOAD_CL: &'static str = include_str!("overload.cl");

#[test]
fn overload_full() {

    let code_result = hlsl_to_cl(OVERLOAD_HLSL, "CSMAIN");
    assert!(code_result.is_ok(), "{:?}", code_result);
    let code = code_result.unwrap();

    let expected = OVERLOAD_CL.to_string().replace("\r\n", "\n");
    for (code_line, expected_line) in code.to_string().lines().zip(expected.lines()) {
        assert_eq!(&code_line[..], expected_line);
    }
    assert_eq!(&code.to_string()[..], expected);

}
