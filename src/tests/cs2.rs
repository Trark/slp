
use hlsl_to_cl;

const CS2_HLSL: &'static [u8] = include_bytes!("cs2.hlsl");
const CS2_CL_EXPECTED: &'static str = include_str!("cs2_expected.cl");

#[test]
fn cs2_full() {

    let code_result = hlsl_to_cl(CS2_HLSL, "CSMAIN");
    assert!(code_result.is_ok(), "{:?}", code_result);
    let code = code_result.unwrap();

    let expected = CS2_CL_EXPECTED.to_string().replace("\r\n", "\n");
    assert_eq!(&code.to_string()[..], expected);

}
