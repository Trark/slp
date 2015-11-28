
use hlsl::ir::*;

/// Creates intrinsic nodes from argument expressions
#[derive(PartialEq, Debug, Clone)]
pub enum IntrinsicFactory {
    Intrinsic0(Intrinsic),
    Intrinsic1(fn(Expression) -> Intrinsic),
    Intrinsic2(fn(Expression, Expression) -> Intrinsic),
    Intrinsic3(fn(Expression, Expression, Expression) -> Intrinsic),
    Intrinsic4(fn(Expression, Expression, Expression, Expression) -> Intrinsic),
}

impl IntrinsicFactory {
    pub fn create_intrinsic(&self, param_values: &[Expression]) -> Intrinsic {
        match *self {
            IntrinsicFactory::Intrinsic0(ref raw) => {
                assert_eq!(param_values.len(), 0);
                raw.clone()
            },
            IntrinsicFactory::Intrinsic1(func) => {
                assert_eq!(param_values.len(), 1);
                func(
                    param_values[0].clone()
                )
            },
            IntrinsicFactory::Intrinsic2(func) => {
                assert_eq!(param_values.len(), 2);
                func(
                    param_values[0].clone(),
                    param_values[1].clone()
                )
            },
            IntrinsicFactory::Intrinsic3(func) => {
                assert_eq!(param_values.len(), 3);
                func(
                    param_values[0].clone(),
                    param_values[1].clone(),
                    param_values[2].clone()
                )
            },
            IntrinsicFactory::Intrinsic4(func) => {
                assert_eq!(param_values.len(), 4);
                func(
                    param_values[0].clone(),
                    param_values[1].clone(),
                    param_values[2].clone(),
                    param_values[3].clone()
                )
            },
        }
    }
}

pub type IntrinsicDefinition = (Type, &'static str, &'static [ParamType], IntrinsicFactory);

const T_MOD: TypeModifier = TypeModifier { is_const: false, row_order: RowOrder::Column, precise: false, volatile: false };

const T_VOID_TY: Type = Type(TypeLayout::Void, T_MOD);
const T_INT_TY: Type = Type(TypeLayout::Scalar(ScalarType::Int), T_MOD);
const T_INT1_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 1), T_MOD);
const T_INT2_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 2), T_MOD);
const T_INT3_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 3), T_MOD);
const T_INT4_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 4), T_MOD);
const T_UINT_TY: Type = Type(TypeLayout::Scalar(ScalarType::UInt), T_MOD);
const T_UINT1_TY: Type = Type(TypeLayout::Vector(ScalarType::UInt, 1), T_MOD);
const T_UINT2_TY: Type = Type(TypeLayout::Vector(ScalarType::UInt, 2), T_MOD);
const T_UINT3_TY: Type = Type(TypeLayout::Vector(ScalarType::UInt, 3), T_MOD);
const T_UINT4_TY: Type = Type(TypeLayout::Vector(ScalarType::UInt, 4), T_MOD);
const T_FLOAT_TY: Type = Type(TypeLayout::Scalar(ScalarType::Float), T_MOD);
const T_FLOAT1_TY: Type = Type(TypeLayout::Vector(ScalarType::Float, 1), T_MOD);
const T_FLOAT2_TY: Type = Type(TypeLayout::Vector(ScalarType::Float, 2), T_MOD);
const T_FLOAT3_TY: Type = Type(TypeLayout::Vector(ScalarType::Float, 3), T_MOD);
const T_FLOAT4_TY: Type = Type(TypeLayout::Vector(ScalarType::Float, 4), T_MOD);
const T_DOUBLE_TY: Type = Type(TypeLayout::Scalar(ScalarType::Double), T_MOD);

const T_INT: ParamType = ParamType(T_INT_TY, InputModifier::In, None);
const T_INT1: ParamType = ParamType(T_INT1_TY, InputModifier::In, None);
const T_INT2: ParamType = ParamType(T_INT2_TY, InputModifier::In, None);
const T_INT3: ParamType = ParamType(T_INT3_TY, InputModifier::In, None);
const T_INT4: ParamType = ParamType(T_INT4_TY, InputModifier::In, None);
const T_UINT: ParamType = ParamType(T_UINT_TY, InputModifier::In, None);
const T_UINT1: ParamType = ParamType(T_UINT1_TY, InputModifier::In, None);
const T_UINT2: ParamType = ParamType(T_UINT2_TY, InputModifier::In, None);
const T_UINT3: ParamType = ParamType(T_UINT3_TY, InputModifier::In, None);
const T_UINT4: ParamType = ParamType(T_UINT4_TY, InputModifier::In, None);
const T_FLOAT: ParamType = ParamType(T_FLOAT_TY, InputModifier::In, None);
const T_FLOAT1: ParamType = ParamType(T_FLOAT1_TY, InputModifier::In, None);
const T_FLOAT2: ParamType = ParamType(T_FLOAT2_TY, InputModifier::In, None);
const T_FLOAT3: ParamType = ParamType(T_FLOAT3_TY, InputModifier::In, None);
const T_FLOAT4: ParamType = ParamType(T_FLOAT4_TY, InputModifier::In, None);

use self::IntrinsicFactory::Intrinsic0 as I0;
use self::IntrinsicFactory::Intrinsic1 as I1;
use self::IntrinsicFactory::Intrinsic2 as I2;
use self::IntrinsicFactory::Intrinsic3 as I3;
use self::IntrinsicFactory::Intrinsic4 as I4;

const INTRINSICS: &'static [IntrinsicDefinition] = & [

    (T_VOID_TY, "AllMemoryBarrier", &[], I0(Intrinsic::AllMemoryBarrier)),
    (T_VOID_TY, "AllMemoryBarrierWithGroupSync", &[], I0(Intrinsic::AllMemoryBarrierWithGroupSync)),

    (T_INT_TY, "asint", &[T_UINT], I1(Intrinsic::AsIntU)),
    (T_INT1_TY, "asint", &[T_UINT1], I1(Intrinsic::AsIntU1)),
    (T_INT2_TY, "asint", &[T_UINT2], I1(Intrinsic::AsIntU2)),
    (T_INT3_TY, "asint", &[T_UINT3], I1(Intrinsic::AsIntU3)),
    (T_INT4_TY, "asint", &[T_UINT4], I1(Intrinsic::AsIntU4)),
    (T_INT_TY, "asint", &[T_FLOAT], I1(Intrinsic::AsIntF)),
    (T_INT1_TY, "asint", &[T_FLOAT1], I1(Intrinsic::AsIntF1)),
    (T_INT2_TY, "asint", &[T_FLOAT2], I1(Intrinsic::AsIntF2)),
    (T_INT3_TY, "asint", &[T_FLOAT3], I1(Intrinsic::AsIntF3)),
    (T_INT4_TY, "asint", &[T_FLOAT4], I1(Intrinsic::AsIntF4)),

    (T_UINT_TY, "asuint", &[T_INT], I1(Intrinsic::AsUIntI)),
    (T_UINT1_TY, "asuint", &[T_INT1], I1(Intrinsic::AsUIntI1)),
    (T_UINT2_TY, "asuint", &[T_INT2], I1(Intrinsic::AsUIntI2)),
    (T_UINT3_TY, "asuint", &[T_INT3], I1(Intrinsic::AsUIntI3)),
    (T_UINT4_TY, "asuint", &[T_INT4], I1(Intrinsic::AsUIntI4)),
    (T_UINT_TY, "asuint", &[T_FLOAT], I1(Intrinsic::AsUIntF)),
    (T_UINT1_TY, "asuint", &[T_FLOAT1], I1(Intrinsic::AsUIntF1)),
    (T_UINT2_TY, "asuint", &[T_FLOAT2], I1(Intrinsic::AsUIntF2)),
    (T_UINT3_TY, "asuint", &[T_FLOAT3], I1(Intrinsic::AsUIntF3)),
    (T_UINT4_TY, "asuint", &[T_FLOAT4], I1(Intrinsic::AsUIntF4)),

    (T_FLOAT_TY, "asfloat", &[T_INT], I1(Intrinsic::AsFloatI)),
    (T_FLOAT1_TY, "asfloat", &[T_INT1], I1(Intrinsic::AsFloatI1)),
    (T_FLOAT2_TY, "asfloat", &[T_INT2], I1(Intrinsic::AsFloatI2)),
    (T_FLOAT3_TY, "asfloat", &[T_INT3], I1(Intrinsic::AsFloatI3)),
    (T_FLOAT4_TY, "asfloat", &[T_INT4], I1(Intrinsic::AsFloatI4)),
    (T_FLOAT_TY, "asfloat", &[T_UINT], I1(Intrinsic::AsFloatU)),
    (T_FLOAT1_TY, "asfloat", &[T_UINT1], I1(Intrinsic::AsFloatU1)),
    (T_FLOAT2_TY, "asfloat", &[T_UINT2], I1(Intrinsic::AsFloatU2)),
    (T_FLOAT3_TY, "asfloat", &[T_UINT3], I1(Intrinsic::AsFloatU3)),
    (T_FLOAT4_TY, "asfloat", &[T_UINT4], I1(Intrinsic::AsFloatU4)),
    (T_FLOAT_TY, "asfloat", &[T_FLOAT], I1(Intrinsic::AsFloatF)),
    (T_FLOAT1_TY, "asfloat", &[T_FLOAT1], I1(Intrinsic::AsFloatF1)),
    (T_FLOAT2_TY, "asfloat", &[T_FLOAT2], I1(Intrinsic::AsFloatF2)),
    (T_FLOAT3_TY, "asfloat", &[T_FLOAT3], I1(Intrinsic::AsFloatF3)),
    (T_FLOAT4_TY, "asfloat", &[T_FLOAT4], I1(Intrinsic::AsFloatF4)),

    (T_DOUBLE_TY, "asdouble", &[T_UINT, T_UINT], I2(Intrinsic::AsDouble)),

    (T_INT_TY, "clamp", &[T_INT, T_INT, T_INT], I3(Intrinsic::ClampI)),
    (T_INT1_TY, "clamp", &[T_INT1, T_INT1, T_INT1], I3(Intrinsic::ClampI1)),
    (T_INT2_TY, "clamp", &[T_INT2, T_INT2, T_INT2], I3(Intrinsic::ClampI2)),
    (T_INT3_TY, "clamp", &[T_INT3, T_INT3, T_INT3], I3(Intrinsic::ClampI3)),
    (T_INT4_TY, "clamp", &[T_INT4, T_INT4, T_INT4], I3(Intrinsic::ClampI4)),
    (T_FLOAT_TY, "clamp", &[T_FLOAT, T_FLOAT, T_FLOAT], I3(Intrinsic::ClampF)),
    (T_FLOAT1_TY, "clamp", &[T_FLOAT1, T_FLOAT1, T_FLOAT1], I3(Intrinsic::ClampF1)),
    (T_FLOAT2_TY, "clamp", &[T_FLOAT2, T_FLOAT2, T_FLOAT2], I3(Intrinsic::ClampF2)),
    (T_FLOAT3_TY, "clamp", &[T_FLOAT3, T_FLOAT3, T_FLOAT3], I3(Intrinsic::ClampF3)),
    (T_FLOAT4_TY, "clamp", &[T_FLOAT4, T_FLOAT4, T_FLOAT4], I3(Intrinsic::ClampF4)),

    (T_INT_TY, "min", &[T_INT, T_INT], I2(Intrinsic::Min)),
    (T_INT1_TY, "min", &[T_INT1, T_INT1], I2(Intrinsic::Min)),
    (T_INT2_TY, "min", &[T_INT2, T_INT2], I2(Intrinsic::Min)),
    (T_INT3_TY, "min", &[T_INT3, T_INT3], I2(Intrinsic::Min)),
    (T_INT4_TY, "min", &[T_INT4, T_INT4], I2(Intrinsic::Min)),
    (T_FLOAT_TY, "min", &[T_FLOAT, T_FLOAT], I2(Intrinsic::Min)),
    (T_FLOAT1_TY, "min", &[T_FLOAT1, T_FLOAT1], I2(Intrinsic::Min)),
    (T_FLOAT2_TY, "min", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Min)),
    (T_FLOAT3_TY, "min", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Min)),
    (T_FLOAT4_TY, "min", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Min)),

    (T_INT_TY, "max", &[T_INT, T_INT], I2(Intrinsic::Max)),
    (T_INT1_TY, "max", &[T_INT1, T_INT1], I2(Intrinsic::Max)),
    (T_INT2_TY, "max", &[T_INT2, T_INT2], I2(Intrinsic::Max)),
    (T_INT3_TY, "max", &[T_INT3, T_INT3], I2(Intrinsic::Max)),
    (T_INT4_TY, "max", &[T_INT4, T_INT4], I2(Intrinsic::Max)),
    (T_FLOAT_TY, "max", &[T_FLOAT, T_FLOAT], I2(Intrinsic::Max)),
    (T_FLOAT1_TY, "max", &[T_FLOAT1, T_FLOAT1], I2(Intrinsic::Max)),
    (T_FLOAT2_TY, "max", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Max)),
    (T_FLOAT3_TY, "max", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Max)),
    (T_FLOAT4_TY, "max", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Max)),

    (T_FLOAT4_TY, "float4", &[T_FLOAT, T_FLOAT, T_FLOAT, T_FLOAT], I4(Intrinsic::Float4)),
];

pub fn get_intrinsics() -> &'static [IntrinsicDefinition] {
    INTRINSICS
}

#[test]
fn test_param_count() {

    fn param_count(factory: &IntrinsicFactory) -> usize {
        match *factory {
            IntrinsicFactory::Intrinsic0(_) => 0,
            IntrinsicFactory::Intrinsic1(_) => 1,
            IntrinsicFactory::Intrinsic2(_) => 2,
            IntrinsicFactory::Intrinsic3(_) => 3,
            IntrinsicFactory::Intrinsic4(_) => 4,
        }
    }

    for &(_, _, ref types, ref factory) in INTRINSICS {
        assert_eq!(types.len(), param_count(factory));
    }
}
