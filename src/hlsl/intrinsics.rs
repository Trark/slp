
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

pub type IntrinsicDefinition = (Type, &'static str, &'static [Type], IntrinsicFactory);

const T_VOID: Type = Type::Void;
const T_BOOL: Type = Type::Structured(StructuredType::Data(DataType::Scalar(ScalarType::Bool)));
const T_BOOL1: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Bool, 1)));
const T_BOOL2: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Bool, 2)));
const T_BOOL3: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Bool, 3)));
const T_BOOL4: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Bool, 4)));
const T_INT: Type = Type::Structured(StructuredType::Data(DataType::Scalar(ScalarType::Int)));
const T_INT1: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Int, 1)));
const T_INT2: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Int, 2)));
const T_INT3: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Int, 3)));
const T_INT4: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Int, 4)));
const T_UINT: Type = Type::Structured(StructuredType::Data(DataType::Scalar(ScalarType::UInt)));
const T_UINT1: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 1)));
const T_UINT2: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 2)));
const T_UINT3: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 3)));
const T_UINT4: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::UInt, 4)));
const T_FLOAT: Type = Type::Structured(StructuredType::Data(DataType::Scalar(ScalarType::Float)));
const T_FLOAT1: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Float, 1)));
const T_FLOAT2: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Float, 2)));
const T_FLOAT3: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Float, 3)));
const T_FLOAT4: Type = Type::Structured(StructuredType::Data(DataType::Vector(ScalarType::Float, 4)));
const T_DOUBLE: Type = Type::Structured(StructuredType::Data(DataType::Scalar(ScalarType::Double)));

use self::IntrinsicFactory::Intrinsic0 as I0;
use self::IntrinsicFactory::Intrinsic1 as I1;
use self::IntrinsicFactory::Intrinsic2 as I2;
use self::IntrinsicFactory::Intrinsic3 as I3;
use self::IntrinsicFactory::Intrinsic4 as I4;

const INTRINSICS: &'static [IntrinsicDefinition] = & [

    (T_VOID, "AllMemoryBarrier", &[], I0(Intrinsic::AllMemoryBarrier)),
    (T_VOID, "AllMemoryBarrierWithGroupSync", &[], I0(Intrinsic::AllMemoryBarrierWithGroupSync)),

    (T_INT, "asint", &[T_UINT], I1(Intrinsic::AsIntU)),
    (T_INT1, "asint", &[T_UINT1], I1(Intrinsic::AsIntU1)),
    (T_INT2, "asint", &[T_UINT2], I1(Intrinsic::AsIntU2)),
    (T_INT3, "asint", &[T_UINT3], I1(Intrinsic::AsIntU3)),
    (T_INT4, "asint", &[T_UINT4], I1(Intrinsic::AsIntU4)),
    (T_INT, "asint", &[T_FLOAT], I1(Intrinsic::AsIntF)),
    (T_INT1, "asint", &[T_FLOAT1], I1(Intrinsic::AsIntF1)),
    (T_INT2, "asint", &[T_FLOAT2], I1(Intrinsic::AsIntF2)),
    (T_INT3, "asint", &[T_FLOAT3], I1(Intrinsic::AsIntF3)),
    (T_INT4, "asint", &[T_FLOAT4], I1(Intrinsic::AsIntF4)),

    (T_UINT, "asuint", &[T_INT], I1(Intrinsic::AsUIntI)),
    (T_UINT1, "asuint", &[T_INT1], I1(Intrinsic::AsUIntI1)),
    (T_UINT2, "asuint", &[T_INT2], I1(Intrinsic::AsUIntI2)),
    (T_UINT3, "asuint", &[T_INT3], I1(Intrinsic::AsUIntI3)),
    (T_UINT4, "asuint", &[T_INT4], I1(Intrinsic::AsUIntI4)),
    (T_UINT, "asuint", &[T_FLOAT], I1(Intrinsic::AsUIntF)),
    (T_UINT1, "asuint", &[T_FLOAT1], I1(Intrinsic::AsUIntF1)),
    (T_UINT2, "asuint", &[T_FLOAT2], I1(Intrinsic::AsUIntF2)),
    (T_UINT3, "asuint", &[T_FLOAT3], I1(Intrinsic::AsUIntF3)),
    (T_UINT4, "asuint", &[T_FLOAT4], I1(Intrinsic::AsUIntF4)),

    (T_FLOAT, "asfloat", &[T_BOOL], I1(Intrinsic::AsFloatB)),
    (T_FLOAT1, "asfloat", &[T_BOOL1], I1(Intrinsic::AsFloatB1)),
    (T_FLOAT2, "asfloat", &[T_BOOL2], I1(Intrinsic::AsFloatB2)),
    (T_FLOAT3, "asfloat", &[T_BOOL3], I1(Intrinsic::AsFloatB3)),
    (T_FLOAT4, "asfloat", &[T_BOOL4], I1(Intrinsic::AsFloatB4)),
    (T_FLOAT, "asfloat", &[T_INT], I1(Intrinsic::AsFloatI)),
    (T_FLOAT1, "asfloat", &[T_INT1], I1(Intrinsic::AsFloatI1)),
    (T_FLOAT2, "asfloat", &[T_INT2], I1(Intrinsic::AsFloatI2)),
    (T_FLOAT3, "asfloat", &[T_INT3], I1(Intrinsic::AsFloatI3)),
    (T_FLOAT4, "asfloat", &[T_INT4], I1(Intrinsic::AsFloatI4)),
    (T_FLOAT, "asfloat", &[T_UINT], I1(Intrinsic::AsFloatU)),
    (T_FLOAT1, "asfloat", &[T_UINT1], I1(Intrinsic::AsFloatU1)),
    (T_FLOAT2, "asfloat", &[T_UINT2], I1(Intrinsic::AsFloatU2)),
    (T_FLOAT3, "asfloat", &[T_UINT3], I1(Intrinsic::AsFloatU3)),
    (T_FLOAT4, "asfloat", &[T_UINT4], I1(Intrinsic::AsFloatU4)),
    (T_FLOAT, "asfloat", &[T_FLOAT], I1(Intrinsic::AsFloatF)),
    (T_FLOAT1, "asfloat", &[T_FLOAT1], I1(Intrinsic::AsFloatF1)),
    (T_FLOAT2, "asfloat", &[T_FLOAT2], I1(Intrinsic::AsFloatF2)),
    (T_FLOAT3, "asfloat", &[T_FLOAT3], I1(Intrinsic::AsFloatF3)),
    (T_FLOAT4, "asfloat", &[T_FLOAT4], I1(Intrinsic::AsFloatF4)),

    (T_DOUBLE, "asdouble", &[T_UINT, T_UINT], I2(Intrinsic::AsDouble)),

    (T_INT, "clamp", &[T_INT, T_INT, T_INT], I3(Intrinsic::ClampI)),
    (T_INT1, "clamp", &[T_INT1, T_INT1, T_INT1], I3(Intrinsic::ClampI1)),
    (T_INT2, "clamp", &[T_INT2, T_INT2, T_INT2], I3(Intrinsic::ClampI2)),
    (T_INT3, "clamp", &[T_INT3, T_INT3, T_INT3], I3(Intrinsic::ClampI3)),
    (T_INT4, "clamp", &[T_INT4, T_INT4, T_INT4], I3(Intrinsic::ClampI4)),
    (T_FLOAT, "clamp", &[T_FLOAT, T_FLOAT, T_FLOAT], I3(Intrinsic::ClampF)),
    (T_FLOAT1, "clamp", &[T_FLOAT1, T_FLOAT1, T_FLOAT1], I3(Intrinsic::ClampF1)),
    (T_FLOAT2, "clamp", &[T_FLOAT2, T_FLOAT2, T_FLOAT2], I3(Intrinsic::ClampF2)),
    (T_FLOAT3, "clamp", &[T_FLOAT3, T_FLOAT3, T_FLOAT3], I3(Intrinsic::ClampF3)),
    (T_FLOAT4, "clamp", &[T_FLOAT4, T_FLOAT4, T_FLOAT4], I3(Intrinsic::ClampF4)),

    (T_INT, "min", &[T_INT, T_INT], I2(Intrinsic::Min)),
    (T_INT1, "min", &[T_INT1, T_INT1], I2(Intrinsic::Min)),
    (T_INT2, "min", &[T_INT2, T_INT2], I2(Intrinsic::Min)),
    (T_INT3, "min", &[T_INT3, T_INT3], I2(Intrinsic::Min)),
    (T_INT4, "min", &[T_INT4, T_INT4], I2(Intrinsic::Min)),
    (T_FLOAT, "min", &[T_FLOAT, T_FLOAT], I2(Intrinsic::Min)),
    (T_FLOAT1, "min", &[T_FLOAT1, T_FLOAT1], I2(Intrinsic::Min)),
    (T_FLOAT2, "min", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Min)),
    (T_FLOAT3, "min", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Min)),
    (T_FLOAT4, "min", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Min)),

    (T_INT, "max", &[T_INT, T_INT], I2(Intrinsic::Max)),
    (T_INT1, "max", &[T_INT1, T_INT1], I2(Intrinsic::Max)),
    (T_INT2, "max", &[T_INT2, T_INT2], I2(Intrinsic::Max)),
    (T_INT3, "max", &[T_INT3, T_INT3], I2(Intrinsic::Max)),
    (T_INT4, "max", &[T_INT4, T_INT4], I2(Intrinsic::Max)),
    (T_FLOAT, "max", &[T_FLOAT, T_FLOAT], I2(Intrinsic::Max)),
    (T_FLOAT1, "max", &[T_FLOAT1, T_FLOAT1], I2(Intrinsic::Max)),
    (T_FLOAT2, "max", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Max)),
    (T_FLOAT3, "max", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Max)),
    (T_FLOAT4, "max", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Max)),

    (T_FLOAT4, "float4", &[T_FLOAT, T_FLOAT, T_FLOAT, T_FLOAT], I4(Intrinsic::Float4)),
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
