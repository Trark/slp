
use slp_lang_hir::*;

/// Creates intrinsic nodes from argument expressions
#[derive(PartialEq, Debug, Clone)]
pub enum IntrinsicFactory {
    Intrinsic0(Intrinsic),
    Intrinsic1(fn(Expression) -> Intrinsic),
    Intrinsic2(fn(Expression, Expression) -> Intrinsic),
    Intrinsic3(fn(Expression, Expression, Expression) -> Intrinsic),
}

impl IntrinsicFactory {
    pub fn create_intrinsic(&self, param_values: &[Expression]) -> Intrinsic {
        match *self {
            IntrinsicFactory::Intrinsic0(ref raw) => {
                assert_eq!(param_values.len(), 0);
                raw.clone()
            }
            IntrinsicFactory::Intrinsic1(func) => {
                assert_eq!(param_values.len(), 1);
                func(param_values[0].clone())
            }
            IntrinsicFactory::Intrinsic2(func) => {
                assert_eq!(param_values.len(), 2);
                func(param_values[0].clone(), param_values[1].clone())
            }
            IntrinsicFactory::Intrinsic3(func) => {
                assert_eq!(param_values.len(), 3);
                func(param_values[0].clone(),
                     param_values[1].clone(),
                     param_values[2].clone())
            }
        }
    }
}

pub type IntrinsicDefinition = (Type, &'static str, &'static [ParamType], IntrinsicFactory);

const T_MOD: TypeModifier = TypeModifier {
    is_const: false,
    row_order: RowOrder::Column,
    precise: false,
    volatile: false,
};

const T_VOID_TY: Type = Type(TypeLayout::Void, T_MOD);
const T_BOOL_TY: Type = Type(TypeLayout::Scalar(ScalarType::Bool), T_MOD);
const T_BOOL2_TY: Type = Type(TypeLayout::Vector(ScalarType::Bool, 2), T_MOD);
const T_BOOL3_TY: Type = Type(TypeLayout::Vector(ScalarType::Bool, 3), T_MOD);
const T_BOOL4_TY: Type = Type(TypeLayout::Vector(ScalarType::Bool, 4), T_MOD);
const T_INT_TY: Type = Type(TypeLayout::Scalar(ScalarType::Int), T_MOD);
const T_INT1_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 1), T_MOD);
const T_INT2_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 2), T_MOD);
const T_INT3_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 3), T_MOD);
const T_INT4_TY: Type = Type(TypeLayout::Vector(ScalarType::Int, 4), T_MOD);
const T_UINT_TY: Type = Type(TypeLayout::Scalar(ScalarType::UInt), T_MOD);
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

#[cfg_attr(rustfmt, rustfmt_skip)]
const INTRINSICS: &'static [IntrinsicDefinition] = &[
    (T_VOID_TY, "AllMemoryBarrier", &[], I0(Intrinsic::AllMemoryBarrier)),
    (T_VOID_TY, "AllMemoryBarrierWithGroupSync", &[], I0(Intrinsic::AllMemoryBarrierWithGroupSync)),
    (T_VOID_TY, "DeviceMemoryBarrier", &[], I0(Intrinsic::DeviceMemoryBarrier)),
    (T_VOID_TY, "DeviceMemoryBarrierWithGroupSync", &[], I0(Intrinsic::DeviceMemoryBarrierWithGroupSync)),
    (T_VOID_TY, "GroupMemoryBarrier", &[], I0(Intrinsic::GroupMemoryBarrier)),
    (T_VOID_TY, "GroupMemoryBarrierWithGroupSync", &[], I0(Intrinsic::GroupMemoryBarrierWithGroupSync)),

    (T_INT_TY, "asint", &[T_UINT], I1(Intrinsic::AsIntU)),
    (T_INT2_TY, "asint", &[T_UINT2], I1(Intrinsic::AsIntU2)),
    (T_INT3_TY, "asint", &[T_UINT3], I1(Intrinsic::AsIntU3)),
    (T_INT4_TY, "asint", &[T_UINT4], I1(Intrinsic::AsIntU4)),
    (T_INT_TY, "asint", &[T_FLOAT], I1(Intrinsic::AsIntF)),
    (T_INT2_TY, "asint", &[T_FLOAT2], I1(Intrinsic::AsIntF2)),
    (T_INT3_TY, "asint", &[T_FLOAT3], I1(Intrinsic::AsIntF3)),
    (T_INT4_TY, "asint", &[T_FLOAT4], I1(Intrinsic::AsIntF4)),

    (T_UINT_TY, "asuint", &[T_INT], I1(Intrinsic::AsUIntI)),
    (T_UINT2_TY, "asuint", &[T_INT2], I1(Intrinsic::AsUIntI2)),
    (T_UINT3_TY, "asuint", &[T_INT3], I1(Intrinsic::AsUIntI3)),
    (T_UINT4_TY, "asuint", &[T_INT4], I1(Intrinsic::AsUIntI4)),
    (T_UINT_TY, "asuint", &[T_FLOAT], I1(Intrinsic::AsUIntF)),
    (T_UINT2_TY, "asuint", &[T_FLOAT2], I1(Intrinsic::AsUIntF2)),
    (T_UINT3_TY, "asuint", &[T_FLOAT3], I1(Intrinsic::AsUIntF3)),
    (T_UINT4_TY, "asuint", &[T_FLOAT4], I1(Intrinsic::AsUIntF4)),

    (T_FLOAT_TY, "asfloat", &[T_INT], I1(Intrinsic::AsFloatI)),
    (T_FLOAT2_TY, "asfloat", &[T_INT2], I1(Intrinsic::AsFloatI2)),
    (T_FLOAT3_TY, "asfloat", &[T_INT3], I1(Intrinsic::AsFloatI3)),
    (T_FLOAT4_TY, "asfloat", &[T_INT4], I1(Intrinsic::AsFloatI4)),
    (T_FLOAT_TY, "asfloat", &[T_UINT], I1(Intrinsic::AsFloatU)),
    (T_FLOAT2_TY, "asfloat", &[T_UINT2], I1(Intrinsic::AsFloatU2)),
    (T_FLOAT3_TY, "asfloat", &[T_UINT3], I1(Intrinsic::AsFloatU3)),
    (T_FLOAT4_TY, "asfloat", &[T_UINT4], I1(Intrinsic::AsFloatU4)),
    (T_FLOAT_TY, "asfloat", &[T_FLOAT], I1(Intrinsic::AsFloatF)),
    (T_FLOAT2_TY, "asfloat", &[T_FLOAT2], I1(Intrinsic::AsFloatF2)),
    (T_FLOAT3_TY, "asfloat", &[T_FLOAT3], I1(Intrinsic::AsFloatF3)),
    (T_FLOAT4_TY, "asfloat", &[T_FLOAT4], I1(Intrinsic::AsFloatF4)),

    (T_DOUBLE_TY, "asdouble", &[T_UINT, T_UINT], I2(Intrinsic::AsDouble)),

    (T_INT_TY, "clamp", &[T_INT, T_INT, T_INT], I3(Intrinsic::ClampI)),
    (T_INT2_TY, "clamp", &[T_INT2, T_INT2, T_INT2], I3(Intrinsic::ClampI2)),
    (T_INT3_TY, "clamp", &[T_INT3, T_INT3, T_INT3], I3(Intrinsic::ClampI3)),
    (T_INT4_TY, "clamp", &[T_INT4, T_INT4, T_INT4], I3(Intrinsic::ClampI4)),
    (T_FLOAT_TY, "clamp", &[T_FLOAT, T_FLOAT, T_FLOAT], I3(Intrinsic::ClampF)),
    (T_FLOAT2_TY, "clamp", &[T_FLOAT2, T_FLOAT2, T_FLOAT2], I3(Intrinsic::ClampF2)),
    (T_FLOAT3_TY, "clamp", &[T_FLOAT3, T_FLOAT3, T_FLOAT3], I3(Intrinsic::ClampF3)),
    (T_FLOAT4_TY, "clamp", &[T_FLOAT4, T_FLOAT4, T_FLOAT4], I3(Intrinsic::ClampF4)),

    (T_FLOAT3_TY, "cross", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Cross)),

    (T_FLOAT_TY, "distance", &[T_FLOAT1, T_FLOAT1], I2(Intrinsic::Distance1)),
    (T_FLOAT_TY, "distance", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Distance2)),
    (T_FLOAT_TY, "distance", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Distance3)),
    (T_FLOAT_TY, "distance", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Distance4)),

    (T_INT_TY, "dot", &[T_INT1, T_INT1], I2(Intrinsic::DotI1)),
    (T_INT_TY, "dot", &[T_INT2, T_INT2], I2(Intrinsic::DotI2)),
    (T_INT_TY, "dot", &[T_INT3, T_INT3], I2(Intrinsic::DotI3)),
    (T_INT_TY, "dot", &[T_INT4, T_INT4], I2(Intrinsic::DotI4)),
    (T_FLOAT_TY, "dot", &[T_FLOAT1, T_FLOAT1], I2(Intrinsic::DotF1)),
    (T_FLOAT_TY, "dot", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::DotF2)),
    (T_FLOAT_TY, "dot", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::DotF3)),
    (T_FLOAT_TY, "dot", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::DotF4)),

    (T_FLOAT_TY, "f16tof32", &[T_UINT], I1(Intrinsic::F16ToF32)),
    (T_UINT_TY, "f32tof16", &[T_FLOAT], I1(Intrinsic::F32ToF16)),

    (T_FLOAT_TY, "floor", &[T_FLOAT], I1(Intrinsic::Floor)),
    (T_FLOAT2_TY, "floor", &[T_FLOAT2], I1(Intrinsic::Floor2)),
    (T_FLOAT3_TY, "floor", &[T_FLOAT3], I1(Intrinsic::Floor3)),
    (T_FLOAT4_TY, "floor", &[T_FLOAT4], I1(Intrinsic::Floor4)),

    (T_BOOL_TY, "isnan", &[T_FLOAT], I1(Intrinsic::IsNaN)),
    (T_BOOL2_TY, "isnan", &[T_FLOAT2], I1(Intrinsic::IsNaN2)),
    (T_BOOL3_TY, "isnan", &[T_FLOAT3], I1(Intrinsic::IsNaN3)),
    (T_BOOL4_TY, "isnan", &[T_FLOAT4], I1(Intrinsic::IsNaN4)),

    (T_FLOAT_TY, "length", &[T_FLOAT1], I1(Intrinsic::Length1)),
    (T_FLOAT_TY, "length", &[T_FLOAT2], I1(Intrinsic::Length2)),
    (T_FLOAT_TY, "length", &[T_FLOAT3], I1(Intrinsic::Length3)),
    (T_FLOAT_TY, "length", &[T_FLOAT4], I1(Intrinsic::Length4)),

    (T_INT_TY, "min", &[T_INT, T_INT], I2(Intrinsic::Min)),
    (T_INT2_TY, "min", &[T_INT2, T_INT2], I2(Intrinsic::Min)),
    (T_INT3_TY, "min", &[T_INT3, T_INT3], I2(Intrinsic::Min)),
    (T_INT4_TY, "min", &[T_INT4, T_INT4], I2(Intrinsic::Min)),
    (T_FLOAT_TY, "min", &[T_FLOAT, T_FLOAT], I2(Intrinsic::Min)),
    (T_FLOAT2_TY, "min", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Min)),
    (T_FLOAT3_TY, "min", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Min)),
    (T_FLOAT4_TY, "min", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Min)),

    (T_INT_TY, "max", &[T_INT, T_INT], I2(Intrinsic::Max)),
    (T_INT2_TY, "max", &[T_INT2, T_INT2], I2(Intrinsic::Max)),
    (T_INT3_TY, "max", &[T_INT3, T_INT3], I2(Intrinsic::Max)),
    (T_INT4_TY, "max", &[T_INT4, T_INT4], I2(Intrinsic::Max)),
    (T_FLOAT_TY, "max", &[T_FLOAT, T_FLOAT], I2(Intrinsic::Max)),
    (T_FLOAT2_TY, "max", &[T_FLOAT2, T_FLOAT2], I2(Intrinsic::Max)),
    (T_FLOAT3_TY, "max", &[T_FLOAT3, T_FLOAT3], I2(Intrinsic::Max)),
    (T_FLOAT4_TY, "max", &[T_FLOAT4, T_FLOAT4], I2(Intrinsic::Max)),

    (T_FLOAT1_TY, "normalize", &[T_FLOAT1], I1(Intrinsic::Normalize1)),
    (T_FLOAT2_TY, "normalize", &[T_FLOAT2], I1(Intrinsic::Normalize2)),
    (T_FLOAT3_TY, "normalize", &[T_FLOAT3], I1(Intrinsic::Normalize3)),
    (T_FLOAT4_TY, "normalize", &[T_FLOAT4], I1(Intrinsic::Normalize4)),

    (T_INT_TY, "sign", &[T_INT], I1(Intrinsic::SignI)),
    (T_INT2_TY, "sign", &[T_INT2], I1(Intrinsic::SignI2)),
    (T_INT3_TY, "sign", &[T_INT3], I1(Intrinsic::SignI3)),
    (T_INT4_TY, "sign", &[T_INT4], I1(Intrinsic::SignI4)),
    (T_INT_TY, "sign", &[T_FLOAT], I1(Intrinsic::SignF)),
    (T_INT2_TY, "sign", &[T_FLOAT2], I1(Intrinsic::SignF2)),
    (T_INT3_TY, "sign", &[T_FLOAT3], I1(Intrinsic::SignF3)),
    (T_INT4_TY, "sign", &[T_FLOAT4], I1(Intrinsic::SignF4)),

    (T_FLOAT_TY, "sqrt", &[T_FLOAT], I1(Intrinsic::Sqrt)),
    (T_FLOAT2_TY, "sqrt", &[T_FLOAT2], I1(Intrinsic::Sqrt2)),
    (T_FLOAT3_TY, "sqrt", &[T_FLOAT3], I1(Intrinsic::Sqrt3)),
    (T_FLOAT4_TY, "sqrt", &[T_FLOAT4], I1(Intrinsic::Sqrt4)),
];

pub fn get_intrinsics() -> &'static [IntrinsicDefinition] {
    INTRINSICS
}

pub struct MethodDefinition(pub ObjectType,
                            pub String,
                            pub Vec<(Type, Vec<ParamType>, IntrinsicFactory)>);

#[cfg_attr(rustfmt, rustfmt_skip)]
pub fn get_method(object: &ObjectType, name: &str) -> Result<MethodDefinition, ()> {

    type MethodT0 = (&'static str, Type, &'static [ParamType], IntrinsicFactory);
    fn find_method_0(object: &ObjectType, defs: &'static [MethodT0], name: &str) -> Result<MethodDefinition, ()> {
        let mut methods = vec![];
        for &(ref method_name, ref return_type, ref param_types, ref factory) in defs {
            if *method_name == name {
                methods.push((return_type.clone(), param_types.to_vec(), factory.clone()));
            };
        }
        if methods.len() > 0 {
            Ok(MethodDefinition(object.clone(), name.to_string(), methods))
        } else {
            Err(())
        }
    }

    type MethodT1 = (&'static str, Box<Fn(&Type) -> (Type, Vec<ParamType>)>, IntrinsicFactory);
    fn find_method_1(object: &ObjectType, t0: &Type, defs: &[MethodT1], name: &str) -> Result<MethodDefinition, ()> {
        let mut methods = vec![];
        for &(ref method_name, ref create, ref factory) in defs {
            if *method_name == name {
                let (return_type, param_types) = create(t0);
                methods.push((return_type.clone(), param_types.to_vec(), factory.clone()));
            }
        }
        if methods.len() > 0 {
            Ok(MethodDefinition(object.clone(), name.to_string(), methods))
        } else {
            Err(())
        }
    }

    match *object {
        ObjectType::Buffer(ref data_type) => {
            let methods: &[MethodT1] = &[
                ("Load", Box::new(|ty: &Type| (ty.clone(), vec![T_INT])), I2(Intrinsic::BufferLoad)),
            ];
            find_method_1(object, &Type::from_data(data_type.clone()), &methods, name)
        }
        ObjectType::RWBuffer(ref data_type) => {
            let methods: &[MethodT1] = &[
                ("Load", Box::new(|ty: &Type| (ty.clone(), vec![T_INT])), I2(Intrinsic::RWBufferLoad)),
            ];
            find_method_1(object, &Type::from_data(data_type.clone()), &methods, name)
        }
        ObjectType::StructuredBuffer(ref structured_type) => {
            let methods: &[MethodT1] = &[
                ("Load", Box::new(|ty: &Type| (ty.clone(), vec![T_INT])), I2(Intrinsic::StructuredBufferLoad)),
            ];
            find_method_1(object, &Type::from_structured(structured_type.clone()), &methods, name)
        }
        ObjectType::RWStructuredBuffer(ref structured_type) => {
            let methods: &[MethodT1] = &[
                ("Load", Box::new(|ty: &Type| (ty.clone(), vec![T_INT])), I2(Intrinsic::RWStructuredBufferLoad)),
            ];
            find_method_1(object, &Type::from_structured(structured_type.clone()), &methods, name)
        }
        ObjectType::RWTexture2D(ref data_type) => {
            let methods: &[MethodT1] = &[
                ("Load", Box::new(|ty: &Type| (ty.clone(), vec![T_INT2])), I2(Intrinsic::RWTexture2DLoad)),
            ];
            find_method_1(object, &Type::from_data(data_type.clone()), &methods, name)
        }
        ObjectType::ByteAddressBuffer => {
            const METHODS: &'static [MethodT0] = &[
                ("Load", T_UINT_TY, &[T_UINT], I2(Intrinsic::ByteAddressBufferLoad)),
                ("Load2", T_UINT2_TY, &[T_UINT], I2(Intrinsic::ByteAddressBufferLoad2)),
                ("Load3", T_UINT3_TY, &[T_UINT], I2(Intrinsic::ByteAddressBufferLoad3)),
                ("Load4", T_UINT4_TY, &[T_UINT], I2(Intrinsic::ByteAddressBufferLoad4)),
            ];
            find_method_0(object, METHODS, name)
        }
        ObjectType::RWByteAddressBuffer => {
            const METHODS: &'static [MethodT0] = &[
                ("Load", T_UINT_TY, &[T_UINT], I2(Intrinsic::RWByteAddressBufferLoad)),
                ("Load2", T_UINT2_TY, &[T_UINT], I2(Intrinsic::RWByteAddressBufferLoad2)),
                ("Load3", T_UINT3_TY, &[T_UINT], I2(Intrinsic::RWByteAddressBufferLoad3)),
                ("Load4", T_UINT4_TY, &[T_UINT], I2(Intrinsic::RWByteAddressBufferLoad4)),
                ("Store", T_VOID_TY, &[T_UINT, T_UINT], I3(Intrinsic::RWByteAddressBufferStore)),
                ("Store2", T_VOID_TY, &[T_UINT, T_UINT2], I3(Intrinsic::RWByteAddressBufferStore2)),
                ("Store3", T_VOID_TY, &[T_UINT, T_UINT3], I3(Intrinsic::RWByteAddressBufferStore3)),
                ("Store4", T_VOID_TY, &[T_UINT, T_UINT4], I3(Intrinsic::RWByteAddressBufferStore4)),
            ];
            find_method_0(object, METHODS, name)
        }
        _ => Err(())
    }
}



#[test]
fn test_param_count() {

    fn param_count(factory: &IntrinsicFactory) -> usize {
        match *factory {
            IntrinsicFactory::Intrinsic0(_) => 0,
            IntrinsicFactory::Intrinsic1(_) => 1,
            IntrinsicFactory::Intrinsic2(_) => 2,
            IntrinsicFactory::Intrinsic3(_) => 3,
        }
    }

    for &(_, _, ref types, ref factory) in INTRINSICS {
        assert_eq!(types.len(), param_count(factory));
    }
}
