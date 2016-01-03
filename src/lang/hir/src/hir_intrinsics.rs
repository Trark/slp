
use ir::*;

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic0 {
    AllMemoryBarrier,
    AllMemoryBarrierWithGroupSync,
    DeviceMemoryBarrier,
    DeviceMemoryBarrierWithGroupSync,
    GroupMemoryBarrier,
    GroupMemoryBarrierWithGroupSync,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic1 {
    // Unary operations
    PrefixIncrement(Type),
    PrefixDecrement(Type),
    PostfixIncrement(Type),
    PostfixDecrement(Type),
    Plus(Type),
    Minus(Type),
    LogicalNot(Type),
    BitwiseNot(Type),

    AbsI,
    AbsI2,
    AbsI3,
    AbsI4,
    AbsF,
    AbsF2,
    AbsF3,
    AbsF4,

    AsIntU,
    AsIntU2,
    AsIntU3,
    AsIntU4,
    AsIntF,
    AsIntF2,
    AsIntF3,
    AsIntF4,

    AsUIntI,
    AsUIntI2,
    AsUIntI3,
    AsUIntI4,
    AsUIntF,
    AsUIntF2,
    AsUIntF3,
    AsUIntF4,

    AsFloatI,
    AsFloatI2,
    AsFloatI3,
    AsFloatI4,
    AsFloatU,
    AsFloatU2,
    AsFloatU3,
    AsFloatU4,
    AsFloatF,
    AsFloatF2,
    AsFloatF3,
    AsFloatF4,

    Exp,
    Exp2,
    Exp3,
    Exp4,

    F16ToF32,
    F32ToF16,

    Floor,
    Floor2,
    Floor3,
    Floor4,

    IsNaN,
    IsNaN2,
    IsNaN3,
    IsNaN4,

    Length1,
    Length2,
    Length3,
    Length4,

    Normalize1,
    Normalize2,
    Normalize3,
    Normalize4,

    SignI,
    SignI2,
    SignI3,
    SignI4,
    SignF,
    SignF2,
    SignF3,
    SignF4,

    Sqrt,
    Sqrt2,
    Sqrt3,
    Sqrt4,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic2 {
    // Binary operations
    Add(DataType),
    Subtract(DataType),
    Multiply(DataType),
    Divide(DataType),
    Modulus(DataType),
    LeftShift(DataType),
    RightShift(DataType),
    BitwiseAnd(DataType),
    BitwiseOr(DataType),
    BitwiseXor(DataType),
    BooleanAnd(DataType),
    BooleanOr(DataType),
    LessThan(DataType),
    LessEqual(DataType),
    GreaterThan(DataType),
    GreaterEqual(DataType),
    Equality(DataType),
    Inequality(DataType),
    Assignment(Type),
    SumAssignment(DataType),
    DifferenceAssignment(DataType),
    ProductAssignment(DataType),
    QuotientAssignment(DataType),
    RemainderAssignment(DataType),

    AsDouble,

    Cross,

    Distance1,
    Distance2,
    Distance3,
    Distance4,

    DotI1,
    DotI2,
    DotI3,
    DotI4,
    DotF1,
    DotF2,
    DotF3,
    DotF4,

    MinI,
    MinI2,
    MinI3,
    MinI4,
    MinF,
    MinF2,
    MinF3,
    MinF4,

    MaxI,
    MaxI2,
    MaxI3,
    MaxI4,
    MaxF,
    MaxF2,
    MaxF3,
    MaxF4,

    Step,
    Step2,
    Step3,
    Step4,

    BufferLoad(DataType),
    RWBufferLoad(DataType),
    StructuredBufferLoad(StructuredType),
    RWStructuredBufferLoad(StructuredType),
    RWTexture2DLoad(DataType),

    // ByteAddressBuffer methods
    ByteAddressBufferLoad,
    ByteAddressBufferLoad2,
    ByteAddressBufferLoad3,
    ByteAddressBufferLoad4,
    RWByteAddressBufferLoad,
    RWByteAddressBufferLoad2,
    RWByteAddressBufferLoad3,
    RWByteAddressBufferLoad4,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Intrinsic3 {
    ClampI,
    ClampI2,
    ClampI3,
    ClampI4,
    ClampF,
    ClampF2,
    ClampF3,
    ClampF4,

    Sincos,
    Sincos2,
    Sincos3,
    Sincos4,

    // ByteAddressBuffer methods
    RWByteAddressBufferStore,
    RWByteAddressBufferStore2,
    RWByteAddressBufferStore3,
    RWByteAddressBufferStore4,
}

pub trait Intrinsic {
    fn get_return_type(&self) -> ExpressionType;
}

impl Intrinsic for Intrinsic0 {
    fn get_return_type(&self) -> ExpressionType {
        match *self {
            Intrinsic0::AllMemoryBarrier => Type::void().to_rvalue(),
            Intrinsic0::AllMemoryBarrierWithGroupSync => Type::void().to_rvalue(),
            Intrinsic0::DeviceMemoryBarrier => Type::void().to_rvalue(),
            Intrinsic0::DeviceMemoryBarrierWithGroupSync => Type::void().to_rvalue(),
            Intrinsic0::GroupMemoryBarrier => Type::void().to_rvalue(),
            Intrinsic0::GroupMemoryBarrierWithGroupSync => Type::void().to_rvalue(),
        }
    }
}

impl Intrinsic for Intrinsic1 {
    fn get_return_type(&self) -> ExpressionType {
        match *self {
            Intrinsic1::PrefixIncrement(ref ty) => ty.to_lvalue(),
            Intrinsic1::PrefixDecrement(ref ty) => ty.to_lvalue(),
            Intrinsic1::PostfixIncrement(ref ty) => ty.to_lvalue(),
            Intrinsic1::PostfixDecrement(ref ty) => ty.to_lvalue(),
            Intrinsic1::Plus(ref ty) => ty.to_rvalue(),
            Intrinsic1::Minus(ref ty) => ty.to_rvalue(),
            Intrinsic1::LogicalNot(ref ty) => {
                match ty.0 {
                    TypeLayout::Scalar(_) => Type::bool().to_rvalue(),
                    TypeLayout::Vector(_, x) => Type::booln(x).to_rvalue(),
                    _ => panic!("invalid logical not intrinsic"),
                }
            }
            Intrinsic1::BitwiseNot(ref ty) => ty.to_rvalue(),
            Intrinsic1::AbsI => Type::int().to_rvalue(),
            Intrinsic1::AbsI2 => Type::intn(2).to_rvalue(),
            Intrinsic1::AbsI3 => Type::intn(3).to_rvalue(),
            Intrinsic1::AbsI4 => Type::intn(4).to_rvalue(),
            Intrinsic1::AbsF => Type::float().to_rvalue(),
            Intrinsic1::AbsF2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::AbsF3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::AbsF4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::AsIntU => Type::int().to_rvalue(),
            Intrinsic1::AsIntU2 => Type::intn(2).to_rvalue(),
            Intrinsic1::AsIntU3 => Type::intn(3).to_rvalue(),
            Intrinsic1::AsIntU4 => Type::intn(4).to_rvalue(),
            Intrinsic1::AsIntF => Type::int().to_rvalue(),
            Intrinsic1::AsIntF2 => Type::intn(2).to_rvalue(),
            Intrinsic1::AsIntF3 => Type::intn(3).to_rvalue(),
            Intrinsic1::AsIntF4 => Type::intn(4).to_rvalue(),
            Intrinsic1::AsUIntI => Type::uint().to_rvalue(),
            Intrinsic1::AsUIntI2 => Type::uintn(2).to_rvalue(),
            Intrinsic1::AsUIntI3 => Type::uintn(3).to_rvalue(),
            Intrinsic1::AsUIntI4 => Type::uintn(4).to_rvalue(),
            Intrinsic1::AsUIntF => Type::uint().to_rvalue(),
            Intrinsic1::AsUIntF2 => Type::uintn(2).to_rvalue(),
            Intrinsic1::AsUIntF3 => Type::uintn(3).to_rvalue(),
            Intrinsic1::AsUIntF4 => Type::uintn(4).to_rvalue(),
            Intrinsic1::AsFloatI => Type::float().to_rvalue(),
            Intrinsic1::AsFloatI2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::AsFloatI3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::AsFloatI4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::AsFloatU => Type::float().to_rvalue(),
            Intrinsic1::AsFloatU2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::AsFloatU3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::AsFloatU4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::AsFloatF => Type::float().to_rvalue(),
            Intrinsic1::AsFloatF2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::AsFloatF3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::AsFloatF4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::Exp => Type::float().to_rvalue(),
            Intrinsic1::Exp2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::Exp3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::Exp4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::F16ToF32 => Type::float().to_rvalue(),
            Intrinsic1::F32ToF16 => Type::uint().to_rvalue(),
            Intrinsic1::Floor => Type::float().to_rvalue(),
            Intrinsic1::Floor2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::Floor3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::Floor4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::IsNaN => Type::bool().to_rvalue(),
            Intrinsic1::IsNaN2 => Type::booln(2).to_rvalue(),
            Intrinsic1::IsNaN3 => Type::booln(3).to_rvalue(),
            Intrinsic1::IsNaN4 => Type::booln(4).to_rvalue(),
            Intrinsic1::Length1 => Type::float().to_rvalue(),
            Intrinsic1::Length2 => Type::float().to_rvalue(),
            Intrinsic1::Length3 => Type::float().to_rvalue(),
            Intrinsic1::Length4 => Type::float().to_rvalue(),
            Intrinsic1::Normalize1 => Type::floatn(1).to_rvalue(),
            Intrinsic1::Normalize2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::Normalize3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::Normalize4 => Type::floatn(4).to_rvalue(),
            Intrinsic1::SignI => Type::int().to_rvalue(),
            Intrinsic1::SignI2 => Type::intn(2).to_rvalue(),
            Intrinsic1::SignI3 => Type::intn(3).to_rvalue(),
            Intrinsic1::SignI4 => Type::intn(4).to_rvalue(),
            Intrinsic1::SignF => Type::int().to_rvalue(),
            Intrinsic1::SignF2 => Type::intn(2).to_rvalue(),
            Intrinsic1::SignF3 => Type::intn(3).to_rvalue(),
            Intrinsic1::SignF4 => Type::intn(4).to_rvalue(),
            Intrinsic1::Sqrt => Type::float().to_rvalue(),
            Intrinsic1::Sqrt2 => Type::floatn(2).to_rvalue(),
            Intrinsic1::Sqrt3 => Type::floatn(3).to_rvalue(),
            Intrinsic1::Sqrt4 => Type::floatn(4).to_rvalue(),
        }
    }
}

impl Intrinsic for Intrinsic2 {
    fn get_return_type(&self) -> ExpressionType {
        match *self {
            Intrinsic2::Add(ref dty) |
            Intrinsic2::Subtract(ref dty) |
            Intrinsic2::Multiply(ref dty) |
            Intrinsic2::Divide(ref dty) |
            Intrinsic2::Modulus(ref dty) |
            Intrinsic2::LeftShift(ref dty) |
            Intrinsic2::RightShift(ref dty) |
            Intrinsic2::BitwiseAnd(ref dty) |
            Intrinsic2::BitwiseOr(ref dty) |
            Intrinsic2::BitwiseXor(ref dty) |
            Intrinsic2::BooleanAnd(ref dty) |
            Intrinsic2::BooleanOr(ref dty) => {
                // dty is the type of the arguments operated on, which
                // is the same as the return value
                Type::from_data(dty.clone()).to_rvalue()
            }
            Intrinsic2::LessThan(ref dty) |
            Intrinsic2::LessEqual(ref dty) |
            Intrinsic2::GreaterThan(ref dty) |
            Intrinsic2::GreaterEqual(ref dty) |
            Intrinsic2::Equality(ref dty) |
            Intrinsic2::Inequality(ref dty) => {
                // dty is the type of the arguments operated on, so the return
                // value is a bool with the same dimensions as dty
                Type::from_data(dty.clone()).transform_scalar(ScalarType::Bool).to_rvalue()
            }
            Intrinsic2::Assignment(ref ty) => {
                // ty is the type of the assigned value, so it the return value
                ty.clone().to_lvalue()
            }
            Intrinsic2::SumAssignment(ref dty) |
            Intrinsic2::DifferenceAssignment(ref dty) |
            Intrinsic2::ProductAssignment(ref dty) |
            Intrinsic2::QuotientAssignment(ref dty) |
            Intrinsic2::RemainderAssignment(ref dty) => {
                // dty is the type of the assigned value, so it the return value
                Type::from_data(dty.clone()).to_lvalue()
            }
            Intrinsic2::AsDouble => Type::double().to_rvalue(),
            Intrinsic2::Cross => Type::floatn(3).to_rvalue(),
            Intrinsic2::Distance1 => Type::float().to_rvalue(),
            Intrinsic2::Distance2 => Type::float().to_rvalue(),
            Intrinsic2::Distance3 => Type::float().to_rvalue(),
            Intrinsic2::Distance4 => Type::float().to_rvalue(),
            Intrinsic2::DotI1 => Type::int().to_rvalue(),
            Intrinsic2::DotI2 => Type::int().to_rvalue(),
            Intrinsic2::DotI3 => Type::int().to_rvalue(),
            Intrinsic2::DotI4 => Type::int().to_rvalue(),
            Intrinsic2::DotF1 => Type::float().to_rvalue(),
            Intrinsic2::DotF2 => Type::float().to_rvalue(),
            Intrinsic2::DotF3 => Type::float().to_rvalue(),
            Intrinsic2::DotF4 => Type::float().to_rvalue(),
            Intrinsic2::MinI => Type::int().to_rvalue(),
            Intrinsic2::MinI2 => Type::intn(2).to_rvalue(),
            Intrinsic2::MinI3 => Type::intn(3).to_rvalue(),
            Intrinsic2::MinI4 => Type::intn(4).to_rvalue(),
            Intrinsic2::MinF => Type::float().to_rvalue(),
            Intrinsic2::MinF2 => Type::floatn(2).to_rvalue(),
            Intrinsic2::MinF3 => Type::floatn(3).to_rvalue(),
            Intrinsic2::MinF4 => Type::floatn(4).to_rvalue(),
            Intrinsic2::MaxI => Type::int().to_rvalue(),
            Intrinsic2::MaxI2 => Type::intn(2).to_rvalue(),
            Intrinsic2::MaxI3 => Type::intn(3).to_rvalue(),
            Intrinsic2::MaxI4 => Type::intn(4).to_rvalue(),
            Intrinsic2::MaxF => Type::float().to_rvalue(),
            Intrinsic2::MaxF2 => Type::floatn(2).to_rvalue(),
            Intrinsic2::MaxF3 => Type::floatn(3).to_rvalue(),
            Intrinsic2::MaxF4 => Type::floatn(4).to_rvalue(),
            Intrinsic2::Step => Type::float().to_rvalue(),
            Intrinsic2::Step2 => Type::floatn(2).to_rvalue(),
            Intrinsic2::Step3 => Type::floatn(3).to_rvalue(),
            Intrinsic2::Step4 => Type::floatn(4).to_rvalue(),
            Intrinsic2::BufferLoad(ref dty) => Type::from_data(dty.clone()).to_rvalue(),
            Intrinsic2::RWBufferLoad(ref dty) => Type::from_data(dty.clone()).to_rvalue(),
            Intrinsic2::StructuredBufferLoad(ref sty) => {
                Type::from_structured(sty.clone()).to_rvalue()
            }
            Intrinsic2::RWStructuredBufferLoad(ref sty) => {
                Type::from_structured(sty.clone()).to_rvalue()
            }
            Intrinsic2::RWTexture2DLoad(ref dty) => Type::from_data(dty.clone()).to_rvalue(),
            Intrinsic2::ByteAddressBufferLoad => Type::uint().to_rvalue(),
            Intrinsic2::ByteAddressBufferLoad2 => Type::uintn(2).to_rvalue(),
            Intrinsic2::ByteAddressBufferLoad3 => Type::uintn(3).to_rvalue(),
            Intrinsic2::ByteAddressBufferLoad4 => Type::uintn(4).to_rvalue(),
            Intrinsic2::RWByteAddressBufferLoad => Type::uint().to_rvalue(),
            Intrinsic2::RWByteAddressBufferLoad2 => Type::uintn(2).to_rvalue(),
            Intrinsic2::RWByteAddressBufferLoad3 => Type::uintn(3).to_rvalue(),
            Intrinsic2::RWByteAddressBufferLoad4 => Type::uintn(4).to_rvalue(),
        }
    }
}

impl Intrinsic for Intrinsic3 {
    fn get_return_type(&self) -> ExpressionType {
        match *self {
            Intrinsic3::ClampI => Type::int().to_rvalue(),
            Intrinsic3::ClampI2 => Type::intn(2).to_rvalue(),
            Intrinsic3::ClampI3 => Type::intn(3).to_rvalue(),
            Intrinsic3::ClampI4 => Type::intn(4).to_rvalue(),
            Intrinsic3::ClampF => Type::float().to_rvalue(),
            Intrinsic3::ClampF2 => Type::floatn(2).to_rvalue(),
            Intrinsic3::ClampF3 => Type::floatn(3).to_rvalue(),
            Intrinsic3::ClampF4 => Type::floatn(4).to_rvalue(),
            Intrinsic3::Sincos => Type::void().to_rvalue(),
            Intrinsic3::Sincos2 => Type::void().to_rvalue(),
            Intrinsic3::Sincos3 => Type::void().to_rvalue(),
            Intrinsic3::Sincos4 => Type::void().to_rvalue(),
            Intrinsic3::RWByteAddressBufferStore => Type::void().to_rvalue(),
            Intrinsic3::RWByteAddressBufferStore2 => Type::void().to_rvalue(),
            Intrinsic3::RWByteAddressBufferStore3 => Type::void().to_rvalue(),
            Intrinsic3::RWByteAddressBufferStore4 => Type::void().to_rvalue(),
        }
    }
}
