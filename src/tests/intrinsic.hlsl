
Buffer<uint4> g_roBuffer : register(t0);
RWBuffer<uint4> g_rwBuffer : register(u0);

void test_buffer(uint3 dtid)
{
    uint4 read0 = g_roBuffer[dtid.x];
    uint4 read2 = g_roBuffer.Load(dtid.x);
    uint4 read1 = g_rwBuffer[dtid.x];
    uint4 read3 = g_rwBuffer.Load(dtid.x);
    g_rwBuffer[dtid.x] = read0 + read1 + read2 + read3;
}

struct testStruct {
    float4 value;
};

StructuredBuffer<testStruct> g_roStructuredBuffer : register(t1);
RWStructuredBuffer<testStruct> g_rwStructuredBuffer : register(u1);

void test_structured_buffer(uint3 dtid)
{
    testStruct read0 = g_roStructuredBuffer[dtid.x];
    testStruct read2 = g_roStructuredBuffer.Load(dtid.x);
    testStruct read1 = g_rwStructuredBuffer[dtid.x];
    testStruct read3 = g_rwStructuredBuffer.Load(dtid.x);
    testStruct modified;
    modified.value = read0.value + read1.value + read2.value + read3.value;
    g_rwStructuredBuffer[dtid.x] = modified;
}

RWTexture2D<float4> g_rwRTexture2DFloat : register(u2);
RWTexture2D<int4> g_rwRTexture2DInt : register(u3);
RWTexture2D<uint4> g_rwRTexture2DUInt : register(u4);

void test_texture_2d(uint3 dtid)
{
    int2 coord;
    coord.x = dtid.x;
    coord.y = dtid.y;
    float4 read_load_f = g_rwRTexture2DFloat.Load(coord);
    int4 read_load_i = g_rwRTexture2DInt.Load(coord);
    uint4 read_load_ui = g_rwRTexture2DUInt.Load(coord);
}

ByteAddressBuffer g_roRawBuffer : register(t5);
RWByteAddressBuffer g_rwRawBuffer : register(u5);

void test_byte_address_buffer(uint3 dtid)
{
    uint ro1 = g_roRawBuffer.Load(64u * dtid.x);
    uint2 ro2 = g_roRawBuffer.Load2(64u * dtid.x + 16u);
    uint3 ro3 = g_roRawBuffer.Load3(64u * dtid.x + 32u);
    uint4 ro4 = g_roRawBuffer.Load4(64u * dtid.x + 48u);
    uint rw1 = g_rwRawBuffer.Load(64u * dtid.x);
    uint2 rw2 = g_rwRawBuffer.Load2(64u * dtid.x + 16u);
    uint3 rw3 = g_rwRawBuffer.Load3(64u * dtid.x + 32u);
    uint4 rw4 = g_rwRawBuffer.Load4(64u * dtid.x + 48u);
    g_rwRawBuffer.Store(64u * dtid.x, ro1 + rw1);
    g_rwRawBuffer.Store2(64u * dtid.x + 16u, ro2 + rw2);
    g_rwRawBuffer.Store3(64u * dtid.x + 32u, ro3 + rw3);
    g_rwRawBuffer.Store4(64u * dtid.x + 48u, ro4 + rw4);
}

[numthreads(8, 8, 1)]
void CSMAIN(uint3 dtid : SV_DispatchThreadID)
{
    test_buffer(dtid);
    test_structured_buffer(dtid);
    test_texture_2d(dtid);
    test_byte_address_buffer(dtid);
    AllMemoryBarrier();
    AllMemoryBarrierWithGroupSync();
    DeviceMemoryBarrier();
    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    int i = 4;
    int j;
    j = i++;
    j = ++i;
    j = i--;
    j = --i;
    j = +i;
    j = -i;
    j = ~i;
    j += i;
    j -= i;
    j *= i;
    j /= i;
    j %= i;
    int1 i1;
    int2 i2;
    int3 i3;
    int4 i4;
    uint u = 0u;
    uint1 u1;
    uint2 u2;
    uint3 u3;
    uint4 u4;
    float f = 3.0;
    j += f;
    j -= f;
    j *= f;
    j /= f;
    j %= f;
    float1 f1;
    float2 f2;
    float3 f3;
    float4 f4;
    bool b = !i;
    int asint_0 = asint(u);
    int1 asint_1 = asint(u1);
    int2 asint_2 = asint(u2);
    int3 asint_3 = asint(u3);
    int4 asint_4 = asint(u4);
    int asint_5 = asint(f);
    int1 asint_6 = asint(f1);
    int2 asint_7 = asint(f2);
    int3 asint_8 = asint(f3);
    int4 asint_9 = asint(f4);
    uint asuint_0 = asuint(i);
    uint1 asuint_1 = asuint(i1);
    uint2 asuint_2 = asuint(i2);
    uint3 asuint_3 = asuint(i3);
    uint4 asuint_4 = asuint(i4);
    uint asuint_5 = asuint(f);
    uint1 asuint_6 = asuint(f1);
    uint2 asuint_7 = asuint(f2);
    uint3 asuint_8 = asuint(f3);
    uint4 asuint_9 = asuint(f4);
    float asfloat_0 = asfloat(i);
    float1 asfloat_1 = asfloat(i1);
    float2 asfloat_2 = asfloat(i2);
    float3 asfloat_3 = asfloat(i3);
    float4 asfloat_4 = asfloat(i4);
    float asfloat_5 = asfloat(u);
    float1 asfloat_6 = asfloat(u1);
    float2 asfloat_7 = asfloat(u2);
    float3 asfloat_8 = asfloat(u3);
    float4 asfloat_9 = asfloat(u4);
    float asfloat_10 = asfloat(f);
    float1 asfloat_11 = asfloat(f1);
    float2 asfloat_12 = asfloat(f2);
    float3 asfloat_13 = asfloat(f3);
    float4 asfloat_14 = asfloat(f4);
    int ix;
    int1 ix1;
    int2 ix2;
    int3 ix3;
    int4 ix4;
    int iy;
    int1 iy1;
    int2 iy2;
    int3 iy3;
    int4 iy4;
    int iz;
    int1 iz1;
    int2 iz2;
    int3 iz3;
    int4 iz4;
    float fx;
    float1 fx1;
    float2 fx2;
    float3 fx3;
    float4 fx4;
    float fy;
    float1 fy1;
    float2 fy2;
    float3 fy3;
    float4 fy4;
    float fz;
    float1 fz1;
    float2 fz2;
    float3 fz3;
    float4 fz4;
    int clamp_0 = clamp(ix, iy, iz);
    int1 clamp_1 = clamp(ix1, iy1, iz1);
    int2 clamp_2 = clamp(ix2, iy2, iz2);
    int3 clamp_3 = clamp(ix3, iy3, iz3);
    int4 clamp_4 = clamp(ix4, iy4, iz4);
    float clamp_5 = clamp(fx, fy, fz);
    float1 clamp_6 = clamp(fx1, fy1, fz1);
    float2 clamp_7 = clamp(fx2, fy2, fz2);
    float3 clamp_8 = clamp(fx3, fy3, fz3);
    float4 clamp_9 = clamp(fx4, fy4, fz4);
    float3 cross_0 = cross(fx3, fx3);
    float distance_1 = distance(fx1, fy1);
    float distance_2 = distance(fx2, fy2);
    float distance_3 = distance(fx3, fy3);
    float distance_4 = distance(fx4, fy4);
    float dot_f1 = dot(fx1, fy1);
    float dot_f2 = dot(fx2, fy2);
    float dot_f3 = dot(fx3, fy3);
    float dot_f4 = dot(fx4, fy4);
    bool isnan_s = isnan(fx);
    float length_f1 = length(fx1);
    float length_f2 = length(fx2);
    float length_f3 = length(fx3);
    float length_f4 = length(fx4);
    float1 normalize_f1 = normalize(fx1);
    float2 normalize_f2 = normalize(fx2);
    float3 normalize_f3 = normalize(fx3);
    float4 normalize_f4 = normalize(fx4);
    int sign_s = sign(fx);
    int1 sign_f1 = sign(fx1);
    int2 sign_f2 = sign(fx2);
    int3 sign_f3 = sign(fx3);
    int4 sign_f4 = sign(fx4);
    float sqrt_s = sqrt(fx);
    float1 sqrt_f1 = sqrt(fx1);
    float2 sqrt_f2 = sqrt(fx2);
    float3 sqrt_f3 = sqrt(fx3);
    float4 sqrt_f4 = sqrt(fx4);
    float min_fs = min(fx, fy);
    float1 min_f1 = min(fx1, fy1);
    float2 min_f2 = min(fx2, fy2);
    float3 min_f3 = min(fx3, fy3);
    float4 min_f4 = min(fx4, fy4);
    int min_is = min(ix, iy);
    int1 min_i1 = min(ix1, iy1);
    int2 min_i2 = min(ix2, iy2);
    int3 min_i3 = min(ix3, iy3);
    int4 min_i4 = min(ix4, iy4);
    float max_fs = max(fx, fy);
    float1 max_f1 = max(fx1, fy1);
    float2 max_f2 = max(fx2, fy2);
    float3 max_f3 = max(fx3, fy3);
    float4 max_f4 = max(fx4, fy4);
    int max_is = max(ix, iy);
    int1 max_i1 = max(ix1, iy1);
    int2 max_i2 = max(ix2, iy2);
    int3 max_i3 = max(ix3, iy3);
    int4 max_i4 = max(ix4, iy4);
    uint packed_half = f32tof16(fx);
    float unpacked_half = f16tof32(packed_half);
    float step_fs = step(fx, fy);
    float1 step_f1 = step(fx1, fy1);
    float2 step_f2 = step(fx2, fy2);
    float3 step_f3 = step(fx3, fy3);
    float4 step_f4 = step(fx4, fy4);
    float exp_fs = exp(fx);
    float1 exp_f1 = exp(fx1);
    float2 exp_f2 = exp(fx2);
    float3 exp_f3 = exp(fx3);
    float4 exp_f4 = exp(fx4);
    int abs_is = abs(ix);
    int1 abs_i1 = abs(ix1);
    int2 abs_i2 = abs(ix2);
    int3 abs_i3 = abs(ix3);
    int4 abs_i4 = abs(ix4);
    float abs_fs = abs(fx);
    float1 abs_f1 = abs(fx1);
    float2 abs_f2 = abs(fx2);
    float3 abs_f3 = abs(fx3);
    float4 abs_f4 = abs(fx4);
    (void)sincos(fx, fy, fz);
    (void)sincos(fx2, fy2, fz2);
    (void)sincos(fx3, fy3, fz3);
    (void)sincos(fx4, fy4, fz4);
}
