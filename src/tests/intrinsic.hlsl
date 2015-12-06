
Buffer<uint4> g_roBuffer : register(t0);
RWBuffer<uint4> g_rwBuffer : register(u0);

void test_buffer(uint3 dtid)
{
    uint4 read0 = g_roBuffer[dtid.x];
    uint4 read1 = g_rwBuffer[dtid.x];
    g_rwBuffer[dtid.x] = read0 + read1;
}

struct testStruct {
    float4 value;
};

StructuredBuffer<testStruct> g_roStructuredBuffer : register(t1);
RWStructuredBuffer<testStruct> g_rwStructuredBuffer : register(u1);

void test_structured_buffer(uint3 dtid)
{
    testStruct read0 = g_roStructuredBuffer[dtid.x];
    testStruct read1 = g_rwStructuredBuffer[dtid.x];
    testStruct modified;
    modified.value = read0.value + read1.value;
    g_rwStructuredBuffer[dtid.x] = modified;
}

[numthreads(8, 8, 1)]
void CSMAIN(uint3 dtid : SV_DispatchThreadID)
{
    test_buffer(dtid);
    test_structured_buffer(dtid);
    AllMemoryBarrier();
    AllMemoryBarrierWithGroupSync();
    DeviceMemoryBarrier();
    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    int i = 0;
    int1 i1;
    int2 i2;
    int3 i3;
    int4 i4;
    uint u = 0u;
    uint1 u1;
    uint2 u2;
    uint3 u3;
    uint4 u4;
    float f = 0.0;
    float1 f1;
    float2 f2;
    float3 f3;
    float4 f4;
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
}
