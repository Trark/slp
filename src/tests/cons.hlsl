
static const float s = 0.0;
static const float1 s1 = 1.0;
static const float2 s2 = float2(1.0, 0.0);
static const float3 s3 = float3(1.0, 0.0, 1.0);
static const float4 s4 = float4(0.0, 1.0, 1.0, 0.0);
static const uint u = 0u;
static const uint1 u1 = 1u;
static const uint2 u2 = uint2(1u, 0u);
static const uint3 u3 = uint3(1u, 0u, 1u);
static const uint4 u4 = uint4(0u, 1u, 1u, 0u);

void cons_float4()
{
    float4 target;
    target = float4(1.0, 0.0, 0.0, 1.0);
    target = float4(s1, s1, s1, s1);
    target = float4(s2, s2);
    target = float4(s, s2, s);
    target = float4(s1, s2, s1);
    target = float4(s3, s);
    target = float4(s3, s1);
    target = float4(s, s3);
    target = float4(s1, s3);
    target = float4(s4);
    target = float4(0u, 0, (int)0, 1.0);
    target = float4(u1, u1, u1, u1);
    target = float4(u2, u2);
    target = float4(u, u2, u);
    target = float4(u1, u2, u1);
    target = float4(u3, u);
    target = float4(u3, u1);
    target = float4(u, u3);
    target = float4(u1, u3);
    target = float4(u4);
}

[numthreads(8, 8, 1)]
void CSMAIN()
{
    cons_float4();
}
