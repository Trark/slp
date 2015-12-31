
void cons_float4()
{
    float4 target;
    float s;
    float1 s1;
    float2 s2;
    float3 s3;
    float4 s4;
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
    uint u;
    uint1 u1;
    uint2 u2;
    uint3 u3;
    uint4 u4;
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
