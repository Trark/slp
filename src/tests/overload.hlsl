
void testIntOrUInt(int x)
{
}

void testIntOrUInt(uint x)
{
}

void testIntOrFloat(int x)
{
}

void testIntOrFloat(float x)
{
}

void testBoolOrFloat(bool x)
{
}

void testBoolOrFloat(float x)
{
}

void testIntOrUInt1(int x)
{
}

void testIntOrUInt1(uint1 x)
{
}

void testIntOrUInt4(int x)
{
}

void testIntOrUInt4(uint4 x)
{
}

void testIntOrInt3(int x)
{
}

void testIntOrInt3(int3 x)
{
}

void testVec3(float x, float y, float z)
{
}

void testVec3(float2 x, float2 y, float2 z)
{
}

void testVec3(float3 x, float3 y, float3 z)
{
}

[numthreads(8, 8, 1)]
void CSMAIN(uint3 dtid : SV_DispatchThreadID)
{
    testIntOrUInt((int)0);
    testIntOrUInt((uint)0);

    testIntOrFloat((int)0);
    testIntOrFloat(0.0);
    testIntOrFloat(0);
    testIntOrFloat(0u);

    testBoolOrFloat((bool)0);
    testBoolOrFloat(0.0);
    testBoolOrFloat(0);

    testIntOrUInt1((int)0);
    testIntOrUInt1((uint)0);
    testIntOrUInt1(0u);
    testIntOrUInt1((uint4)0u);
    testIntOrUInt1((uint4)0);
    testIntOrUInt4((int)0);
    testIntOrUInt4((uint)0);
    testIntOrUInt4((uint1)0);
    testIntOrUInt4((uint4)0);
    testIntOrInt3(0);
    testIntOrInt3(0u);
    testIntOrInt3((int)0);
    testIntOrInt3(int3(0, 1, 2));

    max(1.0f, 2.0f);
    max(float2(3.0f, 7.0f), float2(4.0f, 3.0f));
    max(3.0f, float3(4.0f, 3.0f, 3.5f));
    max(float2(3.0f, 7.0f), float3(4.0f, 3.0f, 3.5f));

    testVec3(2.0f, float2(1.0f, 3.0f), float3(2.0f, 7.0f, 2.9f));
}
