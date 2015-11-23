
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
}
