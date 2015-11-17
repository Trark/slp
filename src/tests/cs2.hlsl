
Buffer<uint> g_myInBuffer : register(t0);
RWBuffer<uint> g_myOutBuffer : register(u0);

struct myStruct
{
    float4 pos;
    uint index;
};

struct testStruct_0
{
    uint index;
};

cbuffer myConstants : register(b0)
{
    uint g_offset;
};

void myFunc(uint x)
{
}

void myFunc(float x)
{
    x = 4.f;
}

[numthreads(8, 8, 1)]
void CSMAIN(uint3 dtid : SV_DispatchThreadID)
{
    uint myFunc_1;
    uint alias_var = 2;
    int index = dtid.x + g_offset;
    myFunc_1 = g_myInBuffer.Load(index);
    g_myOutBuffer[index] = myFunc_1;
    uint testStruct = 0;
    if (myFunc_1)
    {
        float4 alias_var = float4(1.0f, 2.4f, 0.3f, 3.4f);
        float4 receive = alias_var;
        uint testStruct = 1;
        testStruct_0 data;
    }
    myFunc(4.0f);
    for (uint x = 4u; x < 10u; ++x)
    {
        myFunc(x);
    }
}
