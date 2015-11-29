
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

void outTest(out float x)
{
    x = 4.f;
}

void outTest(float x, out float y, float z)
{
    y = x + z;
}

[numthreads(8, 8, 1)]
void CSMAIN(uint3 dtid : SV_DispatchThreadID)
{
    uint myFunc_1;
    uint alias_var = 2u;
    int index = dtid.x + g_offset;
    myFunc_1 = g_myInBuffer.Load(index);
    g_myOutBuffer[index] = myFunc_1;
    uint testStruct = 0;
    bool cond = true;
    if (cond)
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
        bool p = false;
        myFunc((uint)p);
    }
    int y = 10;
    while (y > 0)
    {
        y--;
        myFunc((uint)y);
    }
    float u = y + 5.4f;
    outTest(u);
    outTest(4.5, u, u > 4 ? 3.4 : u);
    float vals[3];
    vals[0] = 0.0f;
    vals[1] = 1.0f;
    vals[2] = 2.0f;
    float val0 = vals[0] + 1.0f;
    outTest(vals[2]);
}
