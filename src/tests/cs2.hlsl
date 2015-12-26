
Buffer<uint> g_myInBuffer : register(t0);
RWBuffer<uint> g_myOutBuffer : register(u0);
Buffer<float> unused_t0 : register(t0);
RWBuffer<float> unused_u0 : register(u0);

static const int g_myFour = 4;
static const int unused_static_constant = -4;

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

cbuffer unused_constants : register(b1)
{
    uint unused_constant;
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

uint outTest(float x, out float y, float z)
{
    return y = x + z;
}

void unused_function()
{
    unused_u0[0] = unused_t0[0] + unused_constant + unused_static_constant;
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
    while (y > g_myFour)
    {
        y--;
        myFunc((uint)y);
    }
    float u = y + 5.4f;
    outTest(u);
    float returnValue = outTest(4.5, u, u > 4 ? 3.4 : u);
    float vals[3];
    vals[0] = 0.0f;
    vals[1] = 1.0f;
    vals[2] = 2.0f;
    float val0 = vals[0] + 1.0f;
    outTest(vals[2]);
    myFunc(g_myFour);
    uint2 cast_from;
    int2 cast_t0 = cast_from;
    float s, t1;
    float arr1[3], arr2[4];
    for (uint s = 3u, arr1[2], t2 = 6u; s < t2; s++)
    {
        myFunc(s);
        myFunc(t1);
        myFunc(t2);
        myFunc(arr2[0]);
        myFunc(arr1[0]);
    }
}
