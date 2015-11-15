
Buffer<uint> g_myInBuffer : register(t0);
RWBuffer<uint> g_myOutBuffer : register(u0);

struct myStruct
{
    float4 pos;
    uint index;
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
    uint b = 2;
    myFunc_1 = b;
    b = g_myInBuffer.Load(0);
    g_myOutBuffer[b] = myFunc_1;
    {
        float4 b = float4(1.0f, 2.4f, 0.3f, 3.4f);
        uint g = myFunc_1;
    }
    myFunc(4.0f);
}
