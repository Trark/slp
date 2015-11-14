
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
    uint a;
    uint b = 2;
    a = b;
    b = g_myInBuffer.Load(0);
    g_myOutBuffer[b] = a;
    float4 p = float4(1.0f, 2.4f, 0.3f, 3.4f);
    myFunc(4.0f);
}
