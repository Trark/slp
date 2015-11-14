
Buffer<uint> g_myInBuffer : register(t0);
RWBuffer<uint> g_myOutBuffer : register(u0);

void myFunc(uint x)
{
}

void myFunc_1(float x)
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
}
