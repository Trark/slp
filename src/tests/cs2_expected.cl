
void myFunc_0(uint x)
{
}

void myFunc_1(float x)
{
	x = 4.0f;
}

kernel void MyKernel(__constant uint* g_myInBuffer, __global uint* g_myOutBuffer)
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	uint a;
	uint b = 2;
	a = b;
	b = g_myInBuffer[0];
	g_myOutBuffer[b] = a;
	float4 p = (float4)(1.0f, 2.4f, 0.3f, 3.4f);
	myFunc_1(4.0f);
}
