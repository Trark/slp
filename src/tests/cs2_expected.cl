
void myFunc(uint x)
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
}
