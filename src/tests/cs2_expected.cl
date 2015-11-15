
struct myStruct
{
	float4 pos;
	uint index;
};

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
	uint myFunc_1_0;
	uint b = 2;
	myFunc_1_0 = b;
	b = g_myInBuffer[0];
	g_myOutBuffer[b] = myFunc_1_0;
	{
		float4 b_0 = (float4)(1.0f, 2.4f, 0.3f, 3.4f);
		uint g = myFunc_1_0;
	}
	myFunc_1(4.0f);
}
