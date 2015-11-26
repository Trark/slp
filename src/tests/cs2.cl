
struct myStruct
{
	float4 pos;
	uint index;
};

struct testStruct_0
{
	uint index;
};

struct myConstants_t
{
	uint g_offset;
};

void myFunc_0(uint x)
{
}

void myFunc_1(float x)
{
	x = 4.0f;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel(__constant struct myConstants_t* myConstants, __constant uint* g_myInBuffer, __global uint* g_myOutBuffer)
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	uint myFunc_1_0;
	uint alias_var = 2u;
	int index = (int)(dtid.x + myConstants->g_offset);
	myFunc_1_0 = g_myInBuffer[index];
	g_myOutBuffer[index] = myFunc_1_0;
	uint testStruct = (uint)0;
	bool cond = true;
	if (cond)
	{
		float4 alias_var_0 = (float4)(1.0f, 2.4f, 0.3f, 3.4f);
		float4 receive = alias_var_0;
		uint testStruct_1 = (uint)1;
		struct testStruct_0 data;
	}
	myFunc_1(4.0f);
	for (uint x = 4u; x < 10u; ++x)
	{
		myFunc_0(x);
		bool p = false;
		myFunc_0((uint)p);
	}
	int y = (int)10;
	while (y > (int)0)
	{
		y--;
		myFunc_0((uint)y);
	}
	float u = (float)y + 5.4f;
}
