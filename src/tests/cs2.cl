
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

struct __globals
{
	__constant struct myConstants_t* myConstants;
	__global uint* g_myInBuffer;
	__global uint* g_myOutBuffer;
};

void myFunc_0(__private struct __globals* globals, uint x)
{
}

void myFunc_1(__private struct __globals* globals, float x)
{
	x = 4.0f;
}

void outTest_0(__private struct __globals* globals, __private float* x)
{
	*x = 4.0f;
}

void outTest_1(__private struct __globals* globals, float x, __private float* y, float z)
{
	*y = x + z;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel(__constant struct myConstants_t* myConstants, __global uint* g_myInBuffer, __global uint* g_myOutBuffer)
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	struct __globals __init;
	__init.myConstants = myConstants;
	__init.g_myInBuffer = g_myInBuffer;
	__init.g_myOutBuffer = g_myOutBuffer;
	__private struct __globals* globals = &__init;
	uint myFunc_1_0;
	uint alias_var = 2u;
	int index = (int)(dtid.x + globals->myConstants->g_offset);
	myFunc_1_0 = globals->g_myInBuffer[index];
	globals->g_myOutBuffer[index] = myFunc_1_0;
	uint testStruct = (uint)0;
	bool cond = true;
	if (cond)
	{
		float4 alias_var_0 = (float4)(1.0f, 2.4f, 0.3f, 3.4f);
		float4 receive = alias_var_0;
		uint testStruct_1 = (uint)1;
		struct testStruct_0 data;
	}
	myFunc_1(globals, 4.0f);
	for (uint x = 4u; x < 10u; ++x)
	{
		myFunc_0(globals, x);
		bool p = false;
		myFunc_0(globals, (uint)p);
	}
	int y = (int)10;
	while (y > (int)0)
	{
		y--;
		myFunc_0(globals, (uint)y);
	}
	float u = (float)y + 5.4f;
	outTest_0(globals, &u);
	outTest_1(globals, 4.5f, &u, u > (float)4 ? 3.4f : u);
	float vals[3];
	vals[(int)0] = 0.0f;
	vals[(int)1] = 1.0f;
	vals[(int)2] = 2.0f;
	float val0 = vals[(int)0] + 1.0f;
	outTest_0(globals, &vals[(int)2]);
}
