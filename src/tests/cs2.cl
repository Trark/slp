
int2 cast_uint2_to_int2(uint2 from)
{
	int2 to;
	to[0] = from[0];
	to[1] = from[1];
	return to;
}

int __constant g_myFour = (int)4;

float4 __local sdata[32];

uint __constant g_myArray1[4] = { 5u, 8u, 3u, 2u };

int2 __constant g_myArray2[4] = { (int2)((int)2, (int)3), (int2)((int)9, (int)0), { (int)4, (int)5 }, (int2)((int)6, (int)7) };

struct myStruct
{
	float4 pos;
	uint index;
};

struct testStruct_0
{
	uint index;
	float x;
	float y[2];
};

struct myConstants_t
{
	uint g_offset;
	float g_x;
	float g_uv[2];
};

void myFunc_0(uint x)
{
}

void myFunc_1(float x)
{
	x = 4.0f;
}

void outTest_0(__private float* x)
{
	*x = 4.0f;
}

uint outTest_1(float x, __private float* y, float z)
{
	return (uint)(*y = x + z);
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel
(
	__constant struct myConstants_t* myConstants,
	__global uint* g_myInBuffer,
	__global uint* g_myOutBuffer
)
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	uint myFunc_1_0;
	uint alias_var = 2u << 1u;
	int index = (int)(dtid.x + myConstants->g_offset);
	myFunc_1_0 = g_myInBuffer[index];
	g_myOutBuffer[index] = myFunc_1_0;
	uint testStruct = (uint)0;
	bool cond = true;
	if (cond)
	{
		float4 alias_var_0 = 2.0f * (float4)(1.0f, 2.4f, 0.3f, 3.4f);
		float4 receive = alias_var_0 * 4.0f;
		uint testStruct_1 = (uint)(1 & 4);
		struct testStruct_0 data;
	}
	myFunc_1(4.0f);
	myFunc_1(myConstants->g_x);
	myFunc_1(myConstants->g_uv[(int)1]);
	for (uint x = 4u; x < 10u; ++x)
	{
		myFunc_0(x);
		bool p = false;
		myFunc_0((uint)p);
	}
	int y = (int)10;
	while (y > g_myFour)
	{
		y--;
		myFunc_0((uint)y);
	}
	float u = (float)y + 5.4f;
	outTest_0(&u);
	float returnValue = (float)outTest_1(4.5f, &u, u > (float)4 ? 3.4f : u);
	float vals[3];
	vals[(int)0] = 0.0f;
	vals[(int)1] = INFINITY;
	vals[(int)2] = -INFINITY;
	float val0 = vals[(int)0] + 1.0f;
	outTest_0(&vals[(int)2]);
	myFunc_0((uint)g_myFour);
	uint2 cast_from = { g_myArray1[(int)1], (uint)g_myArray2[(int)3].x };
	int2 cast_t0 = cast_uint2_to_int2(cast_from);
	float s = sdata[(int)0].x;
	float t1;
	float arr1[3];
	float arr2[4];
	uint s_0 = 3u;
	uint arr1_0[2];
	for (uint t2 = 6u; s_0 < t2; s_0++)
	{
		myFunc_0(s_0);
		myFunc_1(t1);
		myFunc_0(t2);
		myFunc_1(arr2[(int)0]);
		myFunc_0(arr1_0[(int)0]);
	}
	bool b_0 = true;
	bool b_1 = (bool)(s * (float)b_0);
	if (true || false)
	{
		myFunc_0(1u);
	}
	else
	{
		myFunc_0(2u);
	}
	if (true && true)
	{
		myFunc_0(3u);
	}
	else
	{
		if (false || (true && false))
		{
			myFunc_0(4u);
		}
		else
		{
			myFunc_0(5u);
		}
	}
	for (; false; 0)
	{
		if (1)
		{
			break;
		}
		else
		{
			continue;
		}
	}
}
