
uint4 cast_int_to_uint4(int from)
{
	uint4 to;
	to[0] = from;
	to[1] = from;
	to[2] = from;
	to[3] = from;
	return to;
}

uint4 cast_uint_to_uint4(uint from)
{
	uint4 to;
	to[0] = from;
	to[1] = from;
	to[2] = from;
	to[3] = from;
	return to;
}

float2 cast_float_to_float2(float from)
{
	float2 to;
	to[0] = from;
	to[1] = from;
	return to;
}

float3 cast_float_to_float3(float from)
{
	float3 to;
	to[0] = from;
	to[1] = from;
	to[2] = from;
	return to;
}

void testIntOrUInt_0(int x)
{
}

void testIntOrUInt_1(uint x)
{
}

void testIntOrFloat_0(int x)
{
}

void testIntOrFloat_1(float x)
{
}

void testBoolOrFloat_0(bool x)
{
}

void testBoolOrFloat_1(float x)
{
}

void testIntOrUInt1_0(int x)
{
}

void testIntOrUInt1_1(uint x)
{
}

void testIntOrUInt4_0(int x)
{
}

void testIntOrUInt4_1(uint4 x)
{
}

void testIntOrInt3_0(int x)
{
}

void testIntOrInt3_1(int3 x)
{
}

void testVec3(float2 x, float2 y, float2 z)
{
}

void testOut(__private float* x)
{
	*x = 3.0f;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	testIntOrUInt_0((int)0);
	testIntOrUInt_1((uint)0);
	testIntOrFloat_0((int)0);
	testIntOrFloat_1(0.0f);
	testIntOrFloat_0((int)0);
	testIntOrFloat_0((int)0u);
	testBoolOrFloat_0((bool)0);
	testBoolOrFloat_1(0.0f);
	testBoolOrFloat_0((bool)0);
	testIntOrUInt1_0((int)0);
	testIntOrUInt1_1((uint)0);
	testIntOrUInt1_1(0u);
	testIntOrUInt1_1(cast_uint_to_uint4(0u).x);
	testIntOrUInt1_1(cast_int_to_uint4(0).x);
	testIntOrUInt4_0((int)0);
	testIntOrUInt4_1(cast_uint_to_uint4((uint)0));
	testIntOrUInt4_1(cast_uint_to_uint4((uint)0));
	testIntOrUInt4_1(cast_int_to_uint4(0));
	testIntOrInt3_0((int)0);
	testIntOrInt3_0((int)0u);
	testIntOrInt3_0((int)0);
	testIntOrInt3_1((int3)((int)0, (int)1, (int)2));
	fmax(1.0f, 2.0f);
	fmax((float2)(3.0f, 7.0f), (float2)(4.0f, 3.0f));
	fmax(cast_float_to_float3(3.0f), (float3)(4.0f, 3.0f, 3.5f));
	fmax((float2)(3.0f, 7.0f), (float3)(4.0f, 3.0f, 3.5f).xy);
	testVec3(cast_float_to_float2(2.0f), (float2)(1.0f, 3.0f), (float3)(2.0f, 7.0f, 2.9f).xy);
	float out_param_scalar;
	float out_param_vector1;
	testOut(&out_param_scalar);
	{
		float cast;
		testOut(&cast);
		out_param_vector1 = cast;
	}
}
