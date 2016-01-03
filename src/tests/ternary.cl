
float4 cast_int4_to_float4(int4 from)
{
	float4 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	to[3] = from[3];
	return to;
}

int4 cast_int_to_int4(int from)
{
	int4 to;
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

float4 cast_float_to_float4(float from)
{
	float4 to;
	to[0] = from;
	to[1] = from;
	to[2] = from;
	to[3] = from;
	return to;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	bool b_b = true ? true : false;
	float f_f = true ? 4.0f : 3.0f;
	float f_u = true ? 4.0f : (float)3u;
	float f_i = true ? 4.0f : (float)(int)3;
	uint i_u = true ? (uint)(int)4 : 3u;
	uint u_i = true ? 2u : (uint)(int)2;
	int i_i = true ? (int)4 : (int)2;
	uint4 u_u4 = true ? cast_uint_to_uint4(4u) : (uint4)(1u, 2u, 3u, 4u);
	int4 i_i4 = true ? cast_int_to_int4(4) : (int4)((int)1, (int)2, (int)3, (int)4);
	float4 f_i4 = true ? cast_float_to_float4(0.5f) : cast_int4_to_float4((int4)((int)1, (int)2, (int)3, (int)4));
}
