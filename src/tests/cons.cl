
float2 cast_uint2_to_float2(uint2 from)
{
	float2 to;
	to[0] = from[0];
	to[1] = from[1];
	return to;
}

float3 cast_uint3_to_float3(uint3 from)
{
	float3 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	return to;
}

float4 cast_uint4_to_float4(uint4 from)
{
	float4 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	to[3] = from[3];
	return to;
}

void cons_float4()
{
	float4 target;
	float s;
	float s1;
	float2 s2;
	float3 s3;
	float4 s4;
	target = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
	target = (float4)(s1, s1, s1, s1);
	target = (float4)(s2, s2);
	target = (float4)(s, s2, s);
	target = (float4)(s1, s2, s1);
	target = (float4)(s3, s);
	target = (float4)(s3, s1);
	target = (float4)(s, s3);
	target = (float4)(s1, s3);
	target = (float4)(s4);
	uint u;
	uint u1;
	uint2 u2;
	uint3 u3;
	uint4 u4;
	target = (float4)((float)0u, (float)0, (float)(int)0, 1.0f);
	target = (float4)((float)u1, (float)u1, (float)u1, (float)u1);
	target = (float4)(cast_uint2_to_float2(u2), cast_uint2_to_float2(u2));
	target = (float4)((float)u, cast_uint2_to_float2(u2), (float)u);
	target = (float4)((float)u1, cast_uint2_to_float2(u2), (float)u1);
	target = (float4)(cast_uint3_to_float3(u3), (float)u);
	target = (float4)(cast_uint3_to_float3(u3), (float)u1);
	target = (float4)((float)u, cast_uint3_to_float3(u3));
	target = (float4)((float)u1, cast_uint3_to_float3(u3));
	target = (float4)(cast_uint4_to_float4(u4));
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	cons_float4();
}
