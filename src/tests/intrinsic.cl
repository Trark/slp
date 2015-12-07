
struct testStruct
{
	float4 value;
};

void test_buffer(__global uint4* g_roBuffer, __global struct testStruct* g_roStructuredBuffer, __global uint4* g_rwBuffer, __global struct testStruct* g_rwStructuredBuffer, uint3 dtid)
{
	uint4 read0 = g_roBuffer[(int)dtid.x];
	uint4 read2 = g_roBuffer[(int)dtid.x];
	uint4 read1 = g_rwBuffer[(int)dtid.x];
	uint4 read3 = g_rwBuffer[(int)dtid.x];
	g_rwBuffer[(int)dtid.x] = read0 + read1 + read2 + read3;
}

void test_structured_buffer(__global uint4* g_roBuffer, __global struct testStruct* g_roStructuredBuffer, __global uint4* g_rwBuffer, __global struct testStruct* g_rwStructuredBuffer, uint3 dtid)
{
	struct testStruct read0 = g_roStructuredBuffer[(int)dtid.x];
	struct testStruct read2 = g_roStructuredBuffer[(int)dtid.x];
	struct testStruct read1 = g_rwStructuredBuffer[(int)dtid.x];
	struct testStruct read3 = g_rwStructuredBuffer[(int)dtid.x];
	struct testStruct modified;
	modified.value = read0.value + read1.value + read2.value + read3.value;
	g_rwStructuredBuffer[(int)dtid.x] = modified;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel(__global uint4* g_roBuffer, __global struct testStruct* g_roStructuredBuffer, __global uint4* g_rwBuffer, __global struct testStruct* g_rwStructuredBuffer)
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	test_buffer(g_roBuffer, g_roStructuredBuffer, g_rwBuffer, g_rwStructuredBuffer, dtid);
	test_structured_buffer(g_roBuffer, g_roStructuredBuffer, g_rwBuffer, g_rwStructuredBuffer, dtid);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	int i = (int)0;
	int i1;
	int2 i2;
	int3 i3;
	int4 i4;
	uint u = 0u;
	uint u1;
	uint2 u2;
	uint3 u3;
	uint4 u4;
	float f = 0.0f;
	float f1;
	float2 f2;
	float3 f3;
	float4 f4;
	int asint_0 = as_int(u);
	int asint_1 = as_int(u1);
	int2 asint_2 = as_int2(u2);
	int3 asint_3 = as_int3(u3);
	int4 asint_4 = as_int4(u4);
	int asint_5 = as_int(f);
	int asint_6 = as_int(f1);
	int2 asint_7 = as_int2(f2);
	int3 asint_8 = as_int3(f3);
	int4 asint_9 = as_int4(f4);
	uint asuint_0 = as_uint(i);
	uint asuint_1 = as_uint(i1);
	uint2 asuint_2 = as_uint2(i2);
	uint3 asuint_3 = as_uint3(i3);
	uint4 asuint_4 = as_uint4(i4);
	uint asuint_5 = as_uint(f);
	uint asuint_6 = as_uint(f1);
	uint2 asuint_7 = as_uint2(f2);
	uint3 asuint_8 = as_uint3(f3);
	uint4 asuint_9 = as_uint4(f4);
	float asfloat_0 = as_float(i);
	float asfloat_1 = as_float(i1);
	float2 asfloat_2 = as_float2(i2);
	float3 asfloat_3 = as_float3(i3);
	float4 asfloat_4 = as_float4(i4);
	float asfloat_5 = as_float(u);
	float asfloat_6 = as_float(u1);
	float2 asfloat_7 = as_float2(u2);
	float3 asfloat_8 = as_float3(u3);
	float4 asfloat_9 = as_float4(u4);
	float asfloat_10 = as_float(f);
	float asfloat_11 = as_float(f1);
	float2 asfloat_12 = as_float2(f2);
	float3 asfloat_13 = as_float3(f3);
	float4 asfloat_14 = as_float4(f4);
	int ix;
	int ix1;
	int2 ix2;
	int3 ix3;
	int4 ix4;
	int iy;
	int iy1;
	int2 iy2;
	int3 iy3;
	int4 iy4;
	int iz;
	int iz1;
	int2 iz2;
	int3 iz3;
	int4 iz4;
	float fx;
	float fx1;
	float2 fx2;
	float3 fx3;
	float4 fx4;
	float fy;
	float fy1;
	float2 fy2;
	float3 fy3;
	float4 fy4;
	float fz;
	float fz1;
	float2 fz2;
	float3 fz3;
	float4 fz4;
	int clamp_0 = clamp(ix, iy, iz);
	int clamp_1 = clamp(ix1, iy1, iz1);
	int2 clamp_2 = clamp(ix2, iy2, iz2);
	int3 clamp_3 = clamp(ix3, iy3, iz3);
	int4 clamp_4 = clamp(ix4, iy4, iz4);
	float clamp_5 = clamp(fx, fy, fz);
	float clamp_6 = clamp(fx1, fy1, fz1);
	float2 clamp_7 = clamp(fx2, fy2, fz2);
	float3 clamp_8 = clamp(fx3, fy3, fz3);
	float4 clamp_9 = clamp(fx4, fy4, fz4);
	float3 cross_0 = cross(fx3, fx3);
	float distance_1 = length(fx1 - fy1);
	float distance_2 = length(fx2 - fy2);
	float distance_3 = length(fx3 - fy3);
	float distance_4 = length(fx4 - fy4);
	float dot_f1 = dot(fx1, fy1);
	float dot_f2 = dot(fx2, fy2);
	float dot_f3 = dot(fx3, fy3);
	float dot_f4 = dot(fx4, fy4);
}
