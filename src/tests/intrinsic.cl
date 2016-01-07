#pragma OPENCL EXTENSION cl_khr_fp16 : enable

int2 cast_uint2_to_int2(uint2 from)
{
	int2 to;
	to[0] = from[0];
	to[1] = from[1];
	return to;
}

int3 cast_uint3_to_int3(uint3 from)
{
	int3 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	return to;
}

int4 cast_uint4_to_int4(uint4 from)
{
	int4 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	to[3] = from[3];
	return to;
}

int2 cast_float2_to_int2(float2 from)
{
	int2 to;
	to[0] = from[0];
	to[1] = from[1];
	return to;
}

int3 cast_float3_to_int3(float3 from)
{
	int3 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	return to;
}

int4 cast_float4_to_int4(float4 from)
{
	int4 to;
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
	to[3] = from[3];
	return to;
}

struct testStruct
{
	float4 value;
};

void test_buffer(__global uint4* g_roBuffer, __global uint4* g_rwBuffer, uint3 dtid)
{
	uint4 read0 = g_roBuffer[(int)dtid.x];
	uint4 read2 = g_roBuffer[(int)dtid.x];
	uint4 read1 = g_rwBuffer[(int)dtid.x];
	uint4 read3 = g_rwBuffer[(int)dtid.x];
	g_rwBuffer[(int)dtid.x] = read0 + read1 + read2 + read3;
}

void test_structured_buffer(__global struct testStruct* g_roStructuredBuffer, __global struct testStruct* g_rwStructuredBuffer, uint3 dtid)
{
	struct testStruct read0 = g_roStructuredBuffer[(int)dtid.x];
	struct testStruct read2 = g_roStructuredBuffer[(int)dtid.x];
	struct testStruct read1 = g_rwStructuredBuffer[(int)dtid.x];
	struct testStruct read3 = g_rwStructuredBuffer[(int)dtid.x];
	struct testStruct modified;
	modified.value = read0.value + read1.value + read2.value + read3.value;
	g_rwStructuredBuffer[(int)dtid.x] = modified;
}

void test_texture_2d(read_only image2d_t g_rwRTexture2DFloat, read_only image2d_t g_rwRTexture2DInt, read_only image2d_t g_rwRTexture2DUInt, uint3 dtid)
{
	int2 coord;
	coord.x = (int)dtid.x;
	coord.y = (int)dtid.y;
	float4 read_load_f = read_imagef(g_rwRTexture2DFloat, coord);
	int4 read_load_i = read_imagei(g_rwRTexture2DInt, coord);
	uint4 read_load_ui = read_imageui(g_rwRTexture2DUInt, coord);
	write_imagef(g_rwRTexture2DFloat, coord, read_load_f);
	write_imagei(g_rwRTexture2DInt, coord, read_load_i);
	write_imageui(g_rwRTexture2DUInt, coord, read_load_ui);
}

void test_byte_address_buffer(__global uchar* g_roRawBuffer, __global uchar* g_rwRawBuffer, uint3 dtid)
{
	uint ro1 = *(__global uint*)(g_roRawBuffer + 64u * dtid.x);
	uint2 ro2 = *(__global uint2*)(g_roRawBuffer + (64u * dtid.x + 16u));
	uint3 ro3 = *(__global uint3*)(g_roRawBuffer + (64u * dtid.x + 32u));
	uint4 ro4 = *(__global uint4*)(g_roRawBuffer + (64u * dtid.x + 48u));
	uint rw1 = *(__global uint*)(g_rwRawBuffer + 64u * dtid.x);
	uint2 rw2 = *(__global uint2*)(g_rwRawBuffer + (64u * dtid.x + 16u));
	uint3 rw3 = *(__global uint3*)(g_rwRawBuffer + (64u * dtid.x + 32u));
	uint4 rw4 = *(__global uint4*)(g_rwRawBuffer + (64u * dtid.x + 48u));
	*(__global uint*)(g_rwRawBuffer + 64u * dtid.x) = ro1 + rw1;
	*(__global uint2*)(g_rwRawBuffer + (64u * dtid.x + 16u)) = ro2 + rw2;
	*(__global uint3*)(g_rwRawBuffer + (64u * dtid.x + 32u)) = ro3 + rw3;
	*(__global uint4*)(g_rwRawBuffer + (64u * dtid.x + 48u)) = ro4 + rw4;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel(__global uint4* g_roBuffer, __global struct testStruct* g_roStructuredBuffer, __global uchar* g_roRawBuffer, __global uint4* g_rwBuffer, __global struct testStruct* g_rwStructuredBuffer, read_only image2d_t g_rwRTexture2DFloat, read_only image2d_t g_rwRTexture2DInt, read_only image2d_t g_rwRTexture2DUInt, __global uchar* g_rwRawBuffer)
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	test_buffer(g_roBuffer, g_rwBuffer, dtid);
	test_structured_buffer(g_roStructuredBuffer, g_rwStructuredBuffer, dtid);
	test_texture_2d(g_rwRTexture2DFloat, g_rwRTexture2DInt, g_rwRTexture2DUInt, dtid);
	test_byte_address_buffer(g_roRawBuffer, g_rwRawBuffer, dtid);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	int i = (int)4;
	int j;
	j = i++;
	j = ++i;
	j = i--;
	j = --i;
	j = +i;
	j = -i;
	j = ~i;
	j += i;
	j -= i;
	j *= i;
	j /= i;
	j %= i;
	int i1;
	int2 i2;
	int3 i3;
	int4 i4;
	uint u = 0u;
	uint u1;
	uint2 u2;
	uint3 u3;
	uint4 u4;
	float f = 3.0f;
	j += (int)f;
	j -= (int)f;
	j *= (int)f;
	j /= (int)f;
	j %= (int)f;
	float f1;
	float2 f2;
	float3 f3;
	float4 f4;
	bool b = !i;
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
	bool isnan_s = (bool)isnan(fx);
	float length_f1 = length(fx1);
	float length_f2 = length(fx2);
	float length_f3 = length(fx3);
	float length_f4 = length(fx4);
	float normalize_f1 = normalize(fx1);
	float2 normalize_f2 = normalize(fx2);
	float3 normalize_f3 = normalize(fx3);
	float4 normalize_f4 = normalize(fx4);
	int sign_s = (int)sign(fx);
	int sign_f1 = (int)sign(fx1);
	int2 sign_f2 = cast_float2_to_int2(sign(fx2));
	int3 sign_f3 = cast_float3_to_int3(sign(fx3));
	int4 sign_f4 = cast_float4_to_int4(sign(fx4));
	float sqrt_s = sqrt(fx);
	float sqrt_f1 = sqrt(fx1);
	float2 sqrt_f2 = sqrt(fx2);
	float3 sqrt_f3 = sqrt(fx3);
	float4 sqrt_f4 = sqrt(fx4);
	float min_fs = fmin(fx, fy);
	float min_f1 = fmin(fx1, fy1);
	float2 min_f2 = fmin(fx2, fy2);
	float3 min_f3 = fmin(fx3, fy3);
	float4 min_f4 = fmin(fx4, fy4);
	int min_is = min(ix, iy);
	int min_i1 = min(ix1, iy1);
	int2 min_i2 = min(ix2, iy2);
	int3 min_i3 = min(ix3, iy3);
	int4 min_i4 = min(ix4, iy4);
	float max_fs = fmax(fx, fy);
	float max_f1 = fmax(fx1, fy1);
	float2 max_f2 = fmax(fx2, fy2);
	float3 max_f3 = fmax(fx3, fy3);
	float4 max_f4 = fmax(fx4, fy4);
	int max_is = max(ix, iy);
	int max_i1 = max(ix1, iy1);
	int2 max_i2 = max(ix2, iy2);
	int3 max_i3 = max(ix3, iy3);
	int4 max_i4 = max(ix4, iy4);
	uint packed_half = (uint)as_ushort((half)fx);
	float unpacked_half = (float)as_half((ushort)packed_half);
	float step_fs = step(fx, fy);
	float step_f1 = step(fx1, fy1);
	float2 step_f2 = step(fx2, fy2);
	float3 step_f3 = step(fx3, fy3);
	float4 step_f4 = step(fx4, fy4);
	float exp_fs = exp(fx);
	float exp_f1 = exp(fx1);
	float2 exp_f2 = exp(fx2);
	float3 exp_f3 = exp(fx3);
	float4 exp_f4 = exp(fx4);
	int abs_is = (int)abs(ix);
	int abs_i1 = (int)abs(ix1);
	int2 abs_i2 = cast_uint2_to_int2(abs(ix2));
	int3 abs_i3 = cast_uint3_to_int3(abs(ix3));
	int4 abs_i4 = cast_uint4_to_int4(abs(ix4));
	float abs_fs = fabs(fx);
	float abs_f1 = fabs(fx1);
	float2 abs_f2 = fabs(fx2);
	float3 abs_f3 = fabs(fx3);
	float4 abs_f4 = fabs(fx4);
	(void)(fy = sincos(fx, &fz));
	(void)(fy1 = sincos(fx1, &fz1));
	(void)(fy2 = sincos(fx2, &fz2));
	(void)(fy3 = sincos(fx3, &fz3));
	(void)(fy4 = sincos(fx4, &fz4));
	float sin_fs = sin(fx);
	float sin_f1 = sin(fx1);
	float2 sin_f2 = sin(fx2);
	float3 sin_f3 = sin(fx3);
	float4 sin_f4 = sin(fx4);
	float cos_fs = cos(fx);
	float cos_f1 = cos(fx1);
	float2 cos_f2 = cos(fx2);
	float3 cos_f3 = cos(fx3);
	float4 cos_f4 = cos(fx4);
	float pow_fs = pow(fx, fy);
	float pow_f1 = pow(fx1, fy1);
	float2 pow_f2 = pow(fx2, fy2);
	float3 pow_f3 = pow(fx3, fy3);
	float4 pow_f4 = pow(fx4, fy4);
	float smoothstep_fs = smoothstep(fx, fy, fz);
	float smoothstep_f1 = smoothstep(fx1, fy1, fz1);
	float2 smoothstep_f2 = smoothstep(fx2, fy2, fz2);
	float3 smoothstep_f3 = smoothstep(fx3, fy3, fz3);
	float4 smoothstep_f4 = smoothstep(fx4, fy4, fz4);
	float saturate_fs = clamp(fx, 0.0f, 1.0f);
	float saturate_f1 = clamp(fx1, 0.0f, 1.0f);
	float2 saturate_f2 = clamp(fx2, 0.0f, 1.0f);
	float3 saturate_f3 = clamp(fx3, 0.0f, 1.0f);
	float4 saturate_f4 = clamp(fx4, 0.0f, 1.0f);
	float lerp_fs = mix(fx, fy, fz);
	float lerp_f1 = mix(fx1, fy1, fz1);
	float2 lerp_f2 = mix(fx2, fy2, fz2);
	float3 lerp_f3 = mix(fx3, fy3, fz3);
	float4 lerp_f4 = mix(fx4, fy4, fz4);
	float acos_fs = acos(fx);
	float acos_f1 = acos(fx1);
	float2 acos_f2 = acos(fx2);
	float3 acos_f3 = acos(fx3);
	float4 acos_f4 = acos(fx4);
	float asin_fs = asin(fx);
	float asin_f1 = asin(fx1);
	float2 asin_f2 = asin(fx2);
	float3 asin_f3 = asin(fx3);
	float4 asin_f4 = asin(fx4);
}
