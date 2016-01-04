
void vector_int_2()
{
	int2 v;
	v.x = (int)0;
	v.y = (int)1;
	int i_x = v.x;
	int i_y = v.y;
	int i1_x = v.x;
	int i1_y = v.y;
	int2 i2_xx = v.xx;
	int2 i2_xy = v.yx;
	int2 i2_yx = v.xy;
	int2 i2_yy = v.yy;
	int3 i3_xxx = v.xxx;
	int3 i3_xxy = v.xxy;
	int3 i3_xyx = v.xyx;
	int3 i3_xyy = v.xyy;
	int3 i3_yxx = v.yxx;
	int3 i3_yxy = v.yxy;
	int3 i3_yyx = v.yyx;
	int3 i3_yyy = v.yyy;
	int4 i3_xxxx = v.xxxx;
	int4 i3_xxxy = v.xxxy;
	int4 i3_xxyx = v.xxyx;
	int4 i3_xxyy = v.xxyy;
	int4 i3_xyxx = v.xyxx;
	int4 i3_xyxy = v.xyxy;
	int4 i3_xyyx = v.xyyx;
	int4 i3_xyyy = v.xyyy;
	int4 i3_yxxx = v.yxxx;
	int4 i3_yxxy = v.yxxy;
	int4 i3_yxyx = v.yxyx;
	int4 i3_yxyy = v.yxyy;
	int4 i3_yyxx = v.yyxx;
	int4 i3_yyxy = v.yyxy;
	int4 i3_yyyx = v.yyyx;
	int4 i3_yyyy = v.yyyy;
}

void test_call_f0(float3 x)
{
}

void test_call_f1(__private float3* y)
{
	*y = (float3)(1.0f, 2.0f, 3.0f);
}

void test_call_f2(float x, __private float3* y, __private float* z, __private float2* u, float v)
{
	*y = (float3)(x, x, x);
	*z = v;
	*u = (float2)(x, v);
}

void test_call_f1_shim(__private float4* p)
{
	float3 v = (*p).xyz;
	test_call_f1(&v);
	(*p).xyz = v;
}

void test_call_f2_shim(float p, __private float4* p_0, __private float* p_1, __private float3* p_2, float p_3)
{
	float3 v = (*p_0).xyz;
	float2 v_0 = (*p_2).yz;
	test_call_f2(p, &v, p_1, &v_0, p_3);
	(*p_0).xyz = v;
	(*p_2).yz = v_0;
}

void test_call()
{
	float4 f4;
	test_call_f0(f4.xyz);
	test_call_f1_shim(&f4);
	float output_z;
	float3 output_u;
	test_call_f2_shim(3.0f, &f4, &output_z, &output_u, 3.0f);
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	vector_int_2();
	test_call();
}
