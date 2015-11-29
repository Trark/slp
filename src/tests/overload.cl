
struct __globals
{
};

void testIntOrUInt_0(__private struct __globals* globals, int x)
{
}

void testIntOrUInt_1(__private struct __globals* globals, uint x)
{
}

void testIntOrFloat_0(__private struct __globals* globals, int x)
{
}

void testIntOrFloat_1(__private struct __globals* globals, float x)
{
}

void testBoolOrFloat_0(__private struct __globals* globals, bool x)
{
}

void testBoolOrFloat_1(__private struct __globals* globals, float x)
{
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	struct __globals __init;
	__private struct __globals* globals = &__init;
	testIntOrUInt_0(globals, (int)0);
	testIntOrUInt_1(globals, (uint)0);
	testIntOrFloat_0(globals, (int)0);
	testIntOrFloat_1(globals, 0.0f);
	testIntOrFloat_0(globals, (int)0);
	testIntOrFloat_0(globals, (int)0u);
	testBoolOrFloat_0(globals, (bool)0);
	testBoolOrFloat_1(globals, 0.0f);
	testBoolOrFloat_0(globals, (bool)0);
}
