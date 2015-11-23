
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

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	testIntOrUInt_0((int)0);
	testIntOrUInt_1((uint)0);
	testIntOrFloat_0((int)0);
	testIntOrFloat_1(0.0f);
	testIntOrFloat_0(0);
	testIntOrFloat_0(0u);
	testBoolOrFloat_0((bool)0);
	testBoolOrFloat_1(0.0f);
	testBoolOrFloat_0(0);
}
