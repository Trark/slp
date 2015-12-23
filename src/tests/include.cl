
void test(uint x)
{
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void MyKernel()
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	test(dtid.x);
}
