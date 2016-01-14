
void test(uint x)
{
	x = (uint)1;
	x = (uint)1;
	x = 1u + 2u;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void Main()
{
	uint3 dtid = (uint3)(get_global_id(0u), get_global_id(1u), get_global_id(2u));
	test(dtid.x);
}
