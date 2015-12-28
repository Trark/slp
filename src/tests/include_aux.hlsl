
#define DEF_ONE 1
#define DEF_ONE_COPY \
DEF_ONE
#define TEST(x, y) (x + \
	y)

void test(uint x)
{
    x = DEF_ONE;
    x = DEF_ONE_COPY;
    x = TEST(1u,
    2u);
}
