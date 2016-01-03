
[numthreads(8, 8, 1)]
void CSMAIN()
{
    bool b_b = true ? true : false;
    float f_f = true ? 4.0 : 3.0;
    float f_u = true ? 4.0 : 3u;
    float f_i = true ? 4.0 : (int)3;
    uint i_u = true ? (int)4 : 3u;
    uint u_i = true ? 2u : (int)2;
    int i_i = true ? (int)4 : (int)2;
    uint4 u_u4 = true ? 4u : uint4(1u, 2u, 3u, 4u);
    int4 i_i4 = true ? 4 : int4(1, 2, 3, 4);
    float4 f_i4 = true ? 0.5 : int4(1, 2, 3, 4);
}
