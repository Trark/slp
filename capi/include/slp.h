
#include <stdint.h>

extern "C" {

struct slp_hlsl_to_cl_input {
    char const* entry_point;
    char const* main_file;
    char const *main_file_name;
    char const* (*include_handler)(void*, char const*);
    void* user_data;
    char const* kernel_name;
};

struct slp_hlsl_cl_bind {
    uint16_t hlsl_slot;
    uint16_t cl_slot;
};

struct slp_hlsl_cl_bind_table {
    slp_hlsl_cl_bind const* binds;
    uint16_t count;
};

struct slp_hlsl_to_cl_output {
    char const* error;
    char const* source;
    char const* kernel_name;
    slp_hlsl_cl_bind_table constant_buffers;
    slp_hlsl_cl_bind_table shader_resource_views;
    slp_hlsl_cl_bind_table sampler_states;
    slp_hlsl_cl_bind_table unordered_access_views;
    uint64_t dim_x;
    uint64_t dim_y;
    uint64_t dim_z;
};

slp_hlsl_to_cl_output slp_hlsl_to_cl(slp_hlsl_to_cl_input input);
void slp_hlsl_to_cl_free(slp_hlsl_to_cl_output input);

}
