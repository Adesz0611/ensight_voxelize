#define CFD_LIB_IMPLEMENTATION
#include "ensight_gold.h"
#include "cfd_render.h"
#include "cfd_slicer.h"
#include "vtk.h"

#include "lz4.h"


#ifdef _OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <string.h>

#define cstr_equ(lhs, rhs) (strcmp(lhs, rhs) == 0)

u32 parse_resolution_or_exit(char *str);
f32 parse_bbox_or_exit(char *str);
V3 *generate_voxels(CFD_Arena *arena, u32 res_x, u32 res_y, u32 res_z, V3 min_aabb, V3 max_aabb);
b32 export_vol_file(CFD_Arena *arena, char *filename, u32 res_x, u32 res_y, u32 res_z, V3 aabb_min, V3 aabb_max, u8 dim, f32 *values);

b32 export_vol_file(CFD_Arena *arena, char *filename, u32 res_x, u32 res_y, u32 res_z, V3 aabb_min, V3 aabb_max, u8 dim, f32 *values) {
    u64 vert_num = (u64)res_x * (u64)res_y * (u64)res_z;

    // dimension reduction (if dim == 3: values[i] <- magnitude of values[i])
    if (dim == 3) {
        for (u64 i = 0; i < vert_num; ++i) {
            f32 x = values[i * 3 + 0];
            f32 y = values[i * 3 + 1];
            f32 z = values[i * 3 + 2];

            values[i] = sqrtf(x*x+y*y+z*z);
        }
    }

    f32 min_range =  FLT_MAX;
    f32 max_range = -FLT_MAX;

    for (u64 i = 0; i < vert_num; ++i) {
        f32 val = values[i];
        if (val > max_range) max_range = val;
        if (val < min_range) min_range = val;
    }

    u8 *values_u8 = (u8 *)values;
    for (u64 i = 0; i < vert_num; ++i) {
        f32 val = values[i];
        values_u8[i] = (u8)(((val - min_range) / (max_range - min_range)) * 255.f);
    }

    char *input = (char *)values_u8;
    u64 input_size = vert_num * sizeof(u8);
    int max_compressed_size = LZ4_compressBound((int)input_size);
    char *compressed = cfd_arena_push_array(arena, char, (u32)max_compressed_size);

    int _compressed_size = LZ4_compress_default(
        input,
        compressed,
        (int)input_size,
        max_compressed_size
    );

    if (_compressed_size <= 0) {
        cfd_error("Compresson failed!");
        return false;
    }

    u64 compressed_size = (u64)_compressed_size;
    cfd_info("original size: %zu", input_size);
    cfd_info("compressed_size: %d", compressed_size);

    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
#ifdef _MSC_VER
        char error_str[256];
        strerror_s(error_str, 256, errno);
        cfd_error("couldn't open '%s' file for writing: %s", filename, error_str);
#else
        cfd_error("couldn't open '%s' file for writing: %s", filename, strerror(errno));
#endif
        return false;
    }

    u32 flags = 0;

    f32 min_bounds[3] = {
      aabb_min.x, aabb_min.y, aabb_min.z,
    };

    f32 max_bounds[3] = {
      aabb_max.x, aabb_max.y, aabb_max.z,
    };

    u32 magic_number = 0;
    u32 format_type  = 0;
    fwrite(&magic_number, sizeof(u32), 1, f);
    fwrite(&format_type, sizeof(u32), 1, f);
    fwrite(&compressed_size, sizeof(u64), 1, f);
    fwrite(&input_size, sizeof(u64), 1, f);
    fwrite(&flags, sizeof(u32), 1, f);
    fwrite(&res_y, sizeof(u32), 1, f);
    fwrite(&res_x, sizeof(u32), 1, f);
    fwrite(&res_z, sizeof(u32), 1, f);
    fwrite(&min_range, sizeof(f32), 1, f);
    fwrite(&max_range, sizeof(f32), 1, f);
    fwrite(min_bounds, sizeof(f32) * 3, 1, f);
    fwrite(max_bounds, sizeof(f32) * 3, 1, f);
    fwrite(compressed, sizeof(char), compressed_size, f);

    fclose(f);
    return true;
}

u32 parse_resolution_or_exit(char *str) {
    Str8 s;

    s.buffer = (u8 *)str;
    s.len = (u64)strlen(str);

    if (!str8_is_u32(s)) {
        fprintf(stderr, "ERROR while parsing resolution argument, %s is not a number!\n", str);
        exit(1);
    }

    return str8_to_u32(s);
}

f32 parse_bbox_or_exit(char *str) {
    Str8 s;

    s.buffer = (u8 *)str;
    s.len = (u64)strlen(str);

    if (!str8_is_f32(s)) {
        fprintf(stderr, "ERROR while parsing bounding box argument, %s is not a number!\n", str);
        exit(1);
    }

    return str8_to_f32(s);
}

V3 *generate_voxels(CFD_Arena *arena, u32 res_x, u32 res_y, u32 res_z, V3 min_aabb, V3 max_aabb) {
    u32 obj_vert_num = res_x * res_y * res_z;

    f32 x_step = (max_aabb.x - min_aabb.x) / (f32)res_x;
    f32 y_step = (max_aabb.y - min_aabb.y) / (f32)res_y;
    f32 z_step = (max_aabb.z - min_aabb.z) / (f32)res_z;

    f32 x_step_half = x_step / 2.0f;
    f32 y_step_half = y_step / 2.0f;
    f32 z_step_half = z_step / 2.0f;

    V3 *vertices = (V3 *)cfd_arena_push_array(arena, V3, obj_vert_num);
    u64 i = 0;

    for (u32 z = 0; z < res_z; ++z) {
        for (u32 y = 0; y < res_y; ++y) {
            for (u32 x = 0; x < res_x; ++x) {
                V3 v;

                v.x = min_aabb.x + (f32)x * x_step + x_step_half;
                v.y = min_aabb.y + (f32)y * y_step + y_step_half;
                v.z = min_aabb.z + (f32)z * z_step + z_step_half;

                vertices[i++] = v;
            }
        }
    }

    return vertices;
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <ensight filename> [options]\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "    --res <x_resolution> <y_resolution> <z_resolution>\n");
        fprintf(stderr, "    --bbox <x_min> <y_min> <z_min> <x_max> <y_max> <z_max>\n");
        return -1;
    }
    char *ensight_filename = *(++argv);
    char *option = *(++argv);
    argc -= 3;


    b32 res_specified = false;
    b32 bbox_specified = false;

    u32 res_x, res_y, res_z;
    V3 specified_min_aabb;
    V3 specified_max_aabb;

    // DUMMY but works for now
    while (true) {
        if (option == NULL) break;

        if (cstr_equ(option, "--res")) {
            if (argc < 3) {
                fprintf(stderr, "ERROR: --res expects 3 arguments!\n");
                return -1;
            }

            res_x = parse_resolution_or_exit(*(++argv));
            res_y = parse_resolution_or_exit(*(++argv));
            res_z = parse_resolution_or_exit(*(++argv));

            printf("resolution_x = %d\n", res_x);
            printf("resolution_y = %d\n", res_y);
            printf("resolution_z = %d\n", res_z);

            argc -= 3;
            res_specified = true;

        } else if (cstr_equ(option, "--bbox")) {
            if (argc < 6) {
                fprintf(stderr, "ERROR: --bbox expects 6 arguments!\n");
                return -1;
            }

            specified_min_aabb.x = parse_bbox_or_exit(*(++argv));
            specified_min_aabb.y = parse_bbox_or_exit(*(++argv));
            specified_min_aabb.z = parse_bbox_or_exit(*(++argv));

            specified_max_aabb.x = parse_bbox_or_exit(*(++argv));
            specified_max_aabb.y = parse_bbox_or_exit(*(++argv));
            specified_max_aabb.z = parse_bbox_or_exit(*(++argv));

            printf("AABB min (x=%f, y=%f, z=%f)\n", (double)specified_min_aabb.x, (double)specified_min_aabb.y, (double)specified_min_aabb.z);
            printf("AABB max (x=%f, y=%f, z=%f)\n", (double)specified_max_aabb.x, (double)specified_max_aabb.y, (double)specified_max_aabb.z);

            argc -= 6;
            bbox_specified = true;

        } else {
            return -1;
        }

        option = *(++argv);
        --argc;
    }

    if (!res_specified) {
        fprintf(stderr, "ERROR: Resolution must be specified!\n");
        return -1;
    }


    CFD_Arena arena, scratch_arena;
    if (unlikely(!cfd_arena_init(&arena, GB(16)))) return 1;
    if (unlikely(!cfd_arena_init(&scratch_arena, GB(16)))) return 1;

    char *case_filename = ensight_filename;

    Ensight_Case encase;
    if (!ensight_read_case(&arena, &encase, case_filename)) return 1;

    CFD_UnstructuredGrid mesh;
    CFD_BVH8Tree bvhtree;
    V3 *vertices = NULL;
    u64 vertex_count;
    if (encase.geometry->model->ts == -1) {
        ensight_read_model_merge_parts(&arena, &scratch_arena, &encase, 0, &mesh);
        bvhtree = cfd_build_grid_bvh8(&arena, &scratch_arena, &mesh);
        if (!bbox_specified) {
            specified_min_aabb = mesh.aabb_min;
            specified_max_aabb = mesh.aabb_max;
        }
        printf("AABB min (x=%f, y=%f, z=%f)\n", (double)specified_min_aabb.x, (double)specified_min_aabb.y, (double)specified_min_aabb.z);
        printf("AABB max (x=%f, y=%f, z=%f)\n", (double)specified_max_aabb.x, (double)specified_max_aabb.y, (double)specified_max_aabb.z);
        vertices = generate_voxels(&arena, res_x, res_y, res_z, specified_min_aabb, specified_max_aabb);
        vertex_count = res_x * res_y * res_z;
    } else {
        fprintf(stderr, "ERROR: this program is only working with static geometry!\n");
        return 1;
    }

    CFD_Arena variable_arena;
    if (unlikely(!cfd_arena_init(&variable_arena, GB(16)))) return 1;

    for (u32 var_idx = 0; var_idx < encase.variable->len; ++var_idx) {
        Ensight_Variable *var = &encase.variable->elems[var_idx];

        if (var->ts == -1) {
            continue;
        }

        s32 time_set_number = ensight_get_time_set_index(&encase, var->ts);
        if (time_set_number == -1) return 1;

        u8 var_dim = (var->type == ENSIGHT_VARIABLE_SCALAR_PER_NODE || var->type == ENSIGHT_VARIABLE_SCALAR_PER_ELEMENT) ? 1 : 3;
        Ensight_Time *time = &encase.time->elems[time_set_number];
        for (u32 time_idx = 0; time_idx < time->number_of_steps; ++time_idx) {
            f32 *var_data = ensight_read_variable(&variable_arena, &encase, var_idx, time_idx);
            f32 *node_data = NULL;
            if (var->type == ENSIGHT_VARIABLE_SCALAR_PER_NODE ||
                var->type == ENSIGHT_VARIABLE_VECTOR_PER_NODE) {
                node_data = var_data;
            } else {
                node_data = cfd_cell_data_to_node_data(&variable_arena, &scratch_arena, &mesh, var_data, var_dim);
            }

            f32 *values = cfd_arena_push_array(&variable_arena, f32, vertex_count * var_dim);

            cfd_sample_points_tetra_node_data_parallel(
                &bvhtree,
                &mesh,
                node_data,
                var_dim,
                vertices,
                vertex_count,
                1e-6f,
                values
            );

            // VTK export if you need for testing
            /*
            char vtk_fn[1024];
            snprintf(vtk_fn, 1024, "%.*s_%zu.vtk", str8_arg(var->description), (u64)time->time_values[time_idx]);
            cfd_points_to_vtk(vertices, vertex_count, values, var_dim, vtk_fn);
            */

            char filename[1024];
            snprintf(filename, 1024, "%.*s_%zu.vol", str8_arg(var->description), (u64)time->time_values[time_idx]);
            export_vol_file(&variable_arena, filename, res_x, res_y, res_z, specified_min_aabb, specified_max_aabb, var_dim, values);

            cfd_arena_reset(&variable_arena);
        }
    }

    cfd_arena_log_usage("scratch", &scratch_arena);
    cfd_arena_log_usage("persistant", &arena);

    cfd_arena_destroy(&arena);
    cfd_arena_destroy(&scratch_arena);
    return 0;
}
