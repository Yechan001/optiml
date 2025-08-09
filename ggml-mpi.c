#include "ggml-mpi.h"

#include "ggml.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

struct ggml_mpi_context {
    int rank;
    int size;
};

void ggml_mpi_backend_init(void) {
    MPI_Init(NULL, NULL);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    struct ggml_mpi_context * ctx = calloc(1, sizeof(struct ggml_mpi_context));

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);

    return ctx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

void ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads) {
    UNUSED(ctx_mpi);

    // synchronize the worker node parameters with the root node
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(n_tokens,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_past,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
}


static void pm_broadcast_mode(pm_mode * mode, int root_rank) {
    int imode = (int) *mode;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&imode, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    *mode = (pm_mode) imode;
}

static void pm_reduce_telemetry(pm_telemetry * t_local, pm_telemetry * t_global) {
    // Average across ranks for a cluster-wide picture
    float vals[4]  = { t_local->cpu_util, t_local->gpu_util, t_local->pkg_power_w, t_local->temp_c };
    float gvals[4] = { 0, 0, 0, 0 };
    MPI_Allreduce(vals, gvals, 4, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    t_global->cpu_util    = gvals[0] / world;
    t_global->gpu_util    = gvals[1] / world;
    t_global->pkg_power_w = gvals[2] / world;
    t_global->temp_c      = gvals[3] / world;
}

static void pm_scatter_thread_caps(pm_context * ctx) {
    // Example: root rank distributes a descending thread cap to balance thermals
    int caps[256] = {0};
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    PM_ASSERT(world <= 256);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int base = (ctx->mode == PM_MODE_PERFORMANCE) ? -1 :
                   (ctx->mode == PM_MODE_BALANCED)    ? 16 : 8;
        for (int i = 0; i < world; ++i) {
            if (base < 0) { caps[i] = -1; continue; }
            // Slight tapering by rank
            caps[i] = std::max(4, base - (i % 4) * 2);
        }
    }

    int my_cap = 0;
    MPI_Scatter(caps, 1, MPI_INT, &my_cap, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_cap != 0) {
        ctx->limits.cpu_threads_allowed = my_cap;
    }
}

// --- public-ish API ---

void pm_init(pm_context * ctx, pm_mode mode, int verbose) {
    PM_ASSERT(ctx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->mode    = mode;
    ctx->verbose = verbose;

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->mpi_world);

    // synchronize world and mode
    pm_broadcast_mode(&ctx->mode, /*root=*/0);
    ctx->limits = pm_policy_from_mode(ctx->mode);
    pm_apply_limits(&ctx->limits);

    if (ctx->verbose && ctx->mpi_rank == 0) {
        fprintf(stderr, "pm_init: world=%d, mode=%d\n", ctx->mpi_world, (int) ctx->mode);
    }
}

void pm_set_mode(pm_context * ctx, pm_mode mode) {
    PM_ASSERT(ctx);
    ctx->mode   = mode;
    ctx->limits = pm_policy_from_mode(mode);
    pm_apply_limits(&ctx->limits);

    if (ctx->verbose) {
        fprintf(stderr, "pm_set_mode(rank=%d): mode=%d\n", ctx->mpi_rank, (int) mode);
    }
}

void pm_tick(pm_context * ctx, int interval_ms) {
    PM_ASSERT(ctx);

    // 1) Collect local telemetry
    pm_telemetry local = pm_os_query_telemetry();
    ctx->last = local;

    // 2) Combine to global view
    pm_telemetry global = {};
    pm_reduce_telemetry(&local, &global);

    // 3) Root adapts high-level caps and broadcasts implicit policy via scatter
    if (ctx->mpi_rank == 0) {
        if (ctx->mode == PM_MODE_BALANCED) {
            // If cluster is running hot, nudge toward efficiency envelope
            if (global.temp_c > 88.0f || global.pkg_power_w > 180.0f) {
                // Implicitly reduce thread caps on next scatter
                if (ctx->verbose) {
                    fprintf(stderr, "pm_tick(root): global hot -> tapering thread caps\n");
                }
            }
        }
    }

    pm_scatter_thread_caps(ctx);

    // 4) Local closed-loop trim
    pm_adaptive_trim(ctx);
    pm_apply_limits(&ctx->limits);

    if (ctx->verbose && (ctx->mpi_rank == 0 || ctx->mpi_rank == ctx->mpi_world - 1)) {
        fprintf(stderr,
                "pm_tick(rank=%d): mode=%d cpu_util=%.2f gpu_util=%.2f power=%.1fW temp=%.1fC | cpu=%dKHz threads=%d gpuW=%d\n",
                ctx->mpi_rank, (int) ctx->mode,
                ctx->last.cpu_util, ctx->last.gpu_util, ctx->last.pkg_power_w, ctx->last.temp_c,
                ctx->limits.cpu_freq_khz_target, ctx->limits.cpu_threads_allowed, ctx->limits.gpu_power_limit_w);
    }

    SLEEP_MS(interval_ms);
}

// Example of a blocking control loop that runs N iterations
void pm_run(pm_context * ctx, int iterations, int interval_ms) {
    PM_ASSERT(ctx);
    for (int i = 0; i < iterations; ++i) {
        // synchronize pacing across ranks to stabilize telemetry windows
        MPI_Barrier(MPI_COMM_WORLD);
        pm_tick(ctx, interval_ms);
    }
}

// Optional: synchronize an external mode change from root
void pm_sync_mode_from_root(pm_context * ctx) {
    PM_ASSERT(ctx);
    pm_mode m = ctx->mode;
    pm_broadcast_mode(&m, /*root=*/0);
    if (m != ctx->mode) {
        pm_set_mode(ctx, m);
    }
}

static int ggml_graph_get_node_idx(struct ggml_cgraph * gf, const char * name) {
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == NULL) {
        fprintf(stderr, "%s: tensor %s not found\n", __func__, name);
        return -1;
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }

    fprintf(stderr, "%s: tensor %s not found in graph (should not happen)\n", __func__, name);
    return -1;
}

static void ggml_mpi_tensor_send(struct ggml_tensor * t, int mpi_rank_dst) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    const int retval = MPI_Send(t->data, ggml_nelements(t), mpi_type, mpi_rank_dst, 0, MPI_COMM_WORLD);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

static void ggml_mpi_tensor_recv(struct ggml_tensor * t, int mpi_rank_src) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Status status; UNUSED(status);

    const int retval = MPI_Recv(t->data, ggml_nelements(t), mpi_type, mpi_rank_src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

// TODO: there are many improvements that can be done to this implementation
void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    GGML_ASSERT(inp0 == gf->nodes[0]);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input tokens to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //
    if (mpi_rank > 0) {
        if (mpi_rank == 1) {
            // the first node (1) receives the input tokens from the main node (0)
            ggml_mpi_tensor_recv(inp_tokens, 0);
        } else {
            // recv input data for each node into the "inp0" tensor (i.e. the first node in the compute graph)
            ggml_mpi_tensor_recv(inp0, mpi_rank - 1);
        }
    } else if (mpi_size > 1) {
        // node 0 sends the input tokens to node 1
        ggml_mpi_tensor_send(inp_tokens, 1);

        // recv the output data from the last node
        ggml_mpi_tensor_recv(inp0, mpi_size - 1);
    }

    {
        const int n_per_node = (n_layers + (mpi_size - 1)) / mpi_size;

        const int mpi_idx = mpi_rank > 0 ? mpi_rank - 1 : mpi_size - 1;

        const int il0 =               (mpi_idx + 0) * n_per_node;
        const int il1 = MIN(n_layers, (mpi_idx + 1) * n_per_node);

        char name_l0[GGML_MAX_NAME];
        char name_l1[GGML_MAX_NAME];

        snprintf(name_l0, sizeof(name_l0), "layer_inp_%d", il0);
        snprintf(name_l1, sizeof(name_l1), "layer_inp_%d", il1);

        const int idx_l0 =                ggml_graph_get_node_idx(gf, name_l0);
        const int idx_l1 = mpi_rank > 0 ? ggml_graph_get_node_idx(gf, name_l1) + 1 : gf->n_nodes;

        if (idx_l0 < 0 || idx_l1 < 0) {
            fprintf(stderr, "%s: layer input nodes not found\n", __func__);
            return;
        }

        // attach the input data to all nodes that need it
        // TODO: not great - should be able to do this without modifying the compute graph (see next TODO below)
        for (int i = idx_l0; i < idx_l1; i++) {
            if (gf->nodes[i]->src[0] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[0] =  inp0;
            }
            if (gf->nodes[i]->src[1] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[1] =  inp0;
            }
        }

        // TODO: instead of rearranging the nodes, we should be able to execute a subset of the compute graph
        for (int i = 1; i < idx_l1 - idx_l0; i++) {
            gf->nodes[i] = gf->nodes[idx_l0 + i];
            gf->grads[i] = gf->grads[idx_l0 + i];
        }

        // the first node performs the "get_rows" operation, the rest of the nodes get the data from the previous node
        if (mpi_idx != 0) {
            gf->nodes[0]->op = GGML_OP_NONE;
        }

        gf->n_nodes = idx_l1 - idx_l0;

        //fprintf(stderr, "%s: node %d: processing %d nodes [%d, %d)\n", __func__, mpi_rank, gf->n_nodes, il0, il1);
    }
}

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    UNUSED(n_layers);

    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    // send the output data to the next node
    if (mpi_rank > 0) {
        ggml_mpi_tensor_send(gf->nodes[gf->n_nodes - 1], (mpi_rank + 1) % mpi_size);
    }
}
