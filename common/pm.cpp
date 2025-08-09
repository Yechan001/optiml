#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string>
#include <chrono>
#include <thread>
#include "sampling.h"
#include "train.h"
#include "log.h"

#ifndef UNUSED
#   define UNUSED(x) (void)(x)
#endif

#ifndef PM_ASSERT
#   define PM_ASSERT(x) do { if (!(x)) { fprintf(stderr, "PM_ASSERT failed: %s:%d\n", __FILE__, __LINE__); abort(); } } while (0)
#endif

enum pm_mode {
    PM_MODE_PERFORMANCE = 0,
    PM_MODE_EFFICIENCY  = 1,
    PM_MODE_BALANCED    = 2,
};

struct pm_telemetry {
    float cpu_util;
    float gpu_util;
    float pkg_power_w;
    float temp_c;
};

struct pm_limits {
    int cpu_freq_khz_target;
    int cpu_threads_allowed;
    int gpu_power_limit_w;
};

struct pm_context {
    pm_mode      mode;
    pm_limits    limits;
    pm_telemetry last;
    int          verbose;
    bool         adaptive_enabled;
};

// ---------------------- OS stub functions ----------------------

static void pm_os_apply_cpu_freq(int khz) {
    fprintf(stderr, "[OS] set CPU freq target = %d KHz\n", khz);
}

static void pm_os_apply_thread_cap(int nthreads) {
    fprintf(stderr, "[OS] set thread cap = %d\n", nthreads);
}

static void pm_os_apply_gpu_power_limit(int watts) {
    fprintf(stderr, "[OS] set GPU power limit = %d W\n", watts);
}

static pm_telemetry pm_os_query_telemetry() {
    pm_telemetry t{};
    t.cpu_util    = 0.3f + (float)(rand() % 70) / 100.0f;
    t.gpu_util    = 0.2f + (float)(rand() % 80) / 100.0f;
    t.pkg_power_w = 50.0f + (float)(rand() % 60);
    t.temp_c      = 55.0f + (float)(rand() % 40);
    return t;
}

// ---------------------- policy ----------------------

static pm_limits pm_policy_from_mode(pm_mode mode) {
    pm_limits lim{};
    switch (mode) {
        case PM_MODE_PERFORMANCE:
            lim.cpu_freq_khz_target = 5200000;
            lim.cpu_threads_allowed = -1;
            lim.gpu_power_limit_w   = 300;
            break;
        case PM_MODE_EFFICIENCY:
            lim.cpu_freq_khz_target = 2000000;
            lim.cpu_threads_allowed = 8;
            lim.gpu_power_limit_w   = 120;
            break;
        case PM_MODE_BALANCED:
        default:
            lim.cpu_freq_khz_target = 3600000;
            lim.cpu_threads_allowed = 16;
            lim.gpu_power_limit_w   = 180;
            break;
    }
    return lim;
}

static void pm_apply_limits(const pm_limits &lim) {
    pm_os_apply_cpu_freq(lim.cpu_freq_khz_target);
    pm_os_apply_thread_cap(lim.cpu_threads_allowed);
    pm_os_apply_gpu_power_limit(lim.gpu_power_limit_w);
}

static void pm_adaptive_trim(pm_context * ctx) {
    const auto &t = ctx->last;
    auto &lim = ctx->limits;

    // 过热保护
    if (t.temp_c > 85.0f) {
        lim.cpu_freq_khz_target = (int)(lim.cpu_freq_khz_target * 0.9f);
        lim.gpu_power_limit_w  -= 15;
    }

    // 空闲降频
    if (t.cpu_util < 0.25f && t.gpu_util < 0.25f) {
        lim.cpu_freq_khz_target -= 200000;
        lim.gpu_power_limit_w  -= 10;
    }

    if (lim.cpu_freq_khz_target < 1500000) lim.cpu_freq_khz_target = 1500000;
    if (lim.gpu_power_limit_w   < 80)       lim.gpu_power_limit_w   = 80;
}

// ---------------------- public API ----------------------

void pm_init(pm_context * ctx, pm_mode mode, bool adaptive, int verbose) {
    PM_ASSERT(ctx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->mode             = mode;
    ctx->verbose          = verbose;
    ctx->adaptive_enabled = adaptive;
    ctx->limits           = pm_policy_from_mode(mode);
    pm_apply_limits(ctx->limits);

    if (verbose) {
        fprintf(stderr, "[PM] Initialized in mode=%d adaptive=%d\n", (int)mode, adaptive);
    }
}

void pm_set_mode(pm_context * ctx, pm_mode mode) {
    PM_ASSERT(ctx);
    ctx->mode   = mode;
    ctx->limits = pm_policy_from_mode(mode);
    pm_apply_limits(ctx->limits);
    if (ctx->verbose) {
        fprintf(stderr, "[PM] Mode changed to %d\n", (int)mode);
    }
}

void pm_tick(pm_context * ctx) {
    ctx->last = pm_os_query_telemetry();
    if (ctx->adaptive_enabled) {
        pm_adaptive_trim(ctx);
        pm_apply_limits(ctx->limits);
    }
    if (ctx->verbose) {
        fprintf(stderr, "[PM] cpu_util=%.2f gpu_util=%.2f power=%.1fW temp=%.1fC\n",
                ctx->last.cpu_util, ctx->last.gpu_util, ctx->last.pkg_power_w, ctx->last.temp_c);
    }
}

void pm_run_loop(pm_context * ctx, int interval_ms) {
    while (true) {
        pm_tick(ctx);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}
