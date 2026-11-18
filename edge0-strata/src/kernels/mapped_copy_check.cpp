// Check the latent activation transfer before the host uses CUDA synchronization.
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

void check(cudaError_t rc) {
    if (rc == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    const int rc = _putenv_s(name, value != nullptr ? value : "");
#else
    const int rc = value != nullptr ? setenv(name, value, 1) : unsetenv(name);
#endif
    if (rc != 0) std::exit(1);
}

}  // namespace

int check_shape(int tokens) {
    constexpr int width = 1792;
    const int rows = tokens * 8;
    const int compact_bytes = rows * (width / 32) * 34, f32_bytes = rows * width * 4;
    constexpr uint8_t poison = 0xe5;
    std::vector<uint8_t> compact_ref(compact_bytes), f32_ref(f32_bytes);
    for (size_t i = 0; i < compact_ref.size(); ++i) compact_ref[i] = (uint8_t) (13 * i + 5);
    for (size_t i = 0; i < f32_ref.size(); ++i) f32_ref[i] = (uint8_t) (31 * i + 17);

    uint8_t *compact_src = nullptr, *f32_src = nullptr, *compact_host = nullptr, *f32_host = nullptr;
    void *compact_dst = nullptr, *f32_dst = nullptr, *flag_dst = nullptr;
    uint32_t* flag_host = nullptr;
    check(cudaMalloc(&compact_src, compact_bytes));
    check(cudaMalloc(&f32_src, f32_bytes));
    check(cudaHostAlloc(&compact_host, compact_bytes, cudaHostAllocMapped));
    check(cudaHostAlloc(&f32_host, f32_bytes, cudaHostAllocMapped));
    check(cudaHostAlloc(&flag_host, sizeof(*flag_host), cudaHostAllocMapped));
    check(cudaHostGetDevicePointer(&compact_dst, compact_host, 0));
    check(cudaHostGetDevicePointer(&f32_dst, f32_host, 0));
    check(cudaHostGetDevicePointer(&flag_dst, flag_host, 0));
    check(cudaMemcpy(compact_src, compact_ref.data(), compact_bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(f32_src, f32_ref.data(), f32_bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const struct { const char* mode; const char* diagnostic; bool eligible; bool omit_f32; } cases[] = {
        {nullptr, nullptr, true, false},
        {"kernel", nullptr, true, true},
        {"kernel", nullptr, false, false},
        {"kernel", "STRATA_ACT_CHECK", true, false},
        {"kernel", "STRATA_ACT_FALLBACK", true, false},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); ++i) {
        const auto& c = cases[i];
        set_env("STRATA_LATENT_HOST_COPY", c.mode);
        set_env("STRATA_ACT_CHECK", nullptr);
        set_env("STRATA_ACT_FALLBACK", nullptr);
        if (c.diagnostic != nullptr) set_env(c.diagnostic, "0");   // Presence enables diagnostics.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        if (!strata::kernels::copy_latent_activations_to_host(
                compact_dst, compact_src, compact_bytes, f32_dst, f32_src, f32_bytes, c.eligible, stream))
            return 1;
        strata::kernels::doorbell_publish_value(nullptr, nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr,
                                               static_cast<uint32_t*>(flag_dst), 1, stream);
        check(cudaPeekAtLastError());
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        for (int replay = 0; replay < 64; ++replay) {
            std::memset(compact_host, poison, compact_bytes);
            std::memset(f32_host, poison, f32_bytes);
            *flag_host = 0;
            check(cudaGraphLaunch(executable, stream));
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (*static_cast<volatile uint32_t*>(flag_host) != 1) {
                if (std::chrono::steady_clock::now() >= until) {
                    std::fprintf(stderr, "mapped copy case %zu replay %d missed its ready flag\n", i, replay);
                    return 1;
                }
                std::this_thread::yield();
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);

            const bool compact_ok = std::memcmp(compact_host, compact_ref.data(), compact_bytes) == 0;
            const bool f32_ok = c.omit_f32
                ? std::all_of(f32_host, f32_host + f32_bytes, [](uint8_t v) { return v == poison; })
                : std::memcmp(f32_host, f32_ref.data(), f32_bytes) == 0;
            check(cudaStreamSynchronize(stream));
            if (!compact_ok || !f32_ok) {
                std::fprintf(stderr, "mapped copy case %zu replay %d failed, compact=%d f32=%d\n",
                             i, replay, compact_ok, f32_ok);
                return 1;
            }
        }
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream));
    check(cudaFreeHost(flag_host));
    check(cudaFreeHost(f32_host));
    check(cudaFreeHost(compact_host));
    check(cudaFree(f32_src));
    check(cudaFree(compact_src));
    std::printf("mapped latent activation copy OK: %d token rows, 5 paths, 64 replays each\n", tokens);
    return 0;
}

int main() {
    for (int tokens = 1; tokens <= 8; ++tokens) {
        if (check_shape(tokens) != 0) return 1;
    }
    std::puts("mapped latent activation copy OK (2560 checks, including fallback and reference check)");
}
