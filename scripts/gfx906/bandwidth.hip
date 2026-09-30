// Copy bandwidth and latency between the host and the GPUs and between the GPUs (through pinned host memory as
// llama.cpp does without peer access, and with hipMemcpyPeer), to see what the PCIe / socket links deliver.
// Built and run by scripts/gfx906/bandwidth.sh.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                                    \
    do {                                                                                            \
        hipError_t err_ = (x);                                                                      \
        if (err_ != hipSuccess) {                                                                   \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(err_));   \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// seconds per call of fn, after one warmup call
template <typename F> static double time_per_call(F fn, int iters) {
    fn();
    CHECK(hipDeviceSynchronize());
    const double t0 = now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    return (now() - t0) / iters;
}

int main(int argc, char ** argv) {
    const size_t big   = (argc > 1 ? (size_t) atoll(argv[1]) : 256) << 20; // MB
    const size_t small = argc > 2 ? (size_t) atoll(argv[2]) : 2816*4;       // one activation row of Gemma 4 26B A4B

    int n_dev = 0;
    CHECK(hipGetDeviceCount(&n_dev));
    if (n_dev < 1) {
        fprintf(stderr, "no GPU\n");
        return 1;
    }

    std::vector<void *>      dbuf(n_dev);
    std::vector<hipStream_t> stream(n_dev);
    for (int d = 0; d < n_dev; ++d) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc(&dbuf[d], big));
        CHECK(hipMemset(dbuf[d], 0, big));
        CHECK(hipStreamCreateWithFlags(&stream[d], hipStreamNonBlocking));
    }
    void * host = nullptr;
    CHECK(hipHostMalloc(&host, big, hipHostMallocDefault));
    memset(host, 0, big);

    printf("buffer %zu MB, small copy %zu bytes\n\n", big >> 20, small);

    printf("host <-> GPU (pinned host memory)      GB/s      small us\n");
    for (int d = 0; d < n_dev; ++d) {
        CHECK(hipSetDevice(d));
        const double h2d   = big / time_per_call([&] { CHECK(hipMemcpy(dbuf[d], host, big, hipMemcpyHostToDevice)); }, 5) / 1e9;
        const double d2h   = big / time_per_call([&] { CHECK(hipMemcpy(host, dbuf[d], big, hipMemcpyDeviceToHost)); }, 5) / 1e9;
        const double h2d_s = time_per_call([&] {
            CHECK(hipMemcpyAsync(dbuf[d], host, small, hipMemcpyHostToDevice, stream[d]));
            CHECK(hipStreamSynchronize(stream[d]));
        }, 1000) * 1e6;
        const double d2h_s = time_per_call([&] {
            CHECK(hipMemcpyAsync(host, dbuf[d], small, hipMemcpyDeviceToHost, stream[d]));
            CHECK(hipStreamSynchronize(stream[d]));
        }, 1000) * 1e6;
        printf("  host -> GPU%d                     %8.2f  %8.1f\n", d, h2d, h2d_s);
        printf("  GPU%d -> host                     %8.2f  %8.1f\n", d, d2h, d2h_s);
    }

    if (n_dev < 2) {
        return 0;
    }

    printf("\nGPU -> GPU                             GB/s      small us   peer access\n");
    for (int a = 0; a < n_dev; ++a) {
        for (int b = 0; b < n_dev; ++b) {
            if (a == b) {
                continue;
            }
            int can_peer = 0;
            CHECK(hipDeviceCanAccessPeer(&can_peer, a, b));

            // through pinned host memory: D2H on the stream of a, H2D on the stream of b (llama.cpp staged copy)
            auto staged = [&](size_t n) {
                CHECK(hipSetDevice(a));
                CHECK(hipMemcpyAsync(host, dbuf[a], n, hipMemcpyDeviceToHost, stream[a]));
                CHECK(hipStreamSynchronize(stream[a]));
                CHECK(hipSetDevice(b));
                CHECK(hipMemcpyAsync(dbuf[b], host, n, hipMemcpyHostToDevice, stream[b]));
                CHECK(hipStreamSynchronize(stream[b]));
            };
            const double st_bw = big / time_per_call([&] { staged(big); }, 5) / 1e9;
            const double st_us = time_per_call([&] { staged(small); }, 1000) * 1e6;

            // HIP runtime peer copy (staged by the runtime itself without peer access)
            auto peer = [&](size_t n) {
                CHECK(hipSetDevice(a));
                CHECK(hipMemcpyPeerAsync(dbuf[b], b, dbuf[a], a, n, stream[a]));
                CHECK(hipStreamSynchronize(stream[a]));
            };
            const double pe_bw = big / time_per_call([&] { peer(big); }, 5) / 1e9;
            const double pe_us = time_per_call([&] { peer(small); }, 1000) * 1e6;

            printf("  GPU%d -> GPU%d staged via host      %8.2f  %8.1f   %s\n", a, b, st_bw, st_us, can_peer ? "yes" : "no");
            printf("  GPU%d -> GPU%d hipMemcpyPeer        %8.2f  %8.1f\n", a, b, pe_bw, pe_us);
        }
    }
    return 0;
}
