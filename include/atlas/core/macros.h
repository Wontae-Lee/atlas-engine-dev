#pragma once
#if defined(__CUDACC__)
#define ATLAS_HOST __host__
#define ATLAS_DEVICE __device__
#define ATLAS_ALL_DEVICE __host__ __device__
#define ATLAS_FORCE_INLINE __forceinline__
#define ATLAS_UNROLL _Pragma("unroll")
#else
#define ATLAS_HOST
#define ATLAS_DEVICE
#define ATLAS_ALL_DEVICE
#define ATLAS_UNROLL
#if defined(_MSC_VER)
#define ATLAS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ATLAS_FORCE_INLINE inline
#endif
#endif
#define ATLAS_NODISCARD [[nodiscard]]
#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif
#ifdef ATLAS_TASKING_CUDA
#define PRINT_DEVICE_PROPERTIES(dev)             \
    do {                                         \
        cudaSetDevice(dev);                      \
        cudaDeviceProp prop {};                  \
        cudaGetDeviceProperties(&prop, dev);     \
        printf("GPU[%d]: %s\n", dev, prop.name); \
    } while (0)
#else
#if defined(__linux__)
#define PRINT_DEVICE_PROPERTIES()                                           \
    do {                                                                    \
        FILE* fp = popen("grep 'model name' /proc/cpuinfo | head -1", "r"); \
        if (fp) {                                                           \
            char buf[256];                                                  \
            if (fgets(buf, sizeof(buf), fp)) printf("CPU: %s", buf);        \
            pclose(fp);                                                     \
        }                                                                   \
        fp = popen("nproc", "r");                                           \
        if (fp) {                                                           \
            char buf[64];                                                   \
            if (fgets(buf, sizeof(buf), fp)) printf("Cores: %s", buf);      \
            pclose(fp);                                                     \
        }                                                                   \
    } while (0)
#elif defined(_WIN32)
#include <intrin.h>
#include <windows.h>
#define PRINT_DEVICE_PROPERTIES()                                       \
    do {                                                                \
        int cpuInfo[4];                                                 \
        char brand[0x40];                                               \
        __cpuid(cpuInfo, 0x80000002);                                   \
        memcpy(brand, cpuInfo, 16);                                     \
        __cpuid(cpuInfo, 0x80000003);                                   \
        memcpy(brand + 16, cpuInfo, 16);                                \
        __cpuid(cpuInfo, 0x80000004);                                   \
        memcpy(brand + 32, cpuInfo, 16);                                \
        brand[48] = '\0';                                               \
        SYSTEM_INFO si;                                                 \
        GetSystemInfo(&si);                                             \
        printf("CPU: %s\nCores: %u\n", brand, si.dwNumberOfProcessors); \
    } while (0)
#elif defined(__APPLE__)
#define PRINT_DEVICE_PROPERTIES()                                      \
    do {                                                               \
        FILE* fp = popen("sysctl -n machdep.cpu.brand_string", "r");   \
        if (fp) {                                                      \
            char buf[256];                                             \
            if (fgets(buf, sizeof(buf), fp)) printf("CPU: %s", buf);   \
            pclose(fp);                                                \
        }                                                              \
        fp = popen("sysctl -n hw.physicalcpu", "r");                   \
        if (fp) {                                                      \
            char buf[64];                                              \
            if (fgets(buf, sizeof(buf), fp)) printf("Cores: %s", buf); \
            pclose(fp);                                                \
        }                                                              \
    } while (0)
#else
#define PRINT_DEVICE_PROPERTIES() printf("Unknown platform\n")
#endif
#endif