
#include <atlas/memory/memory.h>
#include <cassert>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

using namespace atlas;

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

struct Data {
    int value;

    __host__ __device__ explicit Data(int v = 0)
        : value(v) { }

    __host__ __device__ int
    twice() const { return value * 2; }

    __host__ __device__ void
    increment() { value++; }
};

void
test_host_semantics() {
    printf("[Test 1] Host semantics\n");

    auto sp1 = make_device_shared<Data>(10);
    assert(sp1.use_count() == 1);

    auto sp2 = sp1;
    assert(sp1.use_count() == 2 && sp2.use_count() == 2);

    auto sp3 = std::move(sp2);
    assert(sp3.use_count() == 2);
    assert(!sp2);
    assert(sp1.use_count() == 2);

    device_shared_ptr<Data> sp4;
    sp4 = sp3;
    assert(sp1.use_count() == 3 && sp3.use_count() == 3 && sp4.use_count() == 3);

    device_shared_ptr<Data> sp5;
    sp5 = std::move(sp4);
    assert(sp5.use_count() == 3);
    assert(!sp4);
    assert(sp1.use_count() == 3);

    assert(sp1->twice() == 20);
    sp1->increment();
    assert(sp5->twice() == 22);

    printf("  use_count ok, value=%d\n", sp1->value);
}

void
test_thrust_for_each_no_atomic() {
    printf("[Test 2] Thrust for_each (raw borrow, no atomics)\n");

    auto sp = make_device_shared<Data>(21);
    assert(sp.use_count() == 1);

    const int N = 1 << 16;
    thrust::device_vector<int> out(N);

    Data* dptr   = sp.get();
    int* out_ptr = thrust::raw_pointer_cast(out.data());

    auto first = thrust::counting_iterator<int>(0);
    auto last  = first + N;

    thrust::for_each(thrust::device, first, last, [dptr, out_ptr] __device__(int i) {
        out_ptr[i] = dptr->twice() + i;
        if (i == 0) dptr->increment();
    });
    CHECK_CUDA(cudaDeviceSynchronize());

    int h0 = out[0];
    int h1 = out[1];
    assert(h0 == 42 && h1 == 43);

    assert(sp.use_count() == 1);

    assert(sp->value == 22);

    printf("  out[0]=%d out[1]=%d, value=%d, use_count=%d\n", h0, h1, sp->value, sp.use_count());
}

__global__ void
kernel_copy_stress(device_shared_ptr<Data> sp, int copies, int* out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int acc = 0;
    for (int k = 0; k < copies; ++k) {
        device_shared_ptr<Data> q = sp;
        acc += q->twice();
    }
    if (i == 0) sp->increment();

    out[i] = acc;
}

void
test_kernel_atomic_stress() {
    printf("[Test 3] Kernel atomic stress (many copies)\n");

    auto sp     = make_device_shared<Data>(5);
    const int N = 256, copies = 1024;

    int* d_out = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_out, N * sizeof(int)));

    kernel_copy_stress<<<1, N>>>(sp, copies, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    int expected_per_thread = copies * (2 * 5);
    assert(d_out[0] == expected_per_thread);

    assert(sp->value == 6);

    assert(sp.use_count() == 1);

    printf("  acc0=%d (exp %d), value=%d, use_count=%d\n",
           d_out[0],
           expected_per_thread,
           sp->value,
           sp.use_count());

    CHECK_CUDA(cudaFree(d_out));
}

void
test_thrust_transform_reduce_delta() {
    printf("[Test 4] Thrust transform_reduce (aggregate delta -> single apply)\n");

    auto sp    = make_device_shared<Data>(7);
    Data* dptr = sp.get();

    const int N     = 1 << 15;
    const int iters = 16;

    auto begin = thrust::counting_iterator<int>(0);
    auto end   = begin + N;

    int total_delta = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        [dptr, iters] __device__(int i) -> int {
            int acc = 0;
            for (int k = 0; k < iters; ++k) {
                acc += dptr->twice();
            }
            (void)acc;
            return 0;
        },
        0,
        thrust::plus<int>());
    CHECK_CUDA(cudaDeviceSynchronize());

    int ref_before = sp.use_count();
    int ref_after  = ref_before + total_delta;
    assert(ref_before == ref_after);
    assert(sp.use_count() == ref_after);

    printf("  total_delta=%d, use_count(before)=%d, after=%d\n",
           total_delta,
           ref_before,
           ref_after);
}

void
test_lifetime_and_destruction() {
    printf("[Test 5] Lifetime / destruction\n");

    {
        auto sp = make_device_shared<Data>(100);
        assert(sp.use_count() == 1);
        assert(sp->twice() == 200);

        {
            auto sp2 = sp;
            assert(sp2.use_count() == 2);
        }
        assert(sp.use_count() == 1);
    }

    printf("  Scope ended. (cannot verify freed pointer safely from here)\n");
}

void
test_null_handle() {
    printf("[Test 6] Null/empty handle\n");
    device_shared_ptr<Data> empty;
    assert(!empty);
    assert(empty.use_count() == 0);

    assert(empty.get() == nullptr);
    printf("  Empty handle ok.\n");
}

__global__ void
kernel_raw_borrow(device_shared_ptr<Data> sp, int* out) {
    int i   = threadIdx.x + blockIdx.x * blockDim.x;
    Data* d = sp.get();
    out[i]  = d->twice() + i;
    if (i == 0) d->increment();
}

void
test_mixed_raw_borrow() {
    printf("[Test 7] Mixed: kernel raw borrow\n");

    auto sp     = make_device_shared<Data>(11);
    const int N = 128;
    int* d_out  = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_out, N * sizeof(int)));

    kernel_raw_borrow<<<1, N>>>(sp, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    assert(d_out[0] == 22);
    assert(d_out[1] == 23);

    assert(sp->value == 12);

    assert(sp.use_count() == 1);

    printf("  out[0]=%d, out[1]=%d, value=%d, use_count=%d\n",
           d_out[0],
           d_out[1],
           sp->value,
           sp.use_count());

    CHECK_CUDA(cudaFree(d_out));
}

int
main() {
    printf("=== device_shared_ptr comprehensive tests ===\n");

    test_host_semantics();
    test_thrust_for_each_no_atomic();
    test_kernel_atomic_stress();
    test_thrust_transform_reduce_delta();
    test_lifetime_and_destruction();
    test_null_handle();
    test_mixed_raw_borrow();

    printf("All tests passed.\n");
    return 0;
}