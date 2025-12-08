#pragma once
#include <atlas/parallel/parallel_for.h>
namespace atlas {
#if defined(ATLAS_TASKING_CUDA)
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
namespace detail {
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        if (first == last) {
            return result;
        }
        return thrust::exclusive_scan(
            thrust::host,
            first,
            last,
            result,
            init,
            binary_op);
    }
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        if (first == last) {
            return result;
        }
        return thrust::exclusive_scan(
            thrust::device,
            first,
            last,
            result,
            init,
            binary_op);
    }
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        if (first == last) {
            return result;
        }
        return thrust::exclusive_scan(
            thrust::seq,
            first,
            last,
            result,
            init,
            binary_op);
    }
}
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::exclusive_scan_host_impl(first, last, result, init, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::exclusive_scan_device_impl(first, last, result, init, binary_op);
    } else {
        return detail::exclusive_scan_serial_impl(first, last, result, init, binary_op);
    }
}
#include <functional>
#include <iterator>
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(
        first,
        last,
        result,
        init,
        std::plus<value_type> {});
}
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(
        first,
        last,
        result,
        value_type {},
        std::plus<value_type> {});
}
#else
#include <functional>
#include <iterator>
#include <tbb/tbb.h>
#include <type_traits>
namespace detail {
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        if (first == last) {
            return result;
        }
        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::exclusive_scan (TBB backend) requires a random-access iterator.");
        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);
        struct ScanBody {
            InputIt first;
            OutputIt result;
            BinaryOp binary_op;
            T init;
            T sum;
            ScanBody(InputIt f, OutputIt r, BinaryOp op, T init_val)
                : first(f)
                , result(r)
                , binary_op(op)
                , init(init_val)
                , sum(init_val) { }
            ScanBody(ScanBody& other, tbb::split)
                : first(other.first)
                , result(other.result)
                , binary_op(other.binary_op)
                , init(other.init)
                , sum(other.init) { }
            void
            operator()(const tbb::blocked_range<diff_t>& r, tbb::pre_scan_tag) {
                T temp = sum;
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    temp = binary_op(temp, first[i]);
                }
                sum = temp;
            }
            void
            operator()(const tbb::blocked_range<diff_t>& r, tbb::final_scan_tag) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    T temp    = sum;
                    sum       = binary_op(sum, first[i]);
                    result[i] = temp;
                }
            }
            void
            reverse_join(ScanBody& rhs) {
                sum = binary_op(rhs.sum, sum);
            }
            void
            assign(ScanBody& rhs) {
                sum = rhs.sum;
            }
        };
        ScanBody body(first, result, binary_op, init);
        tbb::parallel_scan(tbb::blocked_range<diff_t>(0, n), body);
        return result + n;
    }
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        return exclusive_scan_host_impl(first, last, result, init, binary_op);
    }
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        T sum = init;
        for (; first != last; ++first, ++result) {
            *result = sum;
            sum     = binary_op(sum, *first);
        }
        return result;
    }
}
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::exclusive_scan_host_impl(first, last, result, init, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::exclusive_scan_device_impl(first, last, result, init, binary_op);
    } else {
        return detail::exclusive_scan_serial_impl(first, last, result, init, binary_op);
    }
}
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(
        first,
        last,
        result,
        init,
        std::plus<value_type> {});
}
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(
        first,
        last,
        result,
        value_type {},
        std::plus<value_type> {});
}
#endif
}