#pragma once
#include <atlas/parallel/parallel_for.h>
#include <functional>
#include <iterator>

namespace atlas {

namespace detail {

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op);

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op);

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op);

}

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op);

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init);

template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result);

}

#include <atlas/scan/exclusive_scan.hpp>