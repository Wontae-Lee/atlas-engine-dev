#pragma once
/**
 * @file parallel.h
 * @brief Umbrella header for the parallel-algorithm helpers.
 *
 * Pulls in the ExecutionPolicy-dispatched building blocks used across the
 * engine so a translation unit can include one header instead of the individual
 * pieces: `parallel_for` / `parallel_fill` (element and range operations) and
 * `parallel_sort` / `parallel_sort_by_key` (ordering). All are parameterized on
 * @ref atlas::ExecutionPolicy and select a Thrust or standard-library backend at
 * compile time. Note that @ref atlas::atomic_add (`atlas/parallel/atomic.h`) is
 * intentionally not aggregated here; include it directly where a device-side
 * atomic is needed.
 */
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>