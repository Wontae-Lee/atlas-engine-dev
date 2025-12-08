#pragma once
#include <iterator>
#ifdef ATLAS_TASKING_CUDA
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
namespace atlas {
struct device_policy_t { };
static constexpr device_policy_t device {};
template <typename Policy, typename Iterator, typename Predicate>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Iterator
remove_if(Policy, Iterator first, Iterator last, Predicate pred) {
return thrust::remove_if(thrust::device, first, last, pred);
}
}
#else
#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <vector>
namespace atlas {
struct device_policy_t { };
static constexpr device_policy_t device {};
template <typename Policy, typename Iterator, typename Predicate>
inline Iterator
remove_if(Policy, Iterator first, Iterator last, Predicate pred) {
using diff_type  = typename std::iterator_traits<Iterator>::difference_type;
using value_type = typename std::iterator_traits<Iterator>::value_type;
diff_type n = std::distance(first, last);
if (n <= 0) {
return last;
}
std::vector<unsigned char> keep(static_cast<std::size_t>(n));
tbb::parallel_for(
diff_type(0),
n,
[&](diff_type i) {
keep[static_cast<std::size_t>(i)] = pred(*(first + i)) ? 0 : 1;
});
std::vector<diff_type> positions(static_cast<std::size_t>(n));
diff_type total_kept = tbb::parallel_scan(
tbb::blocked_range<diff_type>(0, n),
diff_type(0),
[&](const tbb::blocked_range<diff_type>& r, diff_type sum, bool is_final) {
for (diff_type i = r.begin(); i != r.end(); ++i) {
if (keep[static_cast<std::size_t>(i)]) {
if (is_final) {
positions[static_cast<std::size_t>(i)] = sum;
}
++sum;
} else if (is_final) {
positions[static_cast<std::size_t>(i)] = diff_type(-1);
}
}
return sum;
},
std::plus<diff_type>());
if (total_kept == 0) {
return first;
}
std::vector<value_type> temp(static_cast<std::size_t>(total_kept));
tbb::parallel_for(
diff_type(0),
n,
[&](diff_type i) {
if (keep[static_cast<std::size_t>(i)]) {
diff_type dst                       = positions[static_cast<std::size_t>(i)];
temp[static_cast<std::size_t>(dst)] = *(first + i);
}
});
for (diff_type i = 0; i < total_kept; ++i) {
*(first + i) = std::move(temp[static_cast<std::size_t>(i)]);
}
return first + total_kept;
}
}
#endif