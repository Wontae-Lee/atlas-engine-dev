#pragma once

#include <atlas/core/macros.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/shuffle/shuffle_operator.h>
#include <atlas/source/source_probe.h>

#include <cstddef>

namespace atlas::detail {

template <typename T>
class SourceEmitter final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(const SourceProbe<T>& probe, std::size_t dst_offset, std::size_t emit_count) const;
};

}

#include <atlas/source/detail/source_emitter.hpp>