#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/searcher/searcher.h>

namespace atlas::detail {

template <typename Probe>
ATLAS_HOST void
fill_cell_partition(Probe& probe, const SearcherHostPtr& searcher) noexcept {
    probe.indices_ptr    = searcher->indices();
    probe.cell_start_ptr = searcher->cell_start();
    probe.cell_end_ptr   = searcher->cell_end();
}

template <typename State>
ATLAS_HOST auto
optional_state_ptr(State* state) noexcept
    -> decltype(atlas::raw_pointer_cast(state->data().data())) {
    return state != nullptr
        ? atlas::raw_pointer_cast(state->data().data())
        : nullptr;
}

template <typename Buffer>
ATLAS_HOST auto
optional_buffer_ptr(Buffer& buffer) noexcept
    -> decltype(atlas::raw_pointer_cast(buffer.data())) {
    return buffer.empty()
        ? nullptr
        : atlas::raw_pointer_cast(buffer.data());
}

}
