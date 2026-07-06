#pragma once

#include <atlas/searcher/searcher.h>

namespace atlas {

class OctreeSearcher final : public Searcher {
public:
    class Builder;

    OctreeSearcher() = default;

    ATLAS_HOST explicit OctreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid);

    ~OctreeSearcher() override = default;

    ATLAS_HOST void
    build() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

public:
    ATLAS_HOST void
    build_neighbors(int alive, const Float3* pos);
};

class OctreeSearcher::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST OctreeSearcher
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<OctreeSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};
};

using OctreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher>;

}
