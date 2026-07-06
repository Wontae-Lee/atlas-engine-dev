#pragma once

#include <atlas/searcher/searcher.h>

namespace atlas {

class KdTreeSearcher final : public Searcher {
public:
    class Builder;

    KdTreeSearcher() = default;

    ATLAS_HOST explicit KdTreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid);

    ~KdTreeSearcher() override = default;

    ATLAS_HOST void
    build() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

public:
    ATLAS_HOST void
    build_neighbors(int alive, const Float3* pos);
};

class KdTreeSearcher::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST KdTreeSearcher
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<KdTreeSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};
};

using KdTreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher>;

}
