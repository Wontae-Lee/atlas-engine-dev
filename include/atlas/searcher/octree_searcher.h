#pragma once

#include <atlas/searcher/searcher.h>

namespace atlas {

template <typename T>
class OctreeSearcher final : public Searcher<T> {
public:
    class Builder;

    OctreeSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit OctreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    ~OctreeSearcher() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build() override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_neighbors(int alive, const Vector3<T>* pos);
};

template <typename T>
class OctreeSearcher<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE OctreeSearcher<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<OctreeSearcher<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};
};

}

namespace atlas {

template <typename T>
using OctreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher<T>>;

}

#include <atlas/searcher/octree_searcher.hpp>