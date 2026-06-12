#pragma once

#include <atlas/searcher/searcher.h>

namespace atlas {

template <typename T>
class QuadtreeSearcher final : public Searcher<T> {
public:
    class Builder;

    QuadtreeSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit QuadtreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    ~QuadtreeSearcher() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build() override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_neighbors(int alive, const Vector3<T>* pos);
};

template <typename T>
class QuadtreeSearcher<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE QuadtreeSearcher<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<QuadtreeSearcher<T>>
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
using QuadtreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher<T>>;

}

#include <atlas/searcher/quadtree_searcher.hpp>