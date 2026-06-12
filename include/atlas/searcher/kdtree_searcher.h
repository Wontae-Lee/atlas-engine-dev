#pragma once

#include <atlas/searcher/searcher.h>

namespace atlas {

template <typename T>
class KdTreeSearcher final : public Searcher<T> {
public:
    class Builder;

    KdTreeSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit KdTreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    ~KdTreeSearcher() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build() override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_neighbors(int alive, const Vector3<T>* pos);
};

template <typename T>
class KdTreeSearcher<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE KdTreeSearcher<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KdTreeSearcher<T>>
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
using KdTreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher<T>>;

}

#include <atlas/searcher/kdtree_searcher.hpp>