#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

template <typename T>
class Matter {
public:
    class Builder;

    T mass = T(1);

    Matter() = default;

    virtual ~Matter() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Matter(T mass_);

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

template <typename T>
class Matter<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Matter<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Matter<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_mass(T mass) noexcept;

private:
    void
    validate() const;

private:
    T _mass = T(1);
};

}

namespace atlas {

template <typename T>
using Matter = system::Matter<T>;

template <typename T>
using MatterHostPtr = atlas::host_shared_ptr<system::Matter<T>>;

template <typename T>
using MatterDevicePtr = atlas::device_shared_ptr<system::Matter<T>>;

}

#include <atlas/matter/matter.hpp>