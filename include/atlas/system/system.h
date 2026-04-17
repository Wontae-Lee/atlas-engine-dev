#pragma once

#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::system {
template <typename T>
class System {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(atlas::host_shared_ptr<atlas::Fluid<T>> fluid);


private:
    T _dt { static_cast<T>(0.01) };

};
}

namespace atlas {

template <typename T>
using System = system::System<T>;

template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

}

#include <atlas/system/system.hpp>
