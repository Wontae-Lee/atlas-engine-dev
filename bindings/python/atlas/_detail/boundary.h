#pragma once

#include "handles.h"

#include <atlas/collider/collider.h>
#include <atlas/collider/diffuse_sampling.h>
#include <atlas/collider/isothermal_collider.h>
#include <atlas/math/vector/float3.h>
#include <atlas/sink/sink.h>
#include <atlas/sink/surface_sink.h>
#include <atlas/sink/tracing_sink.h>
#include <atlas/sink/volume_sink.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

namespace {

    struct BoundaryUnit final {
        template <typename Boundary>
        ATLAS_ALL_DEVICE Unit
        operator()(const Boundary& boundary) const noexcept {
            return boundary.unit();
        }
    };

}

}
