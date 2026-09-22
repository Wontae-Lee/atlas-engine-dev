#pragma once

#include "config/simulation_config.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/memory/memory.h>
#include <atlas/system/system.h>

#include <vector>

namespace atlas::interactive {

class SystemFactory final {
public:
    explicit SystemFactory(SimulationConfig config);

    atlas::SystemHostPtr create();
    const SimulationConfig& config() const noexcept;

private:
    SimulationConfig _config;
    std::vector<atlas::host_shared_ptr<atlas::TriangleMesh>> _mesh_owners;
};

}
