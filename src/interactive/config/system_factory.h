/**
 * @file
 * @brief Declares construction of Atlas core systems from interactive configuration.
 */

#pragma once

#include "config/simulation_config.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/memory/memory.h>
#include <atlas/system/system.h>

#include <vector>

namespace atlas::interactive {

/**
 * @brief Builds an Atlas core system from an interactive configuration.
 *
 * The factory retains mesh allocations referenced by the system because Atlas
 * geometry views do not take ownership of their source triangle meshes.
 */
class SystemFactory final {
public:
    /// Stores the configuration used by subsequent calls to create().
    explicit SystemFactory(SimulationConfig config);

    /// Creates a fully owned Atlas system from the stored configuration.
    atlas::SystemHostPtr create();

    /// Returns the immutable configuration used by this factory.
    const SimulationConfig& config() const noexcept;

private:
    SimulationConfig _config; ///< Source description for every generated system.
    std::vector<atlas::host_shared_ptr<atlas::TriangleMesh>> _mesh_owners; ///< Mesh lifetime anchors.
};

}
