/**
 * @file
 * @brief Declares construction of Atlas core objects from interactive configuration.
 */

#pragma once

#include "config/simulation_config.h"

#include <atlas/atlas.h>
#include <atlas/memory/memory.h>
#include <atlas/system/system.h>

#include <cstddef>
#include <vector>

namespace atlas::interactive {

/**
 * @brief Builds Atlas core objects from interactive configuration.
 *
 * The factory retains mesh allocations referenced by built objects because Atlas
 * geometry views do not take ownership of their source triangle meshes.
 */
class CoreFactory final {
public:
    explicit CoreFactory(SimulationConfig config = {});

    atlas::Material build(const SimulationConfig::Material& config);
    atlas::Geometry build(const SimulationConfig::Geometry& config);
    atlas::Unit build(const SimulationConfig::Unit& config);
    atlas::FluidHostPtr build(const SimulationConfig::Fluid& config);
    atlas::UniverseHostPtr build(const SimulationConfig::Universe& config);
    atlas::SolverHostPtr build(const SimulationConfig::Solver& config);
    atlas::SourceHostPtr build(const SimulationConfig::Source& config);
    atlas::GeneratorHostPtr build(const SimulationConfig::Generator& config,
                                  const atlas::MaterialDictionaryHostPtr& materials = {});
    atlas::Collider build(const SimulationConfig::Collider& config);
    atlas::Sink build(const SimulationConfig::Sink& config);
    atlas::CodecHostPtr build(const SimulationConfig::Codec& config);
    atlas::SystemHostPtr build(const SimulationConfig& config);
    atlas::MaterialDictionaryHostPtr build_materials(
        const std::vector<SimulationConfig::Material>& configs);

    atlas::SystemHostPtr create();

    /// Returns the immutable configuration used by this factory.
    const SimulationConfig& config() const noexcept;

private:
    atlas::FluidHostPtr build_fluid(const SimulationConfig::Fluid& config,
                                    const atlas::MaterialDictionaryHostPtr& materials);

    SimulationConfig _config; ///< Source description for every generated system.
    std::vector<atlas::host_shared_ptr<atlas::TriangleMesh>> _mesh_owners; ///< Mesh lifetime anchors.
    std::size_t _mesh_index = 0;
};

using SystemFactory = CoreFactory;

}
