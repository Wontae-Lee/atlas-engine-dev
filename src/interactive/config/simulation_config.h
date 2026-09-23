/**
 * @file
 * @brief Defines the serializable configuration used to construct an Atlas simulation.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace atlas::interactive {

/**
 * @brief Complete application-side description of an Atlas simulation.
 *
 * This plain data model is suitable for transport decoding. SystemFactory
 * translates it into the corresponding Atlas core objects.
 */
struct SimulationConfig {
    /// Three-component vector used before conversion to atlas::Float3.
    using Vec3 = std::array<float, 3>;
    /// Scalar-first quaternion used before conversion to Atlas math types.
    using Quat = std::array<float, 4>;

    /// Supported Atlas material leaf types.
    enum class MaterialKind {
        molecule, ///< Polyatomic molecular species.
        atom,     ///< Neutral atomic species.
        ion,      ///< Charged species.
        neutron,  ///< Neutron species.
        solid     ///< Solid material.
    };
    /// Supported Atlas geometry leaf types.
    enum class GeometryKind {
        box,              ///< Axis-aligned local box.
        circle,           ///< Planar circle.
        cylinder,         ///< Z-axis cylinder.
        plane,            ///< Infinite plane.
        sphere,           ///< Sphere.
        square,           ///< Planar square.
        triangle,         ///< Single triangle.
        triangle_mesh,    ///< Explicit triangle collection.
        polygonal_prism   ///< Regular polygonal prism.
    };
    /// Supported source boundary behaviors.
    enum class SourceKind {
        surface, ///< Emits from a surface boundary.
        volume   ///< Emits throughout an enclosed volume.
    };
    /// Supported particle generator distributions.
    enum class GeneratorKind {
        uniform,            ///< Uniform component sampling.
        jittering,          ///< Jitter around a base value.
        maxwell_sigma,      ///< Maxwell sampling with explicit sigma.
        maxwell_boltzmann   ///< Material-aware Maxwell-Boltzmann sampling.
    };
    /// Supported sink boundary behaviors.
    enum class SinkKind {
        surface, ///< Removes particles crossing a surface.
        volume,  ///< Removes particles inside a volume.
        tracing  ///< Removes particles whose trajectory hits the boundary.
    };
    /// Supported DSMC collision kernels.
    enum class DsmcKernelKind {
        hard_sphere,          ///< Hard-sphere collision model.
        variable_hard_sphere, ///< Variable-hard-sphere collision model.
        variable_soft_sphere  ///< Variable-soft-sphere collision model.
    };
    /// Supported diffuse-reflection sampling distributions.
    enum class DiffuseSamplingKind {
        cosine_weighted, ///< Lambertian hemisphere sampling.
        uniform          ///< Uniform hemisphere sampling.
    };

    /// Material properties used to construct one Atlas species.
    struct Material {
        MaterialKind kind = MaterialKind::molecule; ///< Concrete material leaf type.
        float mass = 0.0f; ///< Single-particle mass.
        float translational_energy = 0.0f; ///< Initial translational energy.
        float rotational_energy = 0.0f; ///< Initial rotational energy.
        float vibrational_energy = 0.0f; ///< Initial vibrational energy.
        float reference_diameter = 0.0f; ///< Collision reference diameter.
        float reference_temperature = 0.0f; ///< Collision reference temperature.
        float viscosity_index = 0.5f; ///< Temperature exponent for viscosity.
        float scattering_parameter = 1.0f; ///< Angular scattering parameter.
    };

    /// Geometry parameters interpreted according to kind.
    struct Geometry {
        GeometryKind kind = GeometryKind::box; ///< Concrete geometry leaf type.
        Vec3 lower_corner { -1.0f, -1.0f, -1.0f }; ///< Box lower corner.
        Vec3 upper_corner { 1.0f, 1.0f, 1.0f }; ///< Box upper corner.
        Vec3 center { 0.0f, 0.0f, 0.0f }; ///< Primitive center.
        Vec3 normal { 0.0f, 0.0f, 1.0f }; ///< Plane or planar-shape normal.
        Vec3 a { 0.0f, 0.0f, 0.0f }; ///< First triangle vertex.
        Vec3 b { 1.0f, 0.0f, 0.0f }; ///< Second triangle vertex.
        Vec3 c { 0.0f, 1.0f, 0.0f }; ///< Third triangle vertex.
        float radius = 1.0f; ///< Radial primitive radius.
        float height = 1.0f; ///< Axial primitive height.
        float offset = 0.0f; ///< Plane offset along its normal.
        float side_length = 1.0f; ///< Square side length.
        int side_count = 3; ///< Polygonal-prism side count.
        bool open = false; ///< Whether a cylinder omits end caps.
        std::vector<std::array<Vec3, 3>> triangles; ///< Explicit faces for triangle-mesh geometry.
    };

    /// Geometry with rigid-body pose and optional kinematics.
    struct Unit {
        Geometry geometry; ///< Local-space boundary geometry.
        Vec3 translation { 0.0f, 0.0f, 0.0f }; ///< Initial world translation.
        Quat orientation { 1.0f, 0.0f, 0.0f, 0.0f }; ///< Initial scalar-first rotation.
        std::optional<Vec3> velocity; ///< Optional linear velocity.
        std::optional<Vec3> acceleration; ///< Optional linear acceleration.
        std::optional<Vec3> angular_velocity; ///< Optional angular velocity.
        std::optional<Vec3> angular_acceleration; ///< Optional angular acceleration.
    };

    /// Fluid capacity, species table, and initial particle states.
    struct Fluid {
        std::size_t buffer_size = 0; ///< Allocated particle capacity.
        std::size_t particle_count = 0; ///< Initial live particle count.
        float statistical_weight = 1.0f; ///< Real particles represented by one simulated particle.
        std::vector<Material> materials; ///< Species material dictionary.
        std::vector<Vec3> position; ///< Initial live positions.
        std::vector<Vec3> velocity; ///< Initial live velocities.
        std::vector<std::size_t> species; ///< Initial live species indices.
        std::optional<std::vector<float>> temperature; ///< Optional initial particle temperature.
        std::optional<std::vector<float>> translational_energy; ///< Optional initial translational energy.
        std::optional<std::vector<float>> rotational_energy; ///< Optional initial rotational energy.
        std::optional<std::vector<float>> vibrational_energy; ///< Optional initial vibrational energy.
    };

    /// Simulation domain, grid resolution, and optional cell states.
    struct Universe {
        Vec3 lower_corner { 0.0f, 0.0f, 0.0f }; ///< Explicit domain lower corner.
        Vec3 upper_corner { 1.0f, 1.0f, 1.0f }; ///< Explicit domain upper corner.
        std::optional<Geometry> geometry; ///< Optional geometry-derived domain.
        float cell_size = 1.0f; ///< Uniform spatial-grid cell edge length.
        std::optional<std::vector<float>> temperature; ///< Optional per-cell temperature.
        std::optional<std::vector<Vec3>> bulk_velocity; ///< Optional per-cell mean velocity.
        std::optional<std::vector<Vec3>> field_force; ///< Optional per-cell external force.
        std::optional<std::vector<Vec3>> gravity; ///< Optional per-cell gravitational acceleration.
        std::optional<std::vector<float>> thermal_energy; ///< Optional per-cell thermal energy.
        std::optional<std::vector<float>> knudsen_number; ///< Optional per-cell Knudsen number.
    };

    /// Configuration for one DSMC solver.
    struct Solver {
        DsmcKernelKind kernel = DsmcKernelKind::hard_sphere; ///< Collision-kernel model.
        int majorant_sample_pairs = 8; ///< Pair samples for majorant estimation.
        int majorant_exhaustive_limit = 5; ///< Cell occupancy threshold for exhaustive estimation.
    };

    /// Boundary and placement rules for particle emission.
    struct Source {
        SourceKind kind = SourceKind::volume; ///< Source leaf type.
        Unit unit; ///< Emission boundary and motion.
        float tolerance = 0.0f; ///< Boundary inclusion tolerance.
        float spacing = 1.0f; ///< Candidate emission-point spacing.
    };

    /// Initial-state distribution applied to emitted particles.
    struct Generator {
        GeneratorKind kind = GeneratorKind::uniform; ///< Generator leaf type.
        std::vector<float> species_ratios { 1.0f }; ///< Relative emitted abundance by species.
        std::vector<float> species_numbers { 0.0f }; ///< Absolute species-number parameters.
        std::vector<float> species_mass; ///< Explicit species masses when no dictionary is used.
        float temperature = 0.0f; ///< Emission temperature.
        float min_value = 0.0f; ///< Uniform-distribution lower bound.
        float max_value = 1.0f; ///< Uniform-distribution upper bound.
        float base_value = 0.0f; ///< Jitter distribution center.
        float jitter_radius = 1.0f; ///< Jitter distribution radius.
        float sigma = 1.0f; ///< Maxwell-sigma distribution scale.
        Vec3 bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Mean emitted-particle velocity.
        std::uint32_t seed = 0; ///< Deterministic random seed.
    };

    /// Source and generator pair installed together on System.
    struct Emitter {
        Source source; ///< Spatial emission policy.
        Generator generator; ///< Emitted-particle state policy.
    };

    /// Isothermal boundary-collision configuration.
    struct Collider {
        Unit unit; ///< Collision boundary and motion.
        float momentum_accommodation_coefficient = 1.0f; ///< Diffuse-reflection fraction.
        float restitution = 1.0f; ///< Normal-velocity restitution coefficient.
        DiffuseSamplingKind diffuse_sampling = DiffuseSamplingKind::uniform; ///< Diffuse direction law.
    };

    /// Particle-removal boundary configuration.
    struct Sink {
        SinkKind kind = SinkKind::volume; ///< Sink leaf type.
        Unit unit; ///< Removal boundary and motion.
        float tolerance = 0.0f; ///< Boundary inclusion tolerance.
    };

    /// Knudsen codec parameters used for per-cell solver allocation.
    struct Codec {
        float representative_characteristic_length = 1.0f; ///< Reference macroscopic length.
        float representative_collision_cross_sectional_area = 1.0f; ///< Reference collision area.
        float representative_statistical_weight = 1.0f; ///< Reference particle weight.
        float representative_cell_volume = 1.0f; ///< Reference grid-cell volume.
    };

    float dt = 0.01f; ///< Simulation time step.
    Fluid fluid; ///< Initial particle storage and states.
    Universe universe; ///< Spatial domain and cell states.
    std::vector<Solver> solvers; ///< Collision solvers installed in declaration order.
    std::vector<Emitter> emitters; ///< Particle emitters installed in declaration order.
    std::vector<Collider> colliders; ///< Boundary colliders installed in declaration order.
    std::vector<Sink> sinks; ///< Particle sinks installed in declaration order.
    std::optional<Codec> codec; ///< Optional per-cell solver selector.
};

}
