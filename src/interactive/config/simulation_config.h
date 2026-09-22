#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace atlas::interactive {

struct SimulationConfig {
    using Vec3 = std::array<float, 3>;
    using Quat = std::array<float, 4>;

    enum class MaterialKind { molecule, atom, ion, neutron, solid };
    enum class GeometryKind {
        box,
        circle,
        cylinder,
        plane,
        sphere,
        square,
        triangle,
        triangle_mesh,
        polygonal_prism
    };
    enum class SourceKind { surface, volume };
    enum class GeneratorKind { uniform, jittering, maxwell_sigma, maxwell_boltzmann };
    enum class SinkKind { surface, volume, tracing };
    enum class DsmcKernelKind { hard_sphere, variable_hard_sphere, variable_soft_sphere };
    enum class DiffuseSamplingKind { cosine_weighted, uniform };

    struct Material {
        MaterialKind kind = MaterialKind::molecule;
        float mass = 0.0f;
        float translational_energy = 0.0f;
        float rotational_energy = 0.0f;
        float vibrational_energy = 0.0f;
        float reference_diameter = 0.0f;
        float reference_temperature = 0.0f;
        float viscosity_index = 0.5f;
        float scattering_parameter = 1.0f;
    };

    struct Geometry {
        GeometryKind kind = GeometryKind::box;
        Vec3 lower_corner { -1.0f, -1.0f, -1.0f };
        Vec3 upper_corner { 1.0f, 1.0f, 1.0f };
        Vec3 center { 0.0f, 0.0f, 0.0f };
        Vec3 normal { 0.0f, 0.0f, 1.0f };
        Vec3 a { 0.0f, 0.0f, 0.0f };
        Vec3 b { 1.0f, 0.0f, 0.0f };
        Vec3 c { 0.0f, 1.0f, 0.0f };
        float radius = 1.0f;
        float height = 1.0f;
        float offset = 0.0f;
        float side_length = 1.0f;
        int side_count = 3;
        bool open = false;
        std::vector<std::array<Vec3, 3>> triangles;
    };

    struct Unit {
        Geometry geometry;
        Vec3 translation { 0.0f, 0.0f, 0.0f };
        Quat orientation { 1.0f, 0.0f, 0.0f, 0.0f };
        std::optional<Vec3> velocity;
        std::optional<Vec3> acceleration;
        std::optional<Vec3> angular_velocity;
        std::optional<Vec3> angular_acceleration;
    };

    struct Fluid {
        std::size_t buffer_size = 0;
        std::size_t particle_count = 0;
        float statistical_weight = 1.0f;
        std::vector<Material> materials;
        std::vector<Vec3> position;
        std::vector<Vec3> velocity;
        std::vector<std::size_t> species;
        std::optional<std::vector<float>> temperature;
        std::optional<std::vector<float>> translational_energy;
        std::optional<std::vector<float>> rotational_energy;
        std::optional<std::vector<float>> vibrational_energy;
    };

    struct Universe {
        Vec3 lower_corner { 0.0f, 0.0f, 0.0f };
        Vec3 upper_corner { 1.0f, 1.0f, 1.0f };
        std::optional<Geometry> geometry;
        float cell_size = 1.0f;
        std::optional<std::vector<float>> temperature;
        std::optional<std::vector<Vec3>> bulk_velocity;
        std::optional<std::vector<Vec3>> field_force;
        std::optional<std::vector<Vec3>> gravity;
        std::optional<std::vector<float>> thermal_energy;
        std::optional<std::vector<float>> knudsen_number;
    };

    struct Solver {
        DsmcKernelKind kernel = DsmcKernelKind::hard_sphere;
        int majorant_sample_pairs = 8;
        int majorant_exhaustive_limit = 5;
    };

    struct Source {
        SourceKind kind = SourceKind::volume;
        Unit unit;
        float tolerance = 0.0f;
        float spacing = 1.0f;
    };

    struct Generator {
        GeneratorKind kind = GeneratorKind::uniform;
        std::vector<float> species_ratios { 1.0f };
        std::vector<float> species_numbers { 0.0f };
        std::vector<float> species_mass;
        float temperature = 0.0f;
        float min_value = 0.0f;
        float max_value = 1.0f;
        float base_value = 0.0f;
        float jitter_radius = 1.0f;
        float sigma = 1.0f;
        Vec3 bulk_velocity { 0.0f, 0.0f, 0.0f };
        std::uint32_t seed = 0;
    };

    struct Emitter {
        Source source;
        Generator generator;
    };

    struct Collider {
        Unit unit;
        float momentum_accommodation_coefficient = 1.0f;
        float restitution = 1.0f;
        DiffuseSamplingKind diffuse_sampling = DiffuseSamplingKind::uniform;
    };

    struct Sink {
        SinkKind kind = SinkKind::volume;
        Unit unit;
        float tolerance = 0.0f;
    };

    struct Codec {
        float representative_characteristic_length = 1.0f;
        float representative_collision_cross_sectional_area = 1.0f;
        float representative_statistical_weight = 1.0f;
        float representative_cell_volume = 1.0f;
    };

    float dt = 0.01f;
    Fluid fluid;
    Universe universe;
    std::vector<Solver> solvers;
    std::vector<Emitter> emitters;
    std::vector<Collider> colliders;
    std::vector<Sink> sinks;
    std::optional<Codec> codec;
};

}
