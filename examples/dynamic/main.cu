#include <atlas/atlas.h>
#include <vizkit/vizkit.h>

#include <memory>
#include <vector>

namespace {

using sim_t      = float;
using Vector3    = atlas::math::Vector<sim_t, 3>;
using Quaternion = atlas::math::Quaternion<sim_t>;
using LayerPtr   = std::shared_ptr<atlas::vizkit::Layer<sim_t>>;

const Vector3 kZero(0.0f, 0.0f, 0.0f);
constexpr sim_t kPlaneExtent = 1.2f;

atlas::SyncHostPtr<sim_t>
make_sync(const Vector3& position) {
    return atlas::system::Sync<sim_t>::builder()
        .with_rigid_pose(position, Quaternion())
        .make_host_shared();
}

atlas::UnitHostPtr<sim_t>
make_unit(const atlas::GeometryHostPtr<sim_t>& geometry,
          const Vector3& position,
          const Vector3& velocity,
          const Vector3& angular_velocity) {
    return atlas::system::Unit<sim_t>::builder()
        .with_geometry(geometry)
        .with_sync(make_sync(position))
        .with_velocity(velocity)
        .with_acceleration(kZero)
        .with_angular_velocity(angular_velocity)
        .with_angular_acceleration(kZero)
        .make_host_shared();
}

} // namespace

int
main() {
    std::vector<LayerPtr> layers;
    layers.reserve(7);

    {
        const auto geometry = atlas::geometry::Box<sim_t>::builder()
                                  .with_lower_corner(Vector3(-0.8f, -0.4f, -0.5f))
                                  .with_upper_corner(Vector3(0.8f, 0.4f, 0.5f))
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(-3.0f, 1.4f, 0.0f),
            Vector3(0.35f, 0.0f, 0.0f),
            Vector3(0.0f, 0.8f, 0.0f));

        layers.push_back(
            atlas::vizkit::BoxLayer<sim_t>::builder()
                .with_unit(unit)
                .make_shared());
    }

    {
        const auto geometry = atlas::geometry::Sphere<sim_t>::builder()
                                  .with_center(kZero)
                                  .with_radius(0.55f)
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(-1.0f, 1.4f, 0.0f),
            Vector3(-0.2f, 0.0f, 0.0f),
            Vector3(0.4f, 0.7f, 0.0f));

        layers.push_back(
            atlas::vizkit::SphereLayer<sim_t>::builder()
                .with_unit(unit)
                .make_shared());
    }

    {
        const auto geometry = atlas::geometry::Cylinder<sim_t>::builder()
                                  .with_center(kZero)
                                  .with_radius(0.45f)
                                  .with_height(1.4f)
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(1.0f, 1.4f, 0.0f),
            Vector3(0.25f, 0.0f, 0.0f),
            Vector3(0.0f, 0.6f, 0.6f));

        layers.push_back(
            atlas::vizkit::CylinderLayer<sim_t>::builder()
                .with_unit(unit)
                .make_shared());
    }

    {
        const auto geometry = atlas::geometry::Plane<sim_t>::builder()
                                  .with_point_normal(kZero, Vector3(0.0f, 1.0f, 0.0f))
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(3.0f, 1.0f, 0.0f),
            kZero,
            Vector3(0.3f, 0.0f, 0.0f));

        layers.push_back(
            atlas::vizkit::PlaneLayer<sim_t>::builder()
                .with_unit(unit)
                .with_extent(kPlaneExtent)
                .make_shared());
    }

    {
        const auto geometry = atlas::geometry::Circle<sim_t>::builder()
                                  .with_center(kZero)
                                  .with_normal(Vector3(0.35f, 0.75f, 0.55f))
                                  .with_radius(0.7f)
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(-3.0f, -1.4f, 0.0f),
            Vector3(0.15f, 0.0f, 0.0f),
            Vector3(0.55f, 0.15f, 0.65f));

        layers.push_back(
            atlas::vizkit::CircleLayer<sim_t>::builder()
                .with_unit(unit)
                .make_shared());
    }

    {
        const auto geometry = atlas::geometry::Triangle<sim_t>::builder()
                                  .with_vertices(
                                      Vector3(-0.8f, -0.45f, 0.0f),
                                      Vector3(0.8f, -0.35f, 0.0f),
                                      Vector3(0.0f, 0.75f, 0.0f))
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(-0.2f, -1.4f, 0.0f),
            Vector3(0.18f, 0.0f, 0.0f),
            Vector3(0.0f, 0.9f, 0.0f));

        layers.push_back(
            atlas::vizkit::TriangleLayer<sim_t>::builder()
                .with_unit(unit)
                .make_shared());
    }

    {
        atlas::HostBuffer<atlas::TriangleContainer4<sim_t>> triangles(4);

        triangles[0] = atlas::TriangleContainer4<sim_t>(
            Vector3(-0.6f, -0.6f, 0.0f),
            Vector3(0.6f, -0.6f, 0.0f),
            Vector3(0.0f, 0.0f, 0.9f),
            Vector3(0.0f, -0.83205f, 0.5547f));

        triangles[1] = atlas::TriangleContainer4<sim_t>(
            Vector3(0.6f, -0.6f, 0.0f),
            Vector3(0.0f, 0.6f, 0.0f),
            Vector3(0.0f, 0.0f, 0.9f),
            Vector3(0.78087f, 0.39043f, 0.48804f));

        triangles[2] = atlas::TriangleContainer4<sim_t>(
            Vector3(0.0f, 0.6f, 0.0f),
            Vector3(-0.6f, -0.6f, 0.0f),
            Vector3(0.0f, 0.0f, 0.9f),
            Vector3(-0.78087f, 0.39043f, 0.48804f));

        triangles[3] = atlas::TriangleContainer4<sim_t>(
            Vector3(-0.6f, -0.6f, 0.0f),
            Vector3(0.0f, 0.6f, 0.0f),
            Vector3(0.6f, -0.6f, 0.0f),
            Vector3(0.0f, 0.0f, -1.0f));

        const auto geometry = atlas::geometry::TriangleMesh<sim_t>::builder()
                                  .with_triangles(triangles)
                                  .make_host_shared();

        const auto unit = make_unit(
            geometry,
            Vector3(2.8f, -1.4f, 0.0f),
            Vector3(-0.18f, 0.0f, 0.0f),
            Vector3(0.5f, 0.4f, 0.0f));

        layers.push_back(
            atlas::vizkit::TriangleMeshLayer<sim_t>::builder()
                .with_unit(unit)
                .make_shared());
    }

    const auto fluid = atlas::system::Fluid<sim_t>::builder()
                           .with_buffer_size(0)
                           .make_host_shared();

    const auto system = atlas::System<sim_t>::builder()
                            .with_fluid(fluid)
                            .make_host_shared();

    auto viewer = atlas::vizkit::Viewer<sim_t>::builder()
                      .with_system(system)
                      .with_title("ATLAS Dynamic Geometry Viewer")
                      .build();

    for (const auto& layer : layers) {
        viewer.add_layer(layer);
    }

    return viewer.run();
}