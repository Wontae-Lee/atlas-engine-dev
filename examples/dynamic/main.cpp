#include <atlas/atlas.h>
#include <vizkit/vizkit.h>

int
main() {
    using sim_t      = float;
    using Vector3    = atlas::math::Vector<sim_t, 3>;
    using Quaternion = atlas::math::Quaternion<sim_t>;

    const auto make_sync = [](const Vector3& position) {
        return atlas::system::Sync<sim_t>::builder()
            .with_rigid_pose(position, Quaternion())
            .make_host_shared();
    };

    const auto make_unit =
        [&](const atlas::GeometryHostPtr<sim_t>& geometry,
            const Vector3& position,
            const Vector3& velocity,
            const Vector3& angular_velocity) {
            return atlas::system::Unit<sim_t>::builder()
                .with_geometry(geometry)
                .with_sync(make_sync(position))
                .with_velocity(velocity)
                .with_acceleration(Vector3(0.0f, 0.0f, 0.0f))
                .with_angular_velocity(angular_velocity)
                .with_angular_acceleration(Vector3(0.0f, 0.0f, 0.0f))
                .make_host_shared();
        };

    const auto box = atlas::geometry::Box<sim_t>::builder()
                         .with_lower_corner(Vector3(-0.8f, -0.4f, -0.5f))
                         .with_upper_corner(Vector3(0.8f, 0.4f, 0.5f))
                         .make_host_shared();

    const auto sphere = atlas::geometry::Sphere<sim_t>::builder()
                            .with_center(Vector3(0.0f, 0.0f, 0.0f))
                            .with_radius(0.55f)
                            .make_host_shared();

    const auto cylinder = atlas::geometry::Cylinder<sim_t>::builder()
                              .with_center(Vector3(0.0f, 0.0f, 0.0f))
                              .with_radius(0.45f)
                              .with_height(1.4f)
                              .make_host_shared();

    const auto plane = atlas::geometry::Plane<sim_t>::builder()
                           .with_point_normal(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f))
                           .make_host_shared();

    const auto circle = atlas::geometry::Circle<sim_t>::builder()
                            .with_center(Vector3(0.0f, 0.0f, 0.0f))
                            .with_normal(Vector3(0.35f, 0.75f, 0.55f))
                            .with_radius(0.7f)
                            .make_host_shared();

    const auto triangle = atlas::geometry::Triangle<sim_t>::builder()
                              .with_vertices(
                                  Vector3(-0.8f, -0.45f, 0.0f),
                                  Vector3(0.8f, -0.35f, 0.0f),
                                  Vector3(0.0f, 0.75f, 0.0f))
                              .make_host_shared();

    atlas::HostBuffer<atlas::TriangleContainer4<sim_t>> mesh_triangles;
    mesh_triangles.resize(4);
    mesh_triangles[0] = atlas::TriangleContainer4<sim_t>(
        Vector3(-0.6f, -0.6f, 0.0f),
        Vector3(0.6f, -0.6f, 0.0f),
        Vector3(0.0f, 0.0f, 0.9f),
        Vector3(0.0f, -0.83205f, 0.5547f));
    mesh_triangles[1] = atlas::TriangleContainer4<sim_t>(
        Vector3(0.6f, -0.6f, 0.0f),
        Vector3(0.0f, 0.6f, 0.0f),
        Vector3(0.0f, 0.0f, 0.9f),
        Vector3(0.78087f, 0.39043f, 0.48804f));
    mesh_triangles[2] = atlas::TriangleContainer4<sim_t>(
        Vector3(0.0f, 0.6f, 0.0f),
        Vector3(-0.6f, -0.6f, 0.0f),
        Vector3(0.0f, 0.0f, 0.9f),
        Vector3(-0.78087f, 0.39043f, 0.48804f));
    mesh_triangles[3] = atlas::TriangleContainer4<sim_t>(
        Vector3(-0.6f, -0.6f, 0.0f),
        Vector3(0.0f, 0.6f, 0.0f),
        Vector3(0.6f, -0.6f, 0.0f),
        Vector3(0.0f, 0.0f, -1.0f));

    const auto triangle_mesh = atlas::geometry::TriangleMesh<sim_t>::builder()
                                   .with_triangles(mesh_triangles)
                                   .make_host_shared();

    const auto box_unit           = make_unit(box, Vector3(-3.0f, 1.4f, 0.0f), Vector3(0.35f, 0.0f, 0.0f), Vector3(0.0f, 0.8f, 0.0f));
    const auto sphere_unit        = make_unit(sphere, Vector3(-1.0f, 1.4f, 0.0f), Vector3(-0.2f, 0.0f, 0.0f), Vector3(0.4f, 0.7f, 0.0f));
    const auto cylinder_unit      = make_unit(cylinder, Vector3(1.0f, 1.4f, 0.0f), Vector3(0.25f, 0.0f, 0.0f), Vector3(0.0f, 0.6f, 0.6f));
    const auto plane_unit         = make_unit(plane, Vector3(3.0f, 1.0f, 0.0f), Vector3(0.0f, 0.0f, 0.0f), Vector3(0.3f, 0.0f, 0.0f));
    const auto circle_unit        = make_unit(circle, Vector3(-3.0f, -1.4f, 0.0f), Vector3(0.15f, 0.0f, 0.0f), Vector3(0.55f, 0.15f, 0.65f));
    const auto triangle_unit      = make_unit(triangle, Vector3(-0.2f, -1.4f, 0.0f), Vector3(0.18f, 0.0f, 0.0f), Vector3(0.0f, 0.9f, 0.0f));
    const auto triangle_mesh_unit = make_unit(triangle_mesh, Vector3(2.8f, -1.4f, 0.0f), Vector3(-0.18f, 0.0f, 0.0f), Vector3(0.5f, 0.4f, 0.0f));

    const auto box_layer = atlas::vizkit::BoxLayer<sim_t>::builder()
                               .with_unit(box_unit)
                               .make_shared();
    const auto sphere_layer = atlas::vizkit::SphereLayer<sim_t>::builder()
                                  .with_unit(sphere_unit)
                                  .make_shared();
    const auto cylinder_layer = atlas::vizkit::CylinderLayer<sim_t>::builder()
                                    .with_unit(cylinder_unit)
                                    .make_shared();
    const auto plane_layer = atlas::vizkit::PlaneLayer<sim_t>::builder()
                                 .with_unit(plane_unit)
                                 .with_extent(1.2f)
                                 .make_shared();
    const auto circle_layer = atlas::vizkit::CircleLayer<sim_t>::builder()
                                  .with_unit(circle_unit)
                                  .make_shared();
    const auto triangle_layer = atlas::vizkit::TriangleLayer<sim_t>::builder()
                                    .with_unit(triangle_unit)
                                    .make_shared();
    const auto triangle_mesh_layer = atlas::vizkit::TriangleMeshLayer<sim_t>::builder()
                                         .with_unit(triangle_mesh_unit)
                                         .make_shared();

    const auto fluid = atlas::system::Fluid<sim_t>::builder()
                           .with_buffer_size(0)
                           .make_host_shared();

    const auto sim_system = atlas::System<sim_t>::builder()
                                .with_fluid(fluid)
                                .make_host_shared();

    auto viewer = atlas::vizkit::Viewer<sim_t>::builder()
                      .with_system(sim_system)
                      .with_title("ATLAS Dynamic Geometry Viewer")
                      .build();

    viewer.add_layer(box_layer);
    viewer.add_layer(sphere_layer);
    viewer.add_layer(cylinder_layer);
    viewer.add_layer(plane_layer);
    viewer.add_layer(circle_layer);
    viewer.add_layer(triangle_layer);
    viewer.add_layer(triangle_mesh_layer);

    return viewer.run();
}