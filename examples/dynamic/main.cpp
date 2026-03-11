#include <atlas/atlas.h>
#include <vizkit/vizkit.h>

int
main() {
    const auto box = atlas::geometry::Box<float>::builder()
                         .with_lower_corner(atlas::math::Vector<float, 3>(-1.0f, -0.5f, -0.75f))
                         .with_upper_corner(atlas::math::Vector<float, 3>(1.0f, 0.5f, 0.75f))
                         .make_host_shared();

    const auto sync = atlas::system::Sync<float>::builder()
                          .with_rigid_pose(
                              atlas::math::Vector<float, 3>(0.0f, 1.0f, 0.0f),
                              atlas::math::Quaternion<float>())
                          .make_host_shared();

    const auto unit = atlas::system::Unit<float>::builder()
                          .with_geometry(box)
                          .with_sync(sync)
                          .with_velocity(atlas::math::Vector<float, 3>(0.8f, 0.0f, 0.0f))
                          .with_acceleration(atlas::math::Vector<float, 3>(0.0f, 0.0f, 0.0f))
                          .with_angular_velocity(atlas::math::Vector<float, 3>(0.0f, 1.0f, 0.0f))
                          .with_angular_acceleration(atlas::math::Vector<float, 3>(0.0f, 0.0f, 0.0f))
                          .make_host_shared();

    const auto box_layer = atlas::vizkit::BoxLayer<float>::builder()
                               .with_unit(unit)
                               .make_shared();

    auto viewer = atlas::vizkit::Viewer<float>::builder()
                      .with_title("ATLAS Box Viewer")
                      .build();

    viewer.add_layer(box_layer);

    return viewer.run();
}