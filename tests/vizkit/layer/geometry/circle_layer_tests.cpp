#include "../../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#define protected public
#include <vizkit/layer/geometry/circle_layer.h>
#undef protected

#include <gtest/gtest.h>

namespace {

template <typename T>
atlas::UnitHostPtr<T>
make_vizkit_circle_unit(const atlas::Vector3<T>& center,
                        const atlas::Vector3<T>& normal,
                        T radius) {
    const auto geometry = atlas::geometry::Circle<T>::builder()
                              .with_center(center)
                              .with_normal(normal)
                              .with_radius(radius)
                              .make_host_shared();
    const auto sync = atlas::system::Sync<T>::builder().make_host_shared();
    return atlas::system::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

} // namespace

TEST(VizkitCircleLayer, BuildGeometryProducesClosedPolylineOnCircleBoundary) {
    atlas::vizkit::CircleLayer<float> layer(
        make_vizkit_circle_unit<float>(
            atlas::Vector3<float>(1.0f, 2.0f, 3.0f),
            atlas::Vector3<float>(0.0f, 0.0f, 1.0f),
            2.0f),
        8);
    std::vector<atlas::Vector3<float>> positions;

    layer.build_geometry(positions);

    ASSERT_EQ(positions.size(), 16u);
    for (const auto& position : positions) {
        const auto radial = position - atlas::Vector3<float>(1.0f, 2.0f, 3.0f);
        EXPECT_NEAR(radial.z, 0.0f, 1e-5f);
        EXPECT_NEAR(radial.length(), 2.0f, 1e-4f);
    }
}

TEST(VizkitCircleLayer, BuilderRejectsMissingUnitAndInvalidSegments) {
    EXPECT_THROW(
        atlas::vizkit::CircleLayer<float>::builder().build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::vizkit::CircleLayer<float>::builder()
            .with_unit(make_vizkit_circle_unit<float>(
                atlas::Vector3<float>(0.0f, 0.0f, 0.0f),
                atlas::Vector3<float>(0.0f, 0.0f, 1.0f),
                1.0f))
            .with_segments(2)
            .build(),
        std::runtime_error);
}

#endif
