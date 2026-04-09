#include "../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#define private public
#include <vizkit/viewer/viewer.h>
#undef private

#include <gtest/gtest.h>

namespace {

template <typename T>
atlas::FluidHostPtr<T>
make_buffered_fluid(const std::size_t buffer_size) {
    return atlas::system::Fluid<T>::builder()
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

}

TEST(VizkitViewer, BuilderBuildStoresConfiguredValues) {
    const auto sim_system = atlas::System<float>::builder()
                                .with_fluid(make_buffered_fluid<float>(4))
                                .with_dt(0.02f)
                                .make_host_shared();
    const auto viewer = atlas::vizkit::Viewer<float>::builder()
                            .with_system(sim_system)
                            .with_size(1280, 720)
                            .with_title("Vizkit Test")
                            .with_fullscreen(true)
                            .build();

    EXPECT_EQ(viewer._system, sim_system);
    EXPECT_FLOAT_EQ(viewer._system->dt(), 0.02f);
    EXPECT_EQ(viewer._width, 1280);
    EXPECT_EQ(viewer._height, 720);
    EXPECT_STREQ(viewer._title, "Vizkit Test");
    EXPECT_TRUE(viewer._fullscreen);
}

TEST(VizkitViewer, BuilderRejectsInvalidParameters) {
    EXPECT_THROW(
        atlas::vizkit::Viewer<float>::builder()
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::vizkit::Viewer<float>::builder()
            .with_system(atlas::System<float>::builder().with_fluid(make_buffered_fluid<float>(1)).make_host_shared())
            .with_size(-1, 720)
            .build(),
        std::runtime_error);
}

TEST(VizkitViewer, AddLayerAppendsLayerToInternalStorage) {
    auto viewer = atlas::vizkit::Viewer<float>::builder()
                      .with_system(atlas::System<float>::builder().with_fluid(make_buffered_fluid<float>(1)).make_host_shared())
                      .build();
    auto layer = std::make_shared<atlas::test::DummyVizkitLayer<float>>();

    viewer.add_layer(layer);

    ASSERT_EQ(viewer._layers.size(), 1u);
    EXPECT_EQ(viewer._layers.front(), layer);
}

#endif
