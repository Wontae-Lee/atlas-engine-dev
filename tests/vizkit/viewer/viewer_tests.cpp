#include "../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#define private public
#include <vizkit/viewer/viewer.h>
#undef private

#include <gtest/gtest.h>

TEST(VizkitViewer, BuilderBuildStoresConfiguredValues) {
    const auto viewer = atlas::vizkit::Viewer<float>::builder()
                            .with_dt(0.02f)
                            .with_size(1280, 720)
                            .with_title("Vizkit Test")
                            .with_fullscreen(true)
                            .build();

    EXPECT_FLOAT_EQ(viewer._dt, 0.02f);
    EXPECT_EQ(viewer._width, 1280);
    EXPECT_EQ(viewer._height, 720);
    EXPECT_STREQ(viewer._title, "Vizkit Test");
    EXPECT_TRUE(viewer._fullscreen);
}

TEST(VizkitViewer, BuilderRejectsInvalidParameters) {
    EXPECT_THROW(
        atlas::vizkit::Viewer<float>::builder()
            .with_dt(0.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::vizkit::Viewer<float>::builder()
            .with_size(-1, 720)
            .build(),
        std::runtime_error);
}

TEST(VizkitViewer, AddLayerAppendsLayerToInternalStorage) {
    atlas::vizkit::Viewer<float> viewer;
    auto layer = std::make_shared<atlas::test::DummyVizkitLayer<float>>();

    viewer.add_layer(layer);

    ASSERT_EQ(viewer._layers.size(), 1u);
    EXPECT_EQ(viewer._layers.front(), layer);
}

#endif
