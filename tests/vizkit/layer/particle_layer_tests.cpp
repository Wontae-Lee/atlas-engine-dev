#include "../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#define private public
#include <vizkit/layer/particle/particle_layer.h>
#undef private

#include <gtest/gtest.h>

namespace {

template <typename T>
atlas::SystemHostPtr<T>
make_particle_layer_system(const std::size_t buffer_size) {
    return atlas::System<T>::builder()
        .with_fluid(atlas::system::Fluid<T>::builder().with_buffer_size(buffer_size).make_host_shared())
        .make_host_shared();
}

}

TEST(VizkitParticleLayer, BuilderStoresSystem) {
    const auto sim_system = make_particle_layer_system<float>(32);
    const auto color = atlas::Vector4<float>(0.9f, 0.4f, 0.2f, 0.8f);
    const auto layer = atlas::vizkit::ParticleLayer<float>::builder()
                           .with_system(sim_system)
                           .with_color(color)
                           .build();

    EXPECT_EQ(layer._system, sim_system);
    EXPECT_EQ(layer._capacity, 0u);
    EXPECT_FLOAT_EQ(layer._color.x, color.x);
    EXPECT_FLOAT_EQ(layer._color.y, color.y);
    EXPECT_FLOAT_EQ(layer._color.z, color.z);
    EXPECT_FLOAT_EQ(layer._color.w, color.w);
}

TEST(VizkitParticleLayer, BuilderRejectsNullSystem) {
    EXPECT_THROW(
        atlas::vizkit::ParticleLayer<float>::builder().build(),
        std::runtime_error);
}

#endif
