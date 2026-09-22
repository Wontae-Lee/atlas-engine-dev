#include "config/simulation_config.h"
#include "rendering/layer/geometry_layer.h"
#include "view/geometry_render_view.h"

#include <atlas/math/vector/float3.h>
#include <atlas/sync/sync.h>

#include <gtest/gtest.h>

#include <array>

TEST(InteractiveGeometryLayer, DispatchesEveryGeometryTypeToLineVertices) {
    using Config = atlas::interactive::SimulationConfig;
    const std::array<Config::GeometryKind, 9> kinds = {
        Config::GeometryKind::box,
        Config::GeometryKind::circle,
        Config::GeometryKind::cylinder,
        Config::GeometryKind::plane,
        Config::GeometryKind::sphere,
        Config::GeometryKind::square,
        Config::GeometryKind::triangle,
        Config::GeometryKind::triangle_mesh,
        Config::GeometryKind::polygonal_prism
    };

    for (const Config::GeometryKind kind : kinds) {
        Config::Geometry geometry;
        geometry.kind = kind;
        atlas::interactive::GeometryRenderView view;
        view.geometry = &geometry;
        if (kind == Config::GeometryKind::triangle_mesh) {
            geometry.triangles = {
                std::array<Config::Vec3, 3> {
                    Config::Vec3 { 0.0f, 0.0f, 0.0f },
                    Config::Vec3 { 1.0f, 0.0f, 0.0f },
                    Config::Vec3 { 0.0f, 1.0f, 0.0f }
                }
            };
        }

        const auto vertices = atlas::interactive::GeometryLayer::make_line_vertices(
            view, atlas::Float3(-2.0f), atlas::Float3(2.0f));
        EXPECT_FALSE(vertices.empty()) << static_cast<int>(kind);
        EXPECT_EQ(vertices.size() % 2, 0u) << static_cast<int>(kind);
    }
}

TEST(InteractiveGeometryLayer, AppliesTheCurrentUnitPose) {
    atlas::interactive::SimulationConfig::Geometry geometry;
    geometry.kind = atlas::interactive::SimulationConfig::GeometryKind::triangle;
    geometry.a = { 0.0f, 0.0f, 0.0f };
    geometry.b = { 1.0f, 0.0f, 0.0f };
    geometry.c = { 0.0f, 1.0f, 0.0f };
    atlas::interactive::GeometryRenderView view;
    view.geometry = &geometry;
    view.sync.translation = atlas::Float3(3.0f, 4.0f, 5.0f);

    const auto vertices = atlas::interactive::GeometryLayer::make_line_vertices(
        view, atlas::Float3(-2.0f), atlas::Float3(2.0f));
    ASSERT_FALSE(vertices.empty());
    EXPECT_FLOAT_EQ(vertices.front().x, 3.0f);
    EXPECT_FLOAT_EQ(vertices.front().y, 4.0f);
    EXPECT_FLOAT_EQ(vertices.front().z, 5.0f);
}
