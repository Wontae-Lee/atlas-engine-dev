#include "config/simulation_config.h"
#include "session/session.h"
#include "transport/json_codec.h"
#include "rendering/layer/geometry_layer.h"
#include "view/geometry_render_view.h"

#include <atlas/math/vector/float3.h>
#include <atlas/sync/sync.h>

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>

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

TEST(InteractiveGeometryLayer, RendersLoadedMeshWithHingeTranslationAfterRestart) {
    struct MeshFile {
        std::filesystem::path path = std::filesystem::path(testing::TempDir()) / "atlas_render_mesh.obj";
        ~MeshFile() { std::filesystem::remove(path); }
    } file;
    {
        std::ofstream output(file.path);
        output << "v 1 2 0\nv 2 2 0\nv 1 3 0\nf 1 2 3\n";
    }
    auto config = atlas::interactive::JsonCodec::decode_simulation(R"({
        "dt":0.01,"fluid":{"buffer_size":4},
        "universe":{"lower_corner":[-4,-4,-4],"upper_corner":[4,4,4],"cell_size":1},
        "colliders":[{"unit":{
            "geometry":{"type":"triangle_mesh","file":"mesh.obj","translation":[-1,-2,0]},
            "translation":[1,2,0]
        }}]
    })");
    config.colliders[0].unit.geometry.file = file.path.string();
    atlas::interactive::Session session(std::move(config));
    for (int iteration = 0; iteration < 2; ++iteration) {
        const auto scene = session.scene_view();
        ASSERT_EQ(scene.geometries.size(), 1u);
        const auto& view = scene.geometries.front();
        ASSERT_NE(view.mesh, nullptr);
        EXPECT_TRUE(view.geometry->triangles.empty());
        const auto vertices = atlas::interactive::GeometryLayer::make_line_vertices(
            view, scene.lower_corner, scene.upper_corner);
        ASSERT_EQ(vertices.size(), 6u);
        EXPECT_FLOAT_EQ(vertices[0].x, 1.0f);
        EXPECT_FLOAT_EQ(vertices[0].y, 2.0f);
        EXPECT_FLOAT_EQ(vertices[1].x, 2.0f);
        EXPECT_FLOAT_EQ(vertices[1].y, 2.0f);
        if (iteration == 0) {
            std::filesystem::remove(file.path);
            session.restart();
        }
    }
}
