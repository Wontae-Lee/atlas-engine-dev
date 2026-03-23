#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

using namespace atlas;

TEST(GeometryTriangleMesh, ConstructFromCopyBuildsBvhAndIsValid) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {

        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    {

        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_TRUE(mesh.is_valid());

    EXPECT_FALSE(mesh.triangles.empty());
}

TEST(GeometryTriangleMesh, ConstructFromMoveBuildsBvhAndPreservesTriangles) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const std::size_t n = tris.size();

    const atlas::geometry::TriangleMesh<double> mesh(std::move(tris));

    EXPECT_TRUE(mesh.is_valid());
    EXPECT_EQ(mesh.triangles.size(), n);
}

TEST(GeometryTriangleMesh, SetTrianglesRebuildsBvhAndUpdatesBound) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris0;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris0.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris0);

    const auto bb0 = mesh.bound();
    EXPECT_TRUE(atlas::test::is_finite_vec(bb0.lower_corner));
    EXPECT_TRUE(atlas::test::is_finite_vec(bb0.upper_corner));
}

TEST(GeometryTriangleMesh, BuildBvhWithEmptyTrianglesMarksNotBuilt) {

    const atlas::HostBuffer<atlas::TriangleContainer4<double>> empty;

    const atlas::geometry::TriangleMesh<double> mesh(empty);

    EXPECT_FALSE(mesh.is_valid());
    EXPECT_TRUE(mesh.triangles.empty());
}

TEST(GeometryTriangleMesh, MakeGeometryOperatorReturnsDefaultWhenNotBuilt) {

    const atlas::HostBuffer<atlas::TriangleContainer4<double>> empty;

    const atlas::geometry::TriangleMesh<double> mesh(empty);

    const auto tr = mesh.make_geometry_operator();
    (void)tr;
}

TEST(GeometryTriangleMesh, MakeGeometryOperatorReturnsValidTriangleMeshGeometryOperator) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    const auto q = mesh.make_geometry_operator();
    EXPECT_EQ(q.type, geometry::GeometryType::TriangleMesh);
    EXPECT_TRUE(q.is_valid());
}

TEST(GeometryTriangleMesh, IsInsideUsesWindingBasedClosedMeshContainment) {
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, -1.0) });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, -1.0, 0.0) });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(-1.0, 0.0, 0.0) });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(1.0, 1.0, 1.0) });

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_TRUE(mesh.is_inside(atlas::Vector3D(0.1, 0.1, 0.1), 0.0));
    EXPECT_FALSE(mesh.is_inside(atlas::Vector3D(2.0, 2.0, 2.0), 0.0));
    EXPECT_TRUE(mesh.is_inside(atlas::Vector3D(0.6, 0.6, 0.1), 0.5));
}

TEST(GeometryTriangleMesh, IsOnSurfaceDetectsSurfaceBand) {
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0) });

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_TRUE(mesh.is_on_surface(atlas::Vector3D(0.25, 0.25, 0.0), 0.0));
    EXPECT_FALSE(mesh.is_on_surface(atlas::Vector3D(0.25, 0.25, 0.3), 0.0));
    EXPECT_TRUE(mesh.is_on_surface(atlas::Vector3D(0.25, 0.25, 0.1), 0.15));
}

TEST(GeometryTriangleMesh, ClosestPointChoosesNearestTriangleAmongMany) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {

        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {

        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    const auto cp0 = mesh.closest_point(atlas::Vector3D(0.25, 0.25, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(cp0, atlas::Vector3D(0.25, 0.25, 0.0), eps));

    const auto cp1 = mesh.closest_point(atlas::Vector3D(10.25, 0.25, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(cp1, atlas::Vector3D(10.25, 0.25, 0.0), eps));
}

TEST(GeometryTriangleMesh, ClosestNormalReturnsTriangleNormalOfNearestTriangle) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    const auto n0 = mesh.closest_normal(atlas::Vector3D(0.2, 0.2, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(n0, atlas::Vector3D(0.0, 0.0, 1.0), eps));

    const auto n1 = mesh.closest_normal(atlas::Vector3D(10.2, 0.2, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(n1, atlas::Vector3D(0.0, 0.0, 1.0), eps));
}

TEST(GeometryTriangleMesh, SignedDistanceUsesWindingSignForClosedMesh) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, -1.0) });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, -1.0, 0.0) });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(-1.0, 0.0, 0.0) });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(1.0, 1.0, 1.0) });

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_LT(mesh.signed_distance(atlas::Vector3D(0.1, 0.1, 0.1)), 0.0);
    EXPECT_GT(mesh.signed_distance(atlas::Vector3D(2.0, 2.0, 2.0)), 0.0);
}

TEST(GeometryTriangleMesh, CentroidIsAverageOfTriangleCentroidsUniformWeight) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    atlas::TriangleContainer4<double> tc0 {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0)
    };
    tris.push_back(tc0);

    atlas::TriangleContainer4<double> tc1 {
        atlas::Vector3D(10.0, 0.0, 0.0),
        atlas::Vector3D(11.0, 0.0, 0.0),
        atlas::Vector3D(10.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0)
    };
    tris.push_back(tc1);

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    const auto c = mesh.centroid();

    const auto c0 = (tc0.a() + tc0.b() + tc0.c()) * (1.0 / 3.0);
    const auto c1 = (tc1.a() + tc1.b() + tc1.c()) * (1.0 / 3.0);

    const auto expected = (c0 + c1) * 0.5;

    EXPECT_TRUE(atlas::test::vec_near(c, expected, eps));
}

TEST(GeometryTriangleMesh, BoundEnclosesAllTriangleVertices) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    const auto bb = mesh.bound();

    EXPECT_TRUE(atlas::test::is_finite_vec(bb.lower_corner));
    EXPECT_TRUE(atlas::test::is_finite_vec(bb.upper_corner));
}

TEST(GeometryTriangleMesh, IsValidRejectsDegenerateTriangle) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    constexpr atlas::TriangleContainer4<double> tc {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(2.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0)
    };
    tris.push_back(tc);

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_FALSE(mesh.is_valid());
}

TEST(GeometryTriangleMeshBuilder, BuildThrowsIfTrianglesNotSet) {

    EXPECT_THROW((void)atlas::geometry::TriangleMesh<double>::builder().build(), std::runtime_error);
}

TEST(GeometryTriangleMeshBuilder, WithTrianglesBuildCreatesValidMeshAndBuildsBvh) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const auto mesh = atlas::geometry::TriangleMesh<double>::builder()
                          .with_triangles(tris)
                          .build();

    EXPECT_TRUE(mesh.is_valid());
    EXPECT_EQ(mesh.triangles.size(), tris.size());
}

TEST(GeometryTriangleMeshBuilder, WithTrianglesMoveBuildCreatesValidMesh) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const std::size_t n = tris.size();

    const auto mesh = atlas::geometry::TriangleMesh<double>::builder()
                          .with_triangles(std::move(tris))
                          .build();

    EXPECT_TRUE(mesh.is_valid());
    EXPECT_EQ(mesh.triangles.size(), n);
}

TEST(GeometryTriangleMeshBuilder, MakeHostSharedReturnsNonNullAndValid) {

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const auto ptr = atlas::geometry::TriangleMesh<double>::builder()
                         .with_triangles(tris)
                         .make_host_shared();

    ASSERT_TRUE(ptr);

    EXPECT_TRUE(ptr->is_valid());
}

TEST(GeometryTriangleMesh, LoadFromObjReturnsFalseForMissingFile) {

    const atlas::HostBuffer<atlas::TriangleContainer4<double>> empty;

    atlas::geometry::TriangleMesh<double> mesh(empty);

    const bool ok = mesh.load_from_obj("__definitely_not_a_real_file__.obj", false);

    EXPECT_FALSE(ok);
}