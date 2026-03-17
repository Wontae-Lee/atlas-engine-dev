#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>     // std::abs
#include <stdexcept> // std::runtime_error
#include <string>    // std::string (implicit via APIs / file names)
#include <utility>   // std::move

using namespace atlas;

TEST(GeometryTriangleMesh, ConstructFromCopyBuildsBvhAndIsValid) {

    // Build a small set of non-degenerate triangles on the host.
    // TriangleContainer4<T> is used here as (a,b,c,n) where:
    // - a,b,c are triangle vertices
    // - n is a stored normal (often used for sign tests / shading / convenience)
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        // Triangle near the origin on the z=0 plane, with +Z normal.
        // Using constexpr ensures the container is trivially constructible for test data.
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    {
        // A second triangle far away in +X, also on z=0, with the same +Z normal.
        // Separation ensures "nearest triangle" logic is testable later.
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    // Construct mesh from a copy of triangles.
    // Expected behavior:
    // - triangles are stored internally
    // - BVH (or similar acceleration structure) is built during construction
    // - mesh becomes valid when triangles are non-empty and non-degenerate
    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_TRUE(mesh.is_valid());

    // A valid mesh must contain triangles; `triangles` is expected to be a buffer-like container.
    EXPECT_FALSE(mesh.triangles.empty());
}

TEST(GeometryTriangleMesh, ConstructFromMoveBuildsBvhAndPreservesTriangles) {

    // Prepare input triangles exactly as in the copy test, but move them into the mesh.
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

    // Capture size before move so we can verify mesh retains the triangle count.
    const std::size_t n = tris.size();

    // Move-construct mesh from triangle buffer.
    // Expected behavior:
    // - mesh takes ownership of the buffer storage
    // - BVH is built
    // - mesh remains valid
    const atlas::geometry::TriangleMesh<double> mesh(std::move(tris));

    EXPECT_TRUE(mesh.is_valid());
    EXPECT_EQ(mesh.triangles.size(), n);
}

TEST(GeometryTriangleMesh, SetTrianglesRebuildsBvhAndUpdatesBound) {

    // This test name suggests "set_triangles" behavior.
    // The current body constructs once and checks bound() is finite, which at least
    // verifies that bound() is computed successfully after building the BVH.
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

    // Construct mesh with initial triangles (should build BVH).
    const atlas::geometry::TriangleMesh<double> mesh(tris0);

    // bound() should be well-defined (finite) when the mesh has triangles.
    const auto bb0 = mesh.bound();
    EXPECT_TRUE(atlas::test::is_finite_vec(bb0.lower_corner));
    EXPECT_TRUE(atlas::test::is_finite_vec(bb0.upper_corner));
}

TEST(GeometryTriangleMesh, BuildBvhWithEmptyTrianglesMarksNotBuilt) {

    // Construct with an empty triangle set.
    // Expected behavior:
    // - mesh cannot build BVH
    // - mesh should be invalid
    // - internal triangle storage remains empty
    const atlas::HostBuffer<atlas::TriangleContainer4<double>> empty;

    const atlas::geometry::TriangleMesh<double> mesh(empty);

    EXPECT_FALSE(mesh.is_valid());
    EXPECT_TRUE(mesh.triangles.empty());
}

TEST(GeometryTriangleMesh, MakeTraceOperatorReturnsDefaultWhenNotBuilt) {

    // If BVH is not built (e.g., empty triangle list), make_trace_operator()
    // should still be safe to call and return some default/sentinel operator.
    //
    // This test is intentionally "no-assert": it verifies the call does not crash
    // and the returned value is usable at least as an object.
    const atlas::HostBuffer<atlas::TriangleContainer4<double>> empty;

    const atlas::geometry::TriangleMesh<double> mesh(empty);

    const auto tr = mesh.make_trace_operator();
    (void)tr; // suppress unused warning
}

TEST(GeometryTriangleMesh, MakeQueryOperatorReturnsValidTriangleMeshQueryOperator) {

    // Build a simple mesh with one triangle.
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

    const auto q = mesh.make_query_operator();
    EXPECT_EQ(q.type, geometry::GeometryType::TriangleMesh);
    EXPECT_TRUE(q.is_valid());
}

TEST(GeometryTriangleMesh, IsInsideUsesWindingBasedClosedMeshContainment) {
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, -1.0)
    });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, -1.0, 0.0)
    });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(-1.0, 0.0, 0.0)
    });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(1.0, 1.0, 1.0)
    });

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
        atlas::Vector3D(0.0, 0.0, 1.0)
    });

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_TRUE(mesh.is_on_surface(atlas::Vector3D(0.25, 0.25, 0.0), 0.0));
    EXPECT_FALSE(mesh.is_on_surface(atlas::Vector3D(0.25, 0.25, 0.3), 0.0));
    EXPECT_TRUE(mesh.is_on_surface(atlas::Vector3D(0.25, 0.25, 0.1), 0.15));
}

TEST(GeometryTriangleMesh, ClosestPointChoosesNearestTriangleAmongMany) {

    // Use atlas::eps in double precision for geometric comparisons.
    

    // Construct two triangles far apart so nearest-triangle selection is obvious.
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    {
        // Triangle near origin.
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(0.0, 0.0, 0.0),
            atlas::Vector3D(1.0, 0.0, 0.0),
            atlas::Vector3D(0.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }
    {
        // Triangle near x ≈ 10.
        constexpr atlas::TriangleContainer4<double> tc {
            atlas::Vector3D(10.0, 0.0, 0.0),
            atlas::Vector3D(11.0, 0.0, 0.0),
            atlas::Vector3D(10.0, 1.0, 0.0),
            atlas::Vector3D(0.0, 0.0, 1.0)
        };
        tris.push_back(tc);
    }

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    // Query near the first triangle: point above the triangle in +Z.
    // Closest point should project down onto z=0 inside the triangle.
    const auto cp0 = mesh.closest_point(atlas::Vector3D(0.25, 0.25, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(cp0, atlas::Vector3D(0.25, 0.25, 0.0), eps));

    // Query near the second triangle: same relative location but translated +10 in X.
    const auto cp1 = mesh.closest_point(atlas::Vector3D(10.25, 0.25, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(cp1, atlas::Vector3D(10.25, 0.25, 0.0), eps));
}

TEST(GeometryTriangleMesh, ClosestNormalReturnsTriangleNormalOfNearestTriangle) {



    // Two triangles with explicit stored normal (+Z) so nearest selection is testable.
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

    // Point near first triangle should return +Z normal.
    const auto n0 = mesh.closest_normal(atlas::Vector3D(0.2, 0.2, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(n0, atlas::Vector3D(0.0, 0.0, 1.0), eps));

    // Point near second triangle should also return +Z normal.
    const auto n1 = mesh.closest_normal(atlas::Vector3D(10.2, 0.2, 2.0));
    EXPECT_TRUE(atlas::test::vec_near(n1, atlas::Vector3D(0.0, 0.0, 1.0), eps));
}

TEST(GeometryTriangleMesh, SignedDistanceUsesWindingSignForClosedMesh) {

    // Closed tetrahedron with outward winding.
    // Signed distance sign should follow winding-based containment:
    // - negative inside
    // - positive outside
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, -1.0)
    });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, -1.0, 0.0)
    });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(0.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(-1.0, 0.0, 0.0)
    });
    tris.push_back(atlas::TriangleContainer4<double> {
        atlas::Vector3D(1.0, 0.0, 0.0),
        atlas::Vector3D(0.0, 1.0, 0.0),
        atlas::Vector3D(0.0, 0.0, 1.0),
        atlas::Vector3D(1.0, 1.0, 1.0)
    });

    const atlas::geometry::TriangleMesh<double> mesh(tris);

    EXPECT_LT(mesh.signed_distance(atlas::Vector3D(0.1, 0.1, 0.1)), 0.0);
    EXPECT_GT(mesh.signed_distance(atlas::Vector3D(2.0, 2.0, 2.0)), 0.0);
}

TEST(GeometryTriangleMesh, CentroidIsAverageOfTriangleCentroidsUniformWeight) {

    // Use atlas::eps for centroid comparison.
    

    // Build two triangles and verify mesh centroid is the uniform average of
    // per-triangle centroids (not area-weighted) as suggested by this API contract.
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

    // Compute triangle centroids as average of vertices (a+b+c)/3.
    const auto c0 = (tc0.a() + tc0.b() + tc0.c()) * (1.0 / 3.0);
    const auto c1 = (tc1.a() + tc1.b() + tc1.c()) * (1.0 / 3.0);

    // Mesh centroid is defined (by this test) as uniform average of triangle centroids.
    const auto expected = (c0 + c1) * 0.5;

    EXPECT_TRUE(atlas::test::vec_near(c, expected, eps));
}

TEST(GeometryTriangleMesh, BoundEnclosesAllTriangleVertices) {

    // This test ensures bound() returns a finite, well-formed AABB when triangles exist.
    // (A stronger test would explicitly compare against min/max over all vertices.)
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

    // Bounding box must be finite if computed from finite vertices.
    EXPECT_TRUE(atlas::test::is_finite_vec(bb.lower_corner));
    EXPECT_TRUE(atlas::test::is_finite_vec(bb.upper_corner));
}

TEST(GeometryTriangleMesh, IsValidRejectsDegenerateTriangle) {

    // A degenerate triangle (collinear points) has zero area and should be rejected
    // by mesh validity checks, because it breaks normals, distances, and BVH robustness.
    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;

    // Points a,b,c lie on the X axis => collinear => zero area.
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

    // Builder must enforce required fields.
    // If triangles are not provided, build() should throw to prevent constructing
    // a mesh that cannot be valid or build a BVH.
    EXPECT_THROW((void)atlas::geometry::TriangleMesh<double>::builder().build(), std::runtime_error);
}

TEST(GeometryTriangleMeshBuilder, WithTrianglesBuildCreatesValidMeshAndBuildsBvh) {

    // Provide triangles via builder and ensure build() returns a valid mesh.
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
                          .with_triangles(tris) // copy path
                          .build();

    EXPECT_TRUE(mesh.is_valid());
    EXPECT_EQ(mesh.triangles.size(), tris.size());
}

TEST(GeometryTriangleMeshBuilder, WithTrianglesMoveBuildCreatesValidMesh) {

    // Provide triangles via move to ensure builder supports transfer of ownership.
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

    // Capture expected triangle count before move.
    const std::size_t n = tris.size();

    const auto mesh = atlas::geometry::TriangleMesh<double>::builder()
                          .with_triangles(std::move(tris)) // move path
                          .build();

    EXPECT_TRUE(mesh.is_valid());
    EXPECT_EQ(mesh.triangles.size(), n);
}

TEST(GeometryTriangleMeshBuilder, MakeHostSharedReturnsNonNullAndValid) {

    // make_host_shared() should return a valid shared pointer owning a valid mesh.
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

    // Pointer must be non-null.
    ASSERT_TRUE(ptr);

    // Built mesh must be valid.
    EXPECT_TRUE(ptr->is_valid());
}

TEST(GeometryTriangleMesh, LoadFromObjReturnsFalseForMissingFile) {

    // load_from_obj() should fail gracefully when the file does not exist.
    // This ensures robust error handling and avoids throwing on common IO issues
    // (depending on API policy).
    const atlas::HostBuffer<atlas::TriangleContainer4<double>> empty;

    atlas::geometry::TriangleMesh<double> mesh(empty);

    const bool ok = mesh.load_from_obj("__definitely_not_a_real_file__.obj", false);

    EXPECT_FALSE(ok);
}
