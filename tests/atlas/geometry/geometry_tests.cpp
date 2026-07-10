#include <atlas/geometry/geometry.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

using atlas::AABB;
using atlas::Box;
using atlas::Circle;
using atlas::ConceptGeometry;
using atlas::Cylinder;
using atlas::Geometry;
using atlas::GeometryCentroid;
using atlas::GeometryIsValid;
using atlas::GeometryType;
using atlas::GeometryVariant;
using atlas::HitSurface;
using atlas::Plane;
using atlas::Ray;
using atlas::Sphere;
using atlas::Square;
using atlas::Triangle;
using atlas::TriangleMeshView;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

/// A discriminant value that matches no registered leaf, to drive the visit
/// fallback path of an "unknown" active tag.
constexpr GeometryType bogus_tag = static_cast<GeometryType>(999);

} // namespace

// The umbrella must remain a plain value so it can live in DeviceBuffer<Geometry>
// (the header explicitly claims this).
static_assert(std::is_trivially_copyable_v<Geometry>,
              "Geometry must be trivially copyable for device buffers");

// Every registered leaf models the shared read-only query contract.
static_assert(ConceptGeometry<Box>);
static_assert(ConceptGeometry<Circle>);
static_assert(ConceptGeometry<Cylinder>);
static_assert(ConceptGeometry<Plane>);
static_assert(ConceptGeometry<Sphere>);
static_assert(ConceptGeometry<Square>);
static_assert(ConceptGeometry<Triangle>);
static_assert(ConceptGeometry<TriangleMeshView>);

TEST(Geometry, DefaultConstructsUnitSphere) {
    const Geometry geometry {};

    EXPECT_EQ(geometry.type, GeometryType::sphere);
    EXPECT_TRUE(geometry.is_valid());

    // The default leaf is the unit sphere at the origin.
    const Sphere reference {};
    expect_vec_near(geometry.centroid(), reference.centroid());
    EXPECT_FLOAT_EQ(geometry.signed_distance(Float3(3.0f, 0.0f, 0.0f)),
                    reference.signed_distance(Float3(3.0f, 0.0f, 0.0f)));
}

TEST(Geometry, ConstructFromEachLeafSetsMatchingTag) {
    EXPECT_EQ(Geometry(Box()).type, GeometryType::box);
    EXPECT_EQ(Geometry(Circle()).type, GeometryType::circle);
    EXPECT_EQ(Geometry(Cylinder()).type, GeometryType::cylinder);
    EXPECT_EQ(Geometry(Plane()).type, GeometryType::plane);
    EXPECT_EQ(Geometry(Sphere()).type, GeometryType::sphere);
    EXPECT_EQ(Geometry(Square()).type, GeometryType::square);
    EXPECT_EQ(Geometry(Triangle()).type, GeometryType::triangle);

    // The triangle-mesh case stores the trivially-copyable view, not an owning mesh.
    EXPECT_EQ(Geometry(TriangleMeshView {}).type, GeometryType::triangle_mesh);
}

TEST(Geometry, ForwardsClosestQueriesToActiveLeaf) {
    const Sphere sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f);
    const Geometry geometry(sphere);

    const Float3 query(3.0f, 0.0f, 0.0f);

    // The umbrella answer must equal the leaf's own answer, i.e. the unit sphere.
    expect_vec_near(geometry.closest_point(query), sphere.closest_point(query));
    expect_vec_near(geometry.closest_point(query), Float3(1.0f, 0.0f, 0.0f));

    expect_vec_near(geometry.closest_normal(query), sphere.closest_normal(query));
    expect_vec_near(geometry.closest_normal(query), Float3(1.0f, 0.0f, 0.0f));
}

TEST(Geometry, ForwardsSignedDistanceToActiveLeaf) {
    const Sphere sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f);
    const Geometry geometry(sphere);

    const Float3 query(3.0f, 0.0f, 0.0f);

    EXPECT_FLOAT_EQ(geometry.signed_distance(query), sphere.signed_distance(query));
    EXPECT_FLOAT_EQ(geometry.signed_distance(query), 2.0f);
}

TEST(Geometry, ForwardsClassificationQueriesToActiveLeaf) {
    const Sphere sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f);
    const Geometry geometry(sphere);

    EXPECT_EQ(geometry.is_inside(Float3(0.0f, 0.0f, 0.0f)),
              sphere.is_inside(Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(geometry.is_inside(Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_FALSE(geometry.is_inside(Float3(3.0f, 0.0f, 0.0f)));

    EXPECT_EQ(geometry.is_on_surface(Float3(1.0f, 0.0f, 0.0f)),
              sphere.is_on_surface(Float3(1.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(geometry.is_on_surface(Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_EQ(geometry.is_valid(), sphere.is_valid());
    EXPECT_TRUE(geometry.is_valid());
}

TEST(Geometry, ForwardsCentroidAndBoundToActiveLeaf) {
    const Sphere sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f);
    const Geometry geometry(sphere);

    expect_vec_near(geometry.centroid(), sphere.centroid());
    expect_vec_near(geometry.centroid(), Float3(0.0f, 0.0f, 0.0f));

    const AABB bound     = geometry.bound();
    const AABB reference = sphere.bound();
    expect_vec_near(bound.lower_corner, reference.lower_corner);
    expect_vec_near(bound.upper_corner, reference.upper_corner);
    expect_vec_near(bound.lower_corner, Float3(-1.0f, -1.0f, -1.0f));
    expect_vec_near(bound.upper_corner, Float3(1.0f, 1.0f, 1.0f));
}

TEST(Geometry, ForwardsTraceToActiveLeaf) {
    const Sphere sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f);
    const Geometry geometry(sphere);

    const Ray ray(Float3(3.0f, 0.0f, 0.0f), Float3(-1.0f, 0.0f, 0.0f));

    const HitSurface hit           = geometry.trace(ray);
    const HitSurface reference_hit = sphere.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_EQ(hit.is_intersecting, reference_hit.is_intersecting);
    EXPECT_FLOAT_EQ(hit.distance, reference_hit.distance);
    EXPECT_FLOAT_EQ(hit.distance, 2.0f);
    expect_vec_near(hit.point, Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(1.0f, 0.0f, 0.0f));
}

TEST(Geometry, DispatchesToTheLeafSelectedByTag) {
    // A box-tagged umbrella must answer with box geometry, not sphere geometry.
    const Box box(Float3(-2.0f, -2.0f, -2.0f), Float3(2.0f, 2.0f, 2.0f));
    const Geometry geometry(box);

    EXPECT_EQ(geometry.type, GeometryType::box);
    expect_vec_near(geometry.closest_point(Float3(5.0f, 0.0f, 0.0f)),
                    box.closest_point(Float3(5.0f, 0.0f, 0.0f)));

    const AABB bound = geometry.bound();
    expect_vec_near(bound.lower_corner, Float3(-2.0f, -2.0f, -2.0f));
    expect_vec_near(bound.upper_corner, Float3(2.0f, 2.0f, 2.0f));
}

TEST(Geometry, CopyConstructPreservesActiveLeaf) {
    // Braces, not parentheses: `Geometry original(Box())` declares a function.
    const Geometry original(Box {});
    const Geometry copy = original;

    EXPECT_EQ(copy.type, GeometryType::box);

    const Box reference {};
    expect_vec_near(copy.centroid(), reference.centroid());
}

TEST(Geometry, CopyAssignChangesActiveLeaf) {
    Geometry geometry(Sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f));
    ASSERT_EQ(geometry.type, GeometryType::sphere);

    const Geometry source(Plane(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f)));
    geometry = source;

    EXPECT_EQ(geometry.type, GeometryType::plane);

    // After the reassignment the plane's queries drive the umbrella.
    const Plane reference(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(geometry.signed_distance(Float3(0.0f, 0.0f, 2.0f)),
                    reference.signed_distance(Float3(0.0f, 0.0f, 2.0f)));
    expect_vec_near(geometry.closest_normal(Float3(0.0f, 0.0f, 2.0f)),
                    Float3(0.0f, 0.0f, 1.0f));
}

TEST(Geometry, VariantVisitDispatchesToActiveLeaf) {
    const Geometry geometry(Sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f));

    // The public visitor entry point routes to the live leaf just as the member
    // functions do.
    EXPECT_TRUE(GeometryVariant::visit(geometry, GeometryIsValid {}, false));
    expect_vec_near(GeometryVariant::visit(geometry, GeometryCentroid {}, Float3(9.0f, 9.0f, 9.0f)),
                    Float3(0.0f, 0.0f, 0.0f));
}

TEST(Geometry, UnknownActiveTagReturnsQueryFallbacks) {
    Geometry geometry(Sphere(Float3(0.0f, 0.0f, 0.0f), 1.0f));
    geometry.type = bogus_tag; // simulate a corrupt discriminant

    const Float3 query(3.0f, 0.0f, 0.0f);

    // With no leaf matching the tag, every query returns its documented fallback.
    expect_vec_near(geometry.closest_point(query), query);
    expect_vec_near(geometry.closest_normal(query), Float3(0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(std::isinf(geometry.signed_distance(query)));
    EXPECT_FALSE(geometry.is_inside(query));
    EXPECT_FALSE(geometry.is_on_surface(query));
    EXPECT_FALSE(geometry.is_valid());
    expect_vec_near(geometry.centroid(), Float3(0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(geometry.trace(Ray(query, Float3(-1.0f, 0.0f, 0.0f))).is_intersecting);
}
