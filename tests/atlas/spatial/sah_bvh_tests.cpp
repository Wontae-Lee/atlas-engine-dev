#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/geometry/triangle.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

using atlas::AABB;
using atlas::BVHNode;
using atlas::Float3;
using atlas::HitSurface;
using atlas::HostBuffer;
using atlas::Ray;
using atlas::SAHBVH;
using atlas::Triangle;
using atlas::TriangleContainer4;

/** @brief Nearest-hit summary shared by the tree traversal and the brute-force loop. */
struct NearestHit {
    bool hit   = false;
    float dist = 0.0f;
    int prim   = -1;
};

/** @brief Packs three vertices (plus a +z placeholder normal) into a primitive. */
TriangleContainer4
make_tri(const Float3& a, const Float3& b, const Float3& c) {
    return TriangleContainer4(a, b, c, Float3(0.0f, 0.0f, 1.0f));
}

/** @brief Eight small, well-separated triangles on a 2x2x2 grid; centroids all distinct. */
HostBuffer<TriangleContainer4>
grid_triangles() {
    HostBuffer<TriangleContainer4> tris;

    for (int i = 0; i < 8; ++i) {
        const float gx = static_cast<float>((i >> 0) & 1) * 4.0f;
        const float gy = static_cast<float>((i >> 1) & 1) * 4.0f;
        const float gz = static_cast<float>((i >> 2) & 1) * 4.0f;
        const Float3 base(gx, gy, gz);

        tris.push_back(make_tri(base,
                                base + Float3(1.0f, 0.0f, 0.0f),
                                base + Float3(0.0f, 1.0f, 0.0f)));
    }

    return tris;
}

/** @brief Brute-force union of every per-primitive bound. */
AABB
brute_union(const SAHBVH& bvh) {
    AABB u;

    for (std::size_t i = 0; i < bvh.bounds().size(); ++i) {
        u.merge(bvh.bounds()[i]);
    }

    return u;
}

void
expect_contains(const AABB& parent, const AABB& child) {
    constexpr float slack = 1e-4f;

    EXPECT_LE(parent.lower_corner.x, child.lower_corner.x + slack);
    EXPECT_LE(parent.lower_corner.y, child.lower_corner.y + slack);
    EXPECT_LE(parent.lower_corner.z, child.lower_corner.z + slack);
    EXPECT_GE(parent.upper_corner.x, child.upper_corner.x - slack);
    EXPECT_GE(parent.upper_corner.y, child.upper_corner.y - slack);
    EXPECT_GE(parent.upper_corner.z, child.upper_corner.z - slack);
}

/**
 * @brief Recursively walks the built tree asserting every structural invariant.
 *
 * Proves the tree is acyclic and singly-reachable, that child links are in range,
 * that each interior node encloses both children, and collects the leaf primitives.
 */
void
validate_subtree(const HostBuffer<BVHNode>& nodes,
                 const HostBuffer<int>& indices,
                 const int idx,
                 std::vector<char>& visited,
                 std::vector<int>& leaf_prims) {
    const int node_count = static_cast<int>(nodes.size());

    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, node_count);
    ASSERT_FALSE(visited[idx]) << "node " << idx << " reached more than once";
    visited[idx] = 1;

    const BVHNode& node = nodes[idx];

    if (node.is_leaf) {
        EXPECT_GE(node.count, 1);
        EXPECT_GE(node.start, 0);
        EXPECT_LE(node.start + node.count, static_cast<int>(indices.size()));

        for (int k = node.start; k < node.start + node.count; ++k) {
            leaf_prims.push_back(indices[k]);
        }

        return;
    }

    ASSERT_GE(node.left, 0);
    ASSERT_LT(node.left, node_count);
    ASSERT_GE(node.right, 0);
    ASSERT_LT(node.right, node_count);

    validate_subtree(nodes, indices, node.left, visited, leaf_prims);
    validate_subtree(nodes, indices, node.right, visited, leaf_prims);

    expect_contains(node.bounds, nodes[node.left].bounds);
    expect_contains(node.bounds, nodes[node.right].bounds);
}

/** @brief Host-side ray/BVH traversal over the built host node array (no device code). */
void
traverse(const HostBuffer<BVHNode>& nodes,
         const HostBuffer<int>& indices,
         const HostBuffer<TriangleContainer4>& tris,
         const int idx,
         const Ray& ray,
         NearestHit& best) {
    if (idx < 0) {
        return;
    }

    const BVHNode& node = nodes[idx];

    if (!node.bounds.intersects(ray)) {
        return;
    }

    if (node.is_leaf) {
        for (int k = node.start; k < node.start + node.count; ++k) {
            const int pid            = indices[k];
            const Triangle tri(tris[pid].a(), tris[pid].b(), tris[pid].c());
            const HitSurface surface = tri.trace(ray);

            if (surface.is_intersecting && (!best.hit || surface.distance < best.dist)) {
                best.hit  = true;
                best.dist = surface.distance;
                best.prim = pid;
            }
        }

        return;
    }

    traverse(nodes, indices, tris, node.left, ray, best);
    traverse(nodes, indices, tris, node.right, ray, best);
}

/** @brief Nearest forward hit found by testing every primitive directly. */
NearestHit
brute_force(const HostBuffer<TriangleContainer4>& tris, const Ray& ray) {
    NearestHit best;

    for (int i = 0; i < static_cast<int>(tris.size()); ++i) {
        const Triangle tri(tris[i].a(), tris[i].b(), tris[i].c());
        const HitSurface surface = tri.trace(ray);

        if (surface.is_intersecting && (!best.hit || surface.distance < best.dist)) {
            best.hit  = true;
            best.dist = surface.distance;
            best.prim = i;
        }
    }

    return best;
}

}

TEST(SAHBVH, BuildOverEmptySetLeavesHierarchyEmpty) {
    SAHBVH bvh;
    bvh.build(HostBuffer<TriangleContainer4> {});

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
}

TEST(SAHBVH, BuildOverSinglePrimitiveRootIsLeafCoveringIt) {
    HostBuffer<TriangleContainer4> tris;
    tris.push_back(make_tri(Float3(0.0f, 0.0f, 0.0f),
                            Float3(1.0f, 0.0f, 0.0f),
                            Float3(0.0f, 1.0f, 0.0f)));

    SAHBVH bvh;
    bvh.build(tris);

    ASSERT_GE(bvh.root(), 0);
    const BVHNode& root = bvh.nodes()[bvh.root()];

    EXPECT_TRUE(root.is_leaf);
    EXPECT_EQ(root.count, 1);
    ASSERT_EQ(bvh.indices().size(), 1u);
    EXPECT_EQ(bvh.indices()[0], 0);

    const AABB expected = bvh.bounds()[0];
    EXPECT_FLOAT_EQ(root.bounds.lower_corner.x, expected.lower_corner.x);
    EXPECT_FLOAT_EQ(root.bounds.upper_corner.y, expected.upper_corner.y);
}

TEST(SAHBVH, RootBoundEqualsUnionOfAllPrimitiveBounds) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    SAHBVH bvh;
    bvh.set_leaf_size(1);
    bvh.build(tris);

    ASSERT_GE(bvh.root(), 0);
    const AABB expected   = brute_union(bvh);
    const AABB root_bound = bvh.nodes()[bvh.root()].bounds;

    EXPECT_FLOAT_EQ(root_bound.lower_corner.x, expected.lower_corner.x);
    EXPECT_FLOAT_EQ(root_bound.lower_corner.y, expected.lower_corner.y);
    EXPECT_FLOAT_EQ(root_bound.lower_corner.z, expected.lower_corner.z);
    EXPECT_FLOAT_EQ(root_bound.upper_corner.x, expected.upper_corner.x);
    EXPECT_FLOAT_EQ(root_bound.upper_corner.y, expected.upper_corner.y);
    EXPECT_FLOAT_EQ(root_bound.upper_corner.z, expected.upper_corner.z);
}

TEST(SAHBVH, TreeStructureIsAcyclicNestedAndAPermutation) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    SAHBVH bvh;
    bvh.set_leaf_size(1);
    bvh.build(tris);

    ASSERT_GE(bvh.root(), 0);

    std::vector<char> visited(bvh.nodes().size(), 0);
    std::vector<int> leaf_prims;
    validate_subtree(bvh.nodes(), bvh.indices(), bvh.root(), visited, leaf_prims);

    std::sort(leaf_prims.begin(), leaf_prims.end());
    ASSERT_EQ(leaf_prims.size(), tris.size());

    for (int i = 0; i < static_cast<int>(leaf_prims.size()); ++i) {
        EXPECT_EQ(leaf_prims[i], i);
    }
}

TEST(SAHBVH, EveryLeafRespectsLeafSizeOnASeparableSet) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    SAHBVH bvh;
    bvh.set_leaf_size(2);
    bvh.build(tris);

    // With distinct, separable centroids the recursion never falls back to an
    // over-full leaf, so every leaf honours the configured capacity.
    for (std::size_t i = 0; i < bvh.nodes().size(); ++i) {
        if (bvh.nodes()[i].is_leaf) {
            EXPECT_LE(bvh.nodes()[i].count, bvh.leaf_size());
            EXPECT_GE(bvh.nodes()[i].count, 1);
        }
    }
}

TEST(SAHBVH, LargerLeafSizeYieldsFewerNodes) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    SAHBVH fine;
    fine.set_leaf_size(1);
    fine.build(tris);

    SAHBVH coarse;
    coarse.set_leaf_size(8);
    coarse.build(tris);

    // leaf_size >= n collapses the whole set into a single leaf node.
    EXPECT_EQ(coarse.nodes().size(), 1u);
    EXPECT_GT(fine.nodes().size(), coarse.nodes().size());
}

TEST(SAHBVH, BinCountValueStillYieldsAValidHierarchy) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    for (const int bins : { 4, 16, 128 }) {
        SAHBVH bvh;
        bvh.set_leaf_size(1);
        bvh.set_bin_count(bins);
        bvh.build(tris);

        ASSERT_GE(bvh.root(), 0);

        // Correctness must survive every bin resolution: root union and a clean
        // permutation of all primitives.
        const AABB expected   = brute_union(bvh);
        const AABB root_bound = bvh.nodes()[bvh.root()].bounds;
        EXPECT_FLOAT_EQ(root_bound.lower_corner.x, expected.lower_corner.x);
        EXPECT_FLOAT_EQ(root_bound.upper_corner.z, expected.upper_corner.z);

        std::vector<char> visited(bvh.nodes().size(), 0);
        std::vector<int> leaf_prims;
        validate_subtree(bvh.nodes(), bvh.indices(), bvh.root(), visited, leaf_prims);

        std::sort(leaf_prims.begin(), leaf_prims.end());
        ASSERT_EQ(leaf_prims.size(), tris.size());
        for (int i = 0; i < static_cast<int>(leaf_prims.size()); ++i) {
            EXPECT_EQ(leaf_prims[i], i);
        }
    }
}

TEST(SAHBVH, TraversalHitsTheKnownPrimitiveAndMatchesBruteForce) {
    HostBuffer<TriangleContainer4> tris;

    for (int i = 0; i < 6; ++i) {
        const float x = static_cast<float>(i) * 10.0f;
        tris.push_back(make_tri(Float3(x - 1.0f, -1.0f, 0.0f),
                                Float3(x + 1.0f, -1.0f, 0.0f),
                                Float3(x, 1.0f, 0.0f)));
    }

    SAHBVH bvh;
    bvh.set_leaf_size(1);
    bvh.build(tris);
    ASSERT_GE(bvh.root(), 0);

    const Ray ray(Float3(30.0f, 0.0f, -5.0f), Float3(0.0f, 0.0f, 1.0f));

    NearestHit tree;
    traverse(bvh.nodes(), bvh.indices(), tris, bvh.root(), ray, tree);
    const NearestHit brute = brute_force(tris, ray);

    ASSERT_TRUE(tree.hit);
    EXPECT_EQ(tree.prim, 3);
    EXPECT_EQ(tree.prim, brute.prim);
    EXPECT_FLOAT_EQ(tree.dist, brute.dist);
}

TEST(SAHBVH, TraversalReturnsNoHitForAMissAndMatchesBruteForce) {
    HostBuffer<TriangleContainer4> tris;

    for (int i = 0; i < 6; ++i) {
        const float x = static_cast<float>(i) * 10.0f;
        tris.push_back(make_tri(Float3(x - 1.0f, -1.0f, 0.0f),
                                Float3(x + 1.0f, -1.0f, 0.0f),
                                Float3(x, 1.0f, 0.0f)));
    }

    SAHBVH bvh;
    bvh.set_leaf_size(1);
    bvh.build(tris);
    ASSERT_GE(bvh.root(), 0);

    const Ray ray(Float3(1000.0f, 0.0f, -5.0f), Float3(0.0f, 0.0f, 1.0f));

    NearestHit tree;
    traverse(bvh.nodes(), bvh.indices(), tris, bvh.root(), ray, tree);
    const NearestHit brute = brute_force(tris, ray);

    EXPECT_FALSE(tree.hit);
    EXPECT_EQ(tree.hit, brute.hit);
}

TEST(SAHBVH, TraversalReturnsTheNearestOfSeveralHits) {
    HostBuffer<TriangleContainer4> tris;
    tris.push_back(make_tri(Float3(-1.0f, -1.0f, 0.0f),
                            Float3(1.0f, -1.0f, 0.0f),
                            Float3(0.0f, 1.0f, 0.0f)));  // prim 0, nearer
    tris.push_back(make_tri(Float3(-1.0f, -1.0f, 5.0f),
                            Float3(1.0f, -1.0f, 5.0f),
                            Float3(0.0f, 1.0f, 5.0f)));   // prim 1, farther
    tris.push_back(make_tri(Float3(19.0f, -1.0f, 0.0f),
                            Float3(21.0f, -1.0f, 0.0f),
                            Float3(20.0f, 1.0f, 0.0f)));  // prim 2, decoy
    tris.push_back(make_tri(Float3(39.0f, -1.0f, 0.0f),
                            Float3(41.0f, -1.0f, 0.0f),
                            Float3(40.0f, 1.0f, 0.0f)));  // prim 3, decoy

    SAHBVH bvh;
    bvh.set_leaf_size(1);
    bvh.build(tris);
    ASSERT_GE(bvh.root(), 0);

    const Ray ray(Float3(0.0f, 0.0f, -5.0f), Float3(0.0f, 0.0f, 1.0f));

    NearestHit tree;
    traverse(bvh.nodes(), bvh.indices(), tris, bvh.root(), ray, tree);
    const NearestHit brute = brute_force(tris, ray);

    ASSERT_TRUE(tree.hit);
    EXPECT_EQ(tree.prim, 0);
    EXPECT_FLOAT_EQ(tree.dist, 5.0f);
    EXPECT_EQ(tree.prim, brute.prim);
    EXPECT_FLOAT_EQ(tree.dist, brute.dist);
}

TEST(SAHBVH, RebuildOverADifferentSetLeavesNoStaleState) {
    SAHBVH bvh;
    bvh.set_leaf_size(1);
    bvh.build(grid_triangles());
    ASSERT_GT(bvh.nodes().size(), 1u);

    HostBuffer<TriangleContainer4> smaller;
    for (int i = 0; i < 3; ++i) {
        const float x = static_cast<float>(i) * 5.0f;
        smaller.push_back(make_tri(Float3(x, 0.0f, 0.0f),
                                   Float3(x + 1.0f, 0.0f, 0.0f),
                                   Float3(x, 1.0f, 0.0f)));
    }

    bvh.build(smaller);

    EXPECT_EQ(bvh.indices().size(), smaller.size());

    std::vector<char> visited(bvh.nodes().size(), 0);
    std::vector<int> leaf_prims;
    validate_subtree(bvh.nodes(), bvh.indices(), bvh.root(), visited, leaf_prims);

    std::sort(leaf_prims.begin(), leaf_prims.end());
    ASSERT_EQ(leaf_prims.size(), smaller.size());
    for (int i = 0; i < static_cast<int>(leaf_prims.size()); ++i) {
        EXPECT_EQ(leaf_prims[i], i);
    }
}
