#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

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
using atlas::LBVH;
using atlas::Ray;
using atlas::Triangle;
using atlas::TriangleContainer4;

/** @brief Nearest-hit summary shared by the tree traversal and the brute-force loop. */
struct NearestHit {
    bool hit    = false;
    float dist  = 0.0f;
    int prim    = -1;
};

/** @brief Packs three vertices (plus a +z placeholder normal) into a primitive. */
TriangleContainer4
make_tri(const Float3& a, const Float3& b, const Float3& c) {
    return TriangleContainer4(a, b, c, Float3(0.0f, 0.0f, 1.0f));
}

/**
 * @brief Eight small, well-separated triangles laid out on a 2x2x2 grid.
 *
 * Spacing (4 units) dwarfs each triangle's extent (1 unit), so centroids are
 * distinct and the hierarchy always has a real two-sided structure.
 */
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
brute_union(const LBVH& bvh) {
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
 * Marks each visited node to prove the tree is acyclic and every node is reached
 * exactly once, checks child indices are in range, verifies each interior node's
 * bounds enclose both children, and collects every leaf's primitive ids.
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

    // Prune whole subtrees whose bounds the ray never enters.
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

TEST(LBVH, BuildOverEmptySetLeavesHierarchyEmpty) {
    LBVH bvh;
    bvh.build(HostBuffer<TriangleContainer4> {});

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
}

TEST(LBVH, BuildOverSinglePrimitiveRootIsLeafCoveringIt) {
    HostBuffer<TriangleContainer4> tris;
    tris.push_back(make_tri(Float3(0.0f, 0.0f, 0.0f),
                            Float3(1.0f, 0.0f, 0.0f),
                            Float3(0.0f, 1.0f, 0.0f)));

    LBVH bvh;
    bvh.build(tris);

    ASSERT_GE(bvh.root(), 0);
    const BVHNode& root = bvh.nodes()[bvh.root()];

    EXPECT_TRUE(root.is_leaf);
    EXPECT_EQ(root.count, 1);
    ASSERT_EQ(bvh.indices().size(), 1u);
    EXPECT_EQ(bvh.indices()[0], 0);

    // The root's bound is exactly the single triangle's bound.
    const AABB expected = bvh.bounds()[0];
    EXPECT_FLOAT_EQ(root.bounds.lower_corner.x, expected.lower_corner.x);
    EXPECT_FLOAT_EQ(root.bounds.upper_corner.y, expected.upper_corner.y);
}

TEST(LBVH, RootBoundEqualsUnionOfAllPrimitiveBounds) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    LBVH bvh;
    bvh.build(tris);

    ASSERT_GE(bvh.root(), 0);
    const AABB expected  = brute_union(bvh);
    const AABB root_bound = bvh.nodes()[bvh.root()].bounds;

    EXPECT_FLOAT_EQ(root_bound.lower_corner.x, expected.lower_corner.x);
    EXPECT_FLOAT_EQ(root_bound.lower_corner.y, expected.lower_corner.y);
    EXPECT_FLOAT_EQ(root_bound.lower_corner.z, expected.lower_corner.z);
    EXPECT_FLOAT_EQ(root_bound.upper_corner.x, expected.upper_corner.x);
    EXPECT_FLOAT_EQ(root_bound.upper_corner.y, expected.upper_corner.y);
    EXPECT_FLOAT_EQ(root_bound.upper_corner.z, expected.upper_corner.z);
}

TEST(LBVH, TreeStructureIsAcyclicNestedAndAPermutation) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    LBVH bvh;
    bvh.build(tris);

    ASSERT_GE(bvh.root(), 0);

    std::vector<char> visited(bvh.nodes().size(), 0);
    std::vector<int> leaf_prims;
    validate_subtree(bvh.nodes(), bvh.indices(), bvh.root(), visited, leaf_prims);

    // Every leaf primitive id appears exactly once, forming a permutation of [0, n).
    std::sort(leaf_prims.begin(), leaf_prims.end());
    ASSERT_EQ(leaf_prims.size(), tris.size());

    for (int i = 0; i < static_cast<int>(leaf_prims.size()); ++i) {
        EXPECT_EQ(leaf_prims[i], i);
    }
}

TEST(LBVH, EveryLeafHoldsExactlyOnePrimitive) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    LBVH bvh;
    bvh.build(tris);

    // Karras' layout produces exactly one primitive per leaf.
    for (std::size_t i = 0; i < bvh.nodes().size(); ++i) {
        if (bvh.nodes()[i].is_leaf) {
            EXPECT_EQ(bvh.nodes()[i].count, 1);
        }
    }
}

TEST(LBVH, MortonBitsChangeTheBuiltOrdering) {
    // Four coplanar triangles whose centroids are at x = 3, 0, 2, 1: their
    // spatial (Morton) order differs from their input order, so a coarse
    // quantization reorders the leaves differently than a fine one.
    HostBuffer<TriangleContainer4> tris;
    const float xs[4] = { 3.0f, 0.0f, 2.0f, 1.0f };

    for (const float cx : xs) {
        tris.push_back(make_tri(Float3(cx - 0.1f, -0.1f, 0.0f),
                                Float3(cx + 0.1f, -0.1f, 0.0f),
                                Float3(cx, 0.2f, 0.0f)));
    }

    LBVH fine;
    fine.set_morton_bits(10);
    fine.build(tris);

    LBVH coarse;
    coarse.set_morton_bits(1);
    coarse.build(tris);

    const std::vector<int> fine_order(fine.indices().begin(), fine.indices().end());
    const std::vector<int> coarse_order(coarse.indices().begin(), coarse.indices().end());

    EXPECT_NE(fine_order, coarse_order);
}

TEST(LBVH, EmitsExactlyTwoNMinusOneNodesWithOnePrimitivePerLeaf) {
    const HostBuffer<TriangleContainer4> tris = grid_triangles();

    LBVH bvh;
    bvh.build(tris);

    // Karras' layout is structural: n leaves and n-1 internal nodes, one primitive each.
    EXPECT_EQ(bvh.nodes().size(), 2u * tris.size() - 1u);

    for (const BVHNode& node : bvh.nodes()) {
        if (node.is_leaf) {
            EXPECT_EQ(node.count, 1);
        }
    }
}

TEST(LBVH, TraversalHitsTheKnownPrimitiveAndMatchesBruteForce) {
    // Six coplanar triangles at distinct x, each covering the point (x_i, 0).
    HostBuffer<TriangleContainer4> tris;

    for (int i = 0; i < 6; ++i) {
        const float x = static_cast<float>(i) * 10.0f;
        tris.push_back(make_tri(Float3(x - 1.0f, -1.0f, 0.0f),
                                Float3(x + 1.0f, -1.0f, 0.0f),
                                Float3(x, 1.0f, 0.0f)));
    }

    LBVH bvh;
    bvh.build(tris);
    ASSERT_GE(bvh.root(), 0);

    // A ray fired straight up through the fourth triangle's face.
    const Ray ray(Float3(30.0f, 0.0f, -5.0f), Float3(0.0f, 0.0f, 1.0f));

    NearestHit tree;
    traverse(bvh.nodes(), bvh.indices(), tris, bvh.root(), ray, tree);
    const NearestHit brute = brute_force(tris, ray);

    ASSERT_TRUE(tree.hit);
    EXPECT_EQ(tree.prim, 3);
    EXPECT_EQ(tree.hit, brute.hit);
    EXPECT_EQ(tree.prim, brute.prim);
    EXPECT_FLOAT_EQ(tree.dist, brute.dist);
}

TEST(LBVH, TraversalReturnsNoHitForAMissAndMatchesBruteForce) {
    HostBuffer<TriangleContainer4> tris;

    for (int i = 0; i < 6; ++i) {
        const float x = static_cast<float>(i) * 10.0f;
        tris.push_back(make_tri(Float3(x - 1.0f, -1.0f, 0.0f),
                                Float3(x + 1.0f, -1.0f, 0.0f),
                                Float3(x, 1.0f, 0.0f)));
    }

    LBVH bvh;
    bvh.build(tris);
    ASSERT_GE(bvh.root(), 0);

    // Fired far outside every triangle's x-extent.
    const Ray ray(Float3(1000.0f, 0.0f, -5.0f), Float3(0.0f, 0.0f, 1.0f));

    NearestHit tree;
    traverse(bvh.nodes(), bvh.indices(), tris, bvh.root(), ray, tree);
    const NearestHit brute = brute_force(tris, ray);

    EXPECT_FALSE(tree.hit);
    EXPECT_EQ(tree.hit, brute.hit);
}

TEST(LBVH, TraversalReturnsTheNearestOfSeveralHits) {
    // Two triangles stacked along the ray (z = 0 and z = 5) plus decoys.
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

    LBVH bvh;
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

TEST(LBVH, RebuildOverADifferentSetLeavesNoStaleState) {
    LBVH bvh;
    bvh.build(grid_triangles());
    ASSERT_EQ(bvh.nodes().size(), 15u);

    // A smaller set must fully replace the larger one.
    HostBuffer<TriangleContainer4> smaller;
    for (int i = 0; i < 3; ++i) {
        const float x = static_cast<float>(i) * 5.0f;
        smaller.push_back(make_tri(Float3(x, 0.0f, 0.0f),
                                   Float3(x + 1.0f, 0.0f, 0.0f),
                                   Float3(x, 1.0f, 0.0f)));
    }

    bvh.build(smaller);

    EXPECT_EQ(bvh.nodes().size(), 2u * smaller.size() - 1u);
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
