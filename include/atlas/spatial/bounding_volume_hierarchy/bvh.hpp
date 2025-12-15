#pragma once
namespace atlas::spatial {
template <typename T>
ATLAS_DEVICE HitSurface<T>

BvhTraceOperator<T>::operator()(const Ray<T>& r) const {
    HitSurface<T> out{};
    if (root < 0) return out;
    T best_t = std::numeric_limits<T>::max();
    Vector3<T> best_p{}, best_n{};
    bool found = false;
    int stack[64];
    int sp      = 0;
    stack[sp++] = root;
    while (sp) {
        const int ni         = stack[--sp];
        const BVHNode<T>& nd = nodes[ni];
        HitAABB hit          = nd.bounds.trace(r);
        if (!hit.is_intersecting || hit.enter > best_t) continue;
        if (nd.is_leaf) {
            geometry::TriangleTraceOperator<T> tri_op;
            ATLAS_UNROLL
            for (int k = 0; k < nd.count; ++k) {
                const int pid          = indices[nd.start + k];
                const Triangle<T>& tri = tris[pid];
                tri_op.a               = &tri.a;
                tri_op.b               = &tri.b;
                tri_op.c               = &tri.c;
                tri_op.normal          = &tri.normal;
                HitSurface<T> h        = tri_op(r);
                if (h.is_intersecting && h.distance < best_t) {
                    best_t = h.distance;
                    best_p = h.point;
                    best_n = h.normal;
                    found  = true;
                }
            }
        } else {
            if (sp < 63) stack[sp++] = nd.left;
            else stack[63]           = nd.left;
            if (sp < 63) stack[sp++] = nd.right;
            else stack[63]           = nd.right;
        }
    }
    if (found) {
        out.is_intersecting = true;
        out.distance        = best_t;
        out.point           = best_p;
        out.normal          = best_n;
    }
    return out;
}
}