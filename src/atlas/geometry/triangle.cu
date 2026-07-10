#include <atlas/geometry/triangle.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Triangle::Builder
Triangle::builder() noexcept {
    return Builder {};
}

Triangle
Triangle::Builder::build() const {
    // Reject collinear (zero-area) vertices before constructing.
    validate();

    Triangle t {};
    t.a = _a;
    t.b = _b;
    t.c = _c;

    if (_normal.has_value()) {
        // Caller-supplied normal wins over the geometric one.
        t.normal = *_normal;
    } else {
        // Derive the unit face normal from the winding of the vertices.
        t.normal = atlas::normalized_or(
            atlas::cross(t.b - t.a, t.c - t.a),
            Float3(0.0f, 0.0f, 0.0f));
    }

    return t;
}

atlas::host_shared_ptr<Triangle>
Triangle::Builder::make_host_shared() const {
    auto t = build();
    return atlas::make_host_shared<Triangle>(std::move(t));
}

Triangle::Builder&
Triangle::Builder::with_a(const Float3& a_) noexcept {
    _a = a_;
    return *this;
}

Triangle::Builder&
Triangle::Builder::with_b(const Float3& b_) noexcept {
    _b = b_;
    return *this;
}

Triangle::Builder&
Triangle::Builder::with_c(const Float3& c_) noexcept {
    _c = c_;
    return *this;
}

Triangle::Builder&
Triangle::Builder::with_vertices(const Float3& a_, const Float3& b_, const Float3& c_) noexcept {
    _a = a_;
    _b = b_;
    _c = c_;
    return *this;
}

Triangle::Builder&
Triangle::Builder::with_normal(const Float3& normal_) noexcept {
    _normal = normal_;
    return *this;
}

void
Triangle::Builder::validate() const {
    // Reuse the Triangle's own non-degeneracy invariant on a throwaway instance.
    if (!Triangle(_a, _b, _c).is_valid()) {
        throw std::runtime_error("Triangle::Builder: invalid triangle.");
    }
}

}
