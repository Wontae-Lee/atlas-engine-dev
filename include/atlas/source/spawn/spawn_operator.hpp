// atlas/source/spawn/spawn_operator.hpp
#pragma once

namespace atlas::system {

template <typename T>
template <typename Engine>
T
BoxSpawnOperator<T>::uniform_real(Engine& engine, T min_value, T max_value) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T>
template <typename Engine>
Vector3<T>
BoxSpawnOperator<T>::sample_volume(const atlas::geometry::Box<T>& box, Engine& engine) {
    return Vector3<T>(
        uniform_real(engine, box.lower_corner.x, box.upper_corner.x),
        uniform_real(engine, box.lower_corner.y, box.upper_corner.y),
        uniform_real(engine, box.lower_corner.z, box.upper_corner.z));
}

template <typename T>
template <typename Engine>
Vector3<T>
BoxSpawnOperator<T>::sample_surface(const atlas::geometry::Box<T>& box, Engine& engine) {
    const T dx = box.upper_corner.x - box.lower_corner.x;
    const T dy = box.upper_corner.y - box.lower_corner.y;
    const T dz = box.upper_corner.z - box.lower_corner.z;

    const T yz_area = dy * dz;
    const T xz_area = dx * dz;
    const T xy_area = dx * dy;
    const T total   = T(2) * (yz_area + xz_area + xy_area);

    if (total <= T(0)) return box.lower_corner;

    T pick = uniform_real(engine, T(0), total);

    auto sample_x = [&]() { return uniform_real(engine, box.lower_corner.x, box.upper_corner.x); };
    auto sample_y = [&]() { return uniform_real(engine, box.lower_corner.y, box.upper_corner.y); };
    auto sample_z = [&]() { return uniform_real(engine, box.lower_corner.z, box.upper_corner.z); };

    if ((pick -= yz_area) <= T(0)) return Vector3<T>(box.lower_corner.x, sample_y(), sample_z());
    if ((pick -= yz_area) <= T(0)) return Vector3<T>(box.upper_corner.x, sample_y(), sample_z());
    if ((pick -= xz_area) <= T(0)) return Vector3<T>(sample_x(), box.lower_corner.y, sample_z());
    if ((pick -= xz_area) <= T(0)) return Vector3<T>(sample_x(), box.upper_corner.y, sample_z());
    if ((pick -= xy_area) <= T(0)) return Vector3<T>(sample_x(), sample_y(), box.lower_corner.z);
    return Vector3<T>(sample_x(), sample_y(), box.upper_corner.z);
}

template <typename T>
HostBuffer<Vector3<T>>
BoxSpawnOperator<T>::spawn(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || !lower_corner || !upper_corner) return out;

    atlas::geometry::Box<T> box;
    box.lower_corner = *lower_corner;
    box.upper_corner = *upper_corner;
    if (!box.is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(type == SpawnType::Surface ? sample_surface(box, engine) : sample_volume(box, engine));
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
BoxSpawnOperator<T>::operator()(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    return spawn(count, type, seed);
}

template <typename T>
template <typename Engine>
T
SphereSpawnOperator<T>::uniform_real(Engine& engine, T min_value, T max_value) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T>
template <typename Engine>
Vector3<T>
SphereSpawnOperator<T>::sample_unit_direction(Engine& engine) {
    constexpr long double pi_ld = 3.1415926535897932384626433832795L;
    const T pi                  = static_cast<T>(pi_ld);

    const T z   = uniform_real(engine, T(-1), T(1));
    const T phi = uniform_real(engine, T(0), T(2) * pi);
    const T r2  = std::max(T(0), T(1) - z * z);
    const T r   = static_cast<T>(std::sqrt(r2));

    return Vector3<T>(
        r * static_cast<T>(std::cos(phi)),
        r * static_cast<T>(std::sin(phi)),
        z);
}

template <typename T>
template <typename Engine>
Vector3<T>
SphereSpawnOperator<T>::sample_surface(const atlas::geometry::Sphere<T>& sphere, Engine& engine) {
    return sphere.center + sample_unit_direction(engine) * sphere.radius;
}

template <typename T>
template <typename Engine>
Vector3<T>
SphereSpawnOperator<T>::sample_volume(const atlas::geometry::Sphere<T>& sphere, Engine& engine) {
    const T radius = sphere.radius * static_cast<T>(std::cbrt(uniform_real(engine)));
    return sphere.center + sample_unit_direction(engine) * radius;
}

template <typename T>
HostBuffer<Vector3<T>>
SphereSpawnOperator<T>::spawn(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || !center || !radius) return out;

    atlas::geometry::Sphere<T> sphere;
    sphere.center = *center;
    sphere.radius = *radius;
    if (!sphere.is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(type == SpawnType::Surface ? sample_surface(sphere, engine) : sample_volume(sphere, engine));
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
SphereSpawnOperator<T>::operator()(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    return spawn(count, type, seed);
}

template <typename T>
template <typename Engine>
T
CylinderSpawnOperator<T>::uniform_real(Engine& engine, T min_value, T max_value) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T>
template <typename Engine>
Vector3<T>
CylinderSpawnOperator<T>::sample_disk(const Vector3<T>& center,
                                      const Vector3<T>& tangent,
                                      const Vector3<T>& bitangent,
                                      const T radius,
                                      Engine& engine) {
    constexpr long double pi_ld = 3.1415926535897932384626433832795L;
    const T pi                  = static_cast<T>(pi_ld);

    const T r   = radius * static_cast<T>(std::sqrt(uniform_real(engine)));
    const T phi = uniform_real(engine, T(0), T(2) * pi);

    return center
        + tangent * (r * static_cast<T>(std::cos(phi)))
        + bitangent * (r * static_cast<T>(std::sin(phi)));
}

template <typename T>
template <typename Engine>
Vector3<T>
CylinderSpawnOperator<T>::sample_volume(const atlas::geometry::Cylinder<T>& cylinder, Engine& engine) {
    constexpr long double pi_ld = 3.1415926535897932384626433832795L;
    const T pi                  = static_cast<T>(pi_ld);

    const T phi    = uniform_real(engine, T(0), T(2) * pi);
    const T radial = cylinder.radius * static_cast<T>(std::sqrt(uniform_real(engine)));
    const T z      = uniform_real(engine, -cylinder.height * T(0.5), cylinder.height * T(0.5));

    return Vector3<T>(
        cylinder.center.x + radial * static_cast<T>(std::cos(phi)),
        cylinder.center.y + radial * static_cast<T>(std::sin(phi)),
        cylinder.center.z + z);
}

template <typename T>
template <typename Engine>
Vector3<T>
CylinderSpawnOperator<T>::sample_surface(const atlas::geometry::Cylinder<T>& cylinder, Engine& engine) {
    constexpr long double pi_ld = 3.1415926535897932384626433832795L;
    const T pi                  = static_cast<T>(pi_ld);

    const T side_area = T(2) * pi * cylinder.radius * cylinder.height;
    const T cap_area  = pi * cylinder.radius * cylinder.radius;
    const T total     = side_area + T(2) * cap_area;

    if (total <= T(0)) return cylinder.center;

    const T pick = uniform_real(engine, T(0), total);

    if (pick < side_area) {
        const T phi = uniform_real(engine, T(0), T(2) * pi);
        const T z   = uniform_real(engine, -cylinder.height * T(0.5), cylinder.height * T(0.5));
        return Vector3<T>(
            cylinder.center.x + cylinder.radius * static_cast<T>(std::cos(phi)),
            cylinder.center.y + cylinder.radius * static_cast<T>(std::sin(phi)),
            cylinder.center.z + z);
    }

    const T z = (uniform_real(engine) < T(0.5)) ? (-cylinder.height * T(0.5)) : (cylinder.height * T(0.5));

    return sample_disk(
        Vector3<T>(cylinder.center.x, cylinder.center.y, cylinder.center.z + z),
        Vector3<T>(T(1), T(0), T(0)),
        Vector3<T>(T(0), T(1), T(0)),
        cylinder.radius,
        engine);
}

template <typename T>
HostBuffer<Vector3<T>>
CylinderSpawnOperator<T>::spawn(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || !center || !radius || !height) return out;

    atlas::geometry::Cylinder<T> cylinder;
    cylinder.center = *center;
    cylinder.radius = *radius;
    cylinder.height = *height;
    if (!cylinder.is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(type == SpawnType::Surface ? sample_surface(cylinder, engine) : sample_volume(cylinder, engine));
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
CylinderSpawnOperator<T>::operator()(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    return spawn(count, type, seed);
}

template <typename T>
template <typename Engine>
T
TriangleSpawnOperator<T>::uniform_real(Engine& engine, T min_value, T max_value) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T>
template <typename Engine>
Vector3<T>
TriangleSpawnOperator<T>::sample_surface(const atlas::geometry::Triangle<T>& triangle, Engine& engine) {
    const T u  = uniform_real(engine);
    const T v  = uniform_real(engine);
    const T su = static_cast<T>(std::sqrt(u));

    const T w0 = T(1) - su;
    const T w1 = su * (T(1) - v);
    const T w2 = su * v;

    return triangle.a * w0 + triangle.b * w1 + triangle.c * w2;
}

template <typename T>
HostBuffer<Vector3<T>>
TriangleSpawnOperator<T>::spawn(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || type == SpawnType::Volume || !a || !b || !c) return out;

    atlas::geometry::Triangle<T> triangle;
    triangle.a = *a;
    triangle.b = *b;
    triangle.c = *c;
    if (!triangle.is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(sample_surface(triangle, engine));
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
TriangleSpawnOperator<T>::operator()(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    return spawn(count, type, seed);
}

template <typename T>
template <typename Engine>
T
TriangleMeshSpawnOperator<T>::uniform_real(Engine& engine, T min_value, T max_value) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T>
T
TriangleMeshSpawnOperator<T>::triangle_area(const TriangleContainer4<T>& tri) {
    return T(0.5) * atlas::math::cross(tri.b() - tri.a(), tri.c() - tri.a()).length();
}

template <typename T>
bool
TriangleMeshSpawnOperator<T>::is_finite_vector(const Vector3<T>& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

template <typename T>
bool
TriangleMeshSpawnOperator<T>::has_finite_bounds(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds) noexcept {
    return is_finite_vector(bounds.lower_corner)
        && is_finite_vector(bounds.upper_corner)
        && bounds.width() >= T(0)
        && bounds.height() >= T(0)
        && bounds.depth() >= T(0);
}

template <typename T>
template <typename Engine>
Vector3<T>
TriangleMeshSpawnOperator<T>::sample_aabb(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds, Engine& engine) {
    return Vector3<T>(
        uniform_real(engine, bounds.lower_corner.x, bounds.upper_corner.x),
        uniform_real(engine, bounds.lower_corner.y, bounds.upper_corner.y),
        uniform_real(engine, bounds.lower_corner.z, bounds.upper_corner.z));
}

template <typename T>
template <typename Engine>
Vector3<T>
TriangleMeshSpawnOperator<T>::sample_surface(const atlas::geometry::TriangleMesh<T>& mesh,
                                             const HostBuffer<T>& prefix,
                                             const T total_area,
                                             Engine& engine) {
    if (mesh.triangles.empty()) return Vector3<T>(T(0), T(0), T(0));
    if (total_area <= T(0)) return mesh.triangles.front().a();

    const T pick                = uniform_real(engine, T(0), total_area);
    const auto it               = std::lower_bound(prefix.begin(), prefix.end(), pick);
    const std::size_t tri_index = static_cast<std::size_t>(std::distance(prefix.begin(), it));
    const auto& tri             = mesh.triangles[std::min(tri_index, mesh.triangles.size() - 1)];

    const T u  = uniform_real(engine);
    const T v  = uniform_real(engine);
    const T su = static_cast<T>(std::sqrt(u));

    const T w0 = T(1) - su;
    const T w1 = su * (T(1) - v);
    const T w2 = su * v;

    return tri.a() * w0 + tri.b() * w1 + tri.c() * w2;
}

template <typename T>
template <typename Engine>
HostBuffer<Vector3<T>>
TriangleMeshSpawnOperator<T>::sample_volume(const std::size_t count, Engine& engine) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (!mesh) return out;

    const auto bounds = mesh->bound();
    if (!has_finite_bounds(bounds)) return out;

    const std::size_t max_attempts = std::max<std::size_t>(count, 1) * std::max<std::size_t>(max_volume_rejection_iterations, 1);

    std::size_t attempts = 0;
    while (out.size() < count && attempts < max_attempts) {
        const Vector3<T> p = sample_aabb(bounds, engine);
        if (mesh->signed_distance(p) <= T(0)) out.push_back(p);
        ++attempts;
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
TriangleMeshSpawnOperator<T>::spawn(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || !mesh || !triangles || triangles->empty() || !mesh->is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    if (type == SpawnType::Volume) {
        return sample_volume(count, engine);
    }

    HostBuffer<T> prefix(triangles->size(), T(0));
    T total_area = T(0);

    for (std::size_t i = 0; i < triangles->size(); ++i) {
        total_area += triangle_area((*triangles)[i]);
        prefix[i] = total_area;
    }

    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(sample_surface(*mesh, prefix, total_area, engine));
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
TriangleMeshSpawnOperator<T>::operator()(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    return spawn(count, type, seed);
}

template <typename T>
template <typename Engine>
T
PlaneSpawnOperator<T>::uniform_real(Engine& engine, T min_value, T max_value) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T>
template <typename Engine>
Vector3<T>
PlaneSpawnOperator<T>::sample_surface(const atlas::geometry::Plane<T>& plane, const T patch_extent, Engine& engine) {
    const T n2 = plane.normal.length_squared();
    if (n2 <= atlas::eps) return Vector3<T>(T(0), T(0), T(0));

    const T inv_n_len       = T(1) / static_cast<T>(std::sqrt(n2));
    const Vector3<T> n      = plane.normal * inv_n_len;
    const Vector3<T> origin = plane.normal * (plane.offset / n2);

    Vector3<T> tangent;
    Vector3<T> bitangent;
    atlas::random::build_orthonormal_basis(n, tangent, bitangent);

    return origin
        + tangent * uniform_real(engine, -patch_extent, patch_extent)
        + bitangent * uniform_real(engine, -patch_extent, patch_extent);
}

template <typename T>
HostBuffer<Vector3<T>>
PlaneSpawnOperator<T>::spawn(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || type == SpawnType::Volume || !normal || !offset) return out;

    atlas::geometry::Plane<T> plane;
    plane.normal = *normal;
    plane.offset = *offset;
    if (!plane.is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(sample_surface(plane, plane_surface_extent, engine));
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
PlaneSpawnOperator<T>::operator()(const std::size_t count, const SpawnType type, const std::uint32_t seed) const {
    return spawn(count, type, seed);
}

template <typename T>
SpawnOperator<T>::SpawnOperator() noexcept
    : type(SpawnType::Surface)
    , seed(5489u)
    , plane_surface_extent(T(1))
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Sphere) {
    new (&sphere) SphereSpawnOperator<T> {};
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type_,
                                const std::uint32_t seed_,
                                const T plane_surface_extent_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(plane_surface_extent_)
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Sphere) {
    new (&sphere) SphereSpawnOperator<T> {};
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnOperator& other) noexcept
    : type(other.type)
    , seed(other.seed)
    , plane_surface_extent(other.plane_surface_extent)
    , max_volume_rejection_iterations(other.max_volume_rejection_iterations)
    , geometry_type(other.geometry_type) {
    copy_from(other);
}

template <typename T>
SpawnOperator<T>&
SpawnOperator<T>::operator=(const SpawnOperator& other) noexcept {
    if (this == &other) return *this;

    destroy_active();
    type                            = other.type;
    seed                            = other.seed;
    plane_surface_extent            = other.plane_surface_extent;
    max_volume_rejection_iterations = other.max_volume_rejection_iterations;
    geometry_type                   = other.geometry_type;
    copy_from(other);
    return *this;
}

template <typename T>
SpawnOperator<T>::~SpawnOperator() noexcept {
    destroy_active();
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SphereSpawnOperator<T>& op,
                                const SpawnType type_,
                                const std::uint32_t seed_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(T(1))
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Sphere) {
    new (&sphere) SphereSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const CylinderSpawnOperator<T>& op,
                                const SpawnType type_,
                                const std::uint32_t seed_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(T(1))
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Cylinder) {
    new (&cylinder) CylinderSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const PlaneSpawnOperator<T>& op,
                                const SpawnType type_,
                                const std::uint32_t seed_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(op.plane_surface_extent)
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Plane) {
    new (&plane) PlaneSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const BoxSpawnOperator<T>& op,
                                const SpawnType type_,
                                const std::uint32_t seed_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(T(1))
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Box) {
    new (&box) BoxSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const TriangleSpawnOperator<T>& op,
                                const SpawnType type_,
                                const std::uint32_t seed_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(T(1))
    , max_volume_rejection_iterations(64)
    , geometry_type(atlas::geometry::GeometryType::Triangle) {
    new (&triangle) TriangleSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const TriangleMeshSpawnOperator<T>& op,
                                const SpawnType type_,
                                const std::uint32_t seed_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(T(1))
    , max_volume_rejection_iterations(op.max_volume_rejection_iterations)
    , geometry_type(atlas::geometry::GeometryType::TriangleMesh) {
    new (&triangle_mesh) TriangleMeshSpawnOperator<T>(op);
}

template <typename T>
void
SpawnOperator<T>::destroy_active() noexcept {
    switch (geometry_type) {
    case atlas::geometry::GeometryType::Sphere:
        sphere.~SphereSpawnOperator<T>();
        break;
    case atlas::geometry::GeometryType::Cylinder:
        cylinder.~CylinderSpawnOperator<T>();
        break;
    case atlas::geometry::GeometryType::Plane:
        plane.~PlaneSpawnOperator<T>();
        break;
    case atlas::geometry::GeometryType::Box:
        box.~BoxSpawnOperator<T>();
        break;
    case atlas::geometry::GeometryType::Triangle:
        triangle.~TriangleSpawnOperator<T>();
        break;
    case atlas::geometry::GeometryType::TriangleMesh:
        triangle_mesh.~TriangleMeshSpawnOperator<T>();
        break;
    default:
        sphere.~SphereSpawnOperator<T>();
        break;
    }
}

template <typename T>
void
SpawnOperator<T>::copy_from(const SpawnOperator& other) noexcept {
    switch (other.geometry_type) {
    case atlas::geometry::GeometryType::Sphere:
        new (&sphere) SphereSpawnOperator<T>(other.sphere);
        break;
    case atlas::geometry::GeometryType::Cylinder:
        new (&cylinder) CylinderSpawnOperator<T>(other.cylinder);
        break;
    case atlas::geometry::GeometryType::Plane:
        new (&plane) PlaneSpawnOperator<T>(other.plane);
        break;
    case atlas::geometry::GeometryType::Box:
        new (&box) BoxSpawnOperator<T>(other.box);
        break;
    case atlas::geometry::GeometryType::Triangle:
        new (&triangle) TriangleSpawnOperator<T>(other.triangle);
        break;
    case atlas::geometry::GeometryType::TriangleMesh:
        new (&triangle_mesh) TriangleMeshSpawnOperator<T>(other.triangle_mesh);
        break;
    default:
        new (&sphere) SphereSpawnOperator<T>(other.sphere);
        break;
    }
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::spawn(const std::size_t count) const {
    switch (geometry_type) {
    case atlas::geometry::GeometryType::Sphere:
        return sphere(count, type, seed);
    case atlas::geometry::GeometryType::Cylinder:
        return cylinder(count, type, seed);
    case atlas::geometry::GeometryType::Plane:
        return plane(count, type, seed);
    case atlas::geometry::GeometryType::Box:
        return box(count, type, seed);
    case atlas::geometry::GeometryType::Triangle:
        return triangle(count, type, seed);
    case atlas::geometry::GeometryType::TriangleMesh:
        return triangle_mesh(count, type, seed);
    default:
        return {};
    }
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::spawn(const atlas::geometry::Geometry<T>& geometry, const std::size_t count) const {
    switch (geometry.type()) {
    case atlas::geometry::GeometryType::Box: {
        const auto& g = static_cast<const atlas::geometry::Box<T>&>(geometry);
        BoxSpawnOperator<T> op;
        op.lower_corner = &g.lower_corner;
        op.upper_corner = &g.upper_corner;
        return op(count, type, seed);
    }
    case atlas::geometry::GeometryType::Sphere: {
        const auto& g = static_cast<const atlas::geometry::Sphere<T>&>(geometry);
        SphereSpawnOperator<T> op;
        op.center = &g.center;
        op.radius = &g.radius;
        return op(count, type, seed);
    }
    case atlas::geometry::GeometryType::Cylinder: {
        const auto& g = static_cast<const atlas::geometry::Cylinder<T>&>(geometry);
        CylinderSpawnOperator<T> op;
        op.center = &g.center;
        op.radius = &g.radius;
        op.height = &g.height;
        return op(count, type, seed);
    }
    case atlas::geometry::GeometryType::Triangle: {
        const auto& g = static_cast<const atlas::geometry::Triangle<T>&>(geometry);
        TriangleSpawnOperator<T> op;
        op.a = &g.a;
        op.b = &g.b;
        op.c = &g.c;
        return op(count, type, seed);
    }
    case atlas::geometry::GeometryType::TriangleMesh: {
        const auto& g = static_cast<const atlas::geometry::TriangleMesh<T>&>(geometry);
        TriangleMeshSpawnOperator<T> op;
        op.mesh                            = &g;
        op.triangles                       = &g.triangles;
        op.max_volume_rejection_iterations = max_volume_rejection_iterations;
        return op(count, type, seed);
    }
    case atlas::geometry::GeometryType::Plane: {
        const auto& g = static_cast<const atlas::geometry::Plane<T>&>(geometry);
        PlaneSpawnOperator<T> op;
        op.normal               = &g.normal;
        op.offset               = &g.offset;
        op.plane_surface_extent = plane_surface_extent;
        return op(count, type, seed);
    }
    }

    return {};
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::spawn(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry,
                        const std::size_t count) const {
    if (!geometry) return {};
    return spawn(*geometry, count);
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::operator()(const std::size_t count) const {
    return spawn(count);
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::operator()(const atlas::geometry::Geometry<T>& geometry, const std::size_t count) const {
    return spawn(geometry, count);
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::operator()(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry,
                             const std::size_t count) const {
    return spawn(geometry, count);
}

} // namespace atlas::system