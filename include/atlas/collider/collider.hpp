#pragma once
namespace atlas::system {
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Sphere<T>& sphere) {
    auto op = sphere.make_trace_operator();
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Cylinder<T>& cylinder) {
    auto op = cylinder.make_trace_operator();
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Plane<T>& plane) {
    auto op = plane.make_trace_operator();
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Box<T>& box) {
    auto op = box.make_trace_operator();
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Triangle<T>& triangle) {
    auto op = triangle.make_trace_operator();
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::TriangleMesh<T>& mesh) {
    auto op = mesh.make_trace_operator();
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Sphere<T>& sphere,
                              const ColliderSurfaceInteraction<T>& interaction) {
    auto op = sphere.make_trace_operator();
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Cylinder<T>& cylinder,
                              const ColliderSurfaceInteraction<T>& interaction) {
    auto op = cylinder.make_trace_operator();
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Plane<T>& plane,
                              const ColliderSurfaceInteraction<T>& interaction) {
    auto op = plane.make_trace_operator();
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Box<T>& box,
                              const ColliderSurfaceInteraction<T>& interaction) {
    auto op = box.make_trace_operator();
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::Triangle<T>& triangle,
                              const ColliderSurfaceInteraction<T>& interaction) {
    auto op = triangle.make_trace_operator();
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_geometry(const atlas::geometry::TriangleMesh<T>& mesh,
                              const ColliderSurfaceInteraction<T>& interaction) {
    auto op = mesh.make_trace_operator();
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_trace_operator(const atlas::geometry::TraceOperator<T>& op) {
    d_trace_operators.push_back(op);
    ColliderSurfaceInteraction<T> interaction;
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::add_trace_operator(const atlas::geometry::TraceOperator<T>& op,
                                    const ColliderSurfaceInteraction<T>& interaction) {
    d_trace_operators.push_back(op);
    d_surface_interactions.push_back(interaction);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    int
    Collider<T>::number_of_surfaces() const {
    return static_cast<int>(d_trace_operators.size());
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::remove_surface(std::size_t index) {
    if (index >= d_trace_operators.size()) {
        return;
    }
    d_trace_operators.erase(d_trace_operators.begin() + index);
    d_surface_interactions.erase(d_surface_interactions.begin() + index);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

    void
    Collider<T>::collide(const ParticleDeviceProbe<T>& probe, T dt, int& active) const {
    auto device_position  = probe.pos;
    auto device_velocity  = probe.vel;
    auto n_surfaces       = number_of_surfaces();
    auto ops_ptr          = atlas::raw_pointer_cast(d_trace_operators.data());
    auto interactions_ptr = atlas::raw_pointer_cast(d_surface_interactions.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        active,
        [=] ATLAS_DEVICE(int i) {
            Vector3F p0          = device_position[i];
            Vector3F velocity    = device_velocity[i];
            Vector3F direction   = velocity * dt;
            float segment_length = length(direction);
            if (segment_length <= tol) {
                return;
            }
            bool any_hit = false;
            float best_t = far;
            Vector3F best_point {};
            Vector3F best_normal {};
            RayF ray { p0, direction };
            int interaction_index = 0;
            for (int j = 0; j < n_surfaces; ++j) {
                const auto& op        = ops_ptr[j];
                HitSurface<float> hit = op(ray);
                if (hit.is_intersecting && hit.distance <= segment_length && hit.distance < best_t) {
                    any_hit           = true;
                    best_t            = hit.distance;
                    best_point        = hit.point;
                    best_normal       = hit.normal;
                    interaction_index = j;
                }
            }
            if (any_hit) {
                const ColliderSurfaceInteraction<T> interaction = interactions_ptr[interaction_index];
                device_position[i]                              = best_point + best_normal * eps;
                device_velocity[i]                              = interaction(velocity, best_normal);
            } else {
                device_position[i] = p0 + direction;
                device_velocity[i] = velocity;
            }
        });
}
}