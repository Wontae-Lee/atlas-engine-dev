// #pragma once
// #include <atlas/collider/collider_surface_interaction.h>
// #include <atlas/core/macros.h>
// #include <atlas/geometry/trace_operator.h>
// #include <atlas/memory/memory.h>
// #include <atlas/system/particle_data.h>
// #include <cstddef>
//
// namespace atlas {
// namespace system {
//     template <typename T>
//     class Collider final {
//     public:
//         Collider()  = default;
//         ~Collider() = default;
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Sphere<T>& sphere);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Cylinder<T>& cylinder);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Plane<T>& plane);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Box<T>& box);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Triangle<T>& triangle);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::TriangleMesh<T>& mesh);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Sphere<T>& sphere,
//                      const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Cylinder<T>& cylinder,
//                      const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Plane<T>& plane,
//                      const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Box<T>& box,
//                      const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::Triangle<T>& triangle,
//                      const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_geometry(const atlas::geometry::TriangleMesh<T>& mesh,
//                      const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_trace_operator(const atlas::geometry::TraceOperator<T>& op);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         add_trace_operator(const atlas::geometry::TraceOperator<T>& op,
//                            const ColliderSurfaceInteraction<T>& interaction);
//         ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
//         number_of_surfaces() const;
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         remove_surface(std::size_t index);
//         ATLAS_HOST ATLAS_FORCE_INLINE void
//         collide(const ParticleDeviceProbe<T>& probe, T dt) const;
//
//     private:
//         DeviceBuffer<atlas::geometry::TraceOperator<T>> d_trace_operators;
//         DeviceBuffer<ColliderSurfaceInteraction<T>> d_surface_interactions;
//     };
// }
//
// template <typename T>
// using Collider = system::Collider<T>;
// template <typename T>
// using ColliderHostPtr = host_shared_ptr<Collider<T>>;
// template <typename T>
// using ColliderDevicePtr = device_shared_ptr<Collider<T>>;
// }
//
// #include <atlas/collider/collider.hpp>