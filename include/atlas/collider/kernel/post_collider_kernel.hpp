#pragma once

namespace atlas {

namespace detail {

    template <typename T>
    using PostColliderVariant = DeviceVariant<
        PostColliderKernel<T>,
        PostColliderType,
        PostColliderType::fast,
        DeviceVariantCase<
            PostColliderKernel<T>,
            PostColliderType,
            PostColliderType::fast,
            FastColliderKernel<T>,
            &PostColliderKernel<T>::fast>,
        DeviceVariantCase<
            PostColliderKernel<T>,
            PostColliderType,
            PostColliderType::dt_remain,
            DtRemainColliderKernel<T>,
            &PostColliderKernel<T>::dt_remain>,
        DeviceVariantCase<
            PostColliderKernel<T>,
            PostColliderType,
            PostColliderType::precise,
            PreciseColliderKernel<T>,
            &PostColliderKernel<T>::precise>>;

}

template <typename T>
PostColliderKernel<T>::PostColliderKernel() noexcept {
    detail::PostColliderVariant<T>::construct(*this, PostColliderType::fast);
}

template <typename T>
PostColliderKernel<T>::PostColliderKernel(const PostColliderType type_) noexcept {
    detail::PostColliderVariant<T>::construct(*this, type_);
}

template <typename T>
PostColliderKernel<T>::PostColliderKernel(const PostColliderKernel& other) noexcept {
    detail::PostColliderVariant<T>::copy_construct(*this, other);
}

template <typename T>
PostColliderKernel<T>&
PostColliderKernel<T>::operator=(const PostColliderKernel& other) noexcept {
    detail::PostColliderVariant<T>::assign(*this, other);
    return *this;
}

template <typename T>
PostColliderKernel<T>::~PostColliderKernel() noexcept {
    destroy_active();
}

template <typename T>
void
PostColliderKernel<T>::sweep_motion(const Unit<T>& unit,
                                    const Vector3<T>& origin,
                                    const Vector3<T>& incident,
                                    const T incident_speed,
                                    const T dt,
                                    Vector3<T>& sweep_direction,
                                    T& sweep_speed,
                                    T& sweep_length) const noexcept {
    detail::PostColliderVariant<T>::apply(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& kernel) noexcept {
            kernel.sweep_motion(unit, origin, incident, incident_speed, dt, sweep_direction, sweep_speed, sweep_length);
        });
}

template <typename T>
void
PostColliderKernel<T>::operator()(Vector3<T>& position,
                                  Vector3<T>& velocity,
                                  const Vector3<T>& incident,
                                  const Vector3<T>& hit_position,
                                  const Vector3<T>& hit_normal,
                                  const T hit_distance,
                                  const T sweep_speed,
                                  const T dt,
                                  const Unit<T>& unit,
                                  const SurfaceInteractionKernel<T>& interaction) const noexcept {
    detail::PostColliderVariant<T>::apply(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& kernel) noexcept {
            kernel(position, velocity, incident, hit_position, hit_normal, hit_distance, sweep_speed, dt, unit, interaction);
        });
}

template <typename T>
void
PostColliderKernel<T>::destroy_active() noexcept {
    detail::PostColliderVariant<T>::destroy(*this);
}

template <typename T>
void
PostColliderKernel<T>::copy_from(const PostColliderKernel& other) noexcept {
    detail::PostColliderVariant<T>::copy_construct(*this, other);
}

}