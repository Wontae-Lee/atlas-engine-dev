#pragma once
namespace atlas::solver {
template <typename T>
DsmcData<T>::DsmcData()
    : d_g_ref_per_cell()
    , d_n_collisions()
    , d_collision_offset()
    , _number_weight(T(1))
    , _d(T(1))
    , _sigma_t(T(1))
    , _cell_volume(T(1))
    , _total_collisions(0) { }
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE DsmcDeviceProbe<T>
DsmcData<T>::make_device_probe() noexcept {
    DsmcDeviceProbe<T> probe {};
    probe.g_ref_per_cell   = atlas::raw_pointer_cast(d_g_ref_per_cell.data());
    probe.n_collisions     = atlas::raw_pointer_cast(d_n_collisions.data());
    probe.collision_offset = atlas::raw_pointer_cast(d_collision_offset.data());
    return probe;
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcData<T>::reset(int n_cells) {
    d_g_ref_per_cell.resize(n_cells, T(0));
    d_n_collisions.resize(n_cells, 0);
    d_collision_offset.resize(n_cells, 0);
    _total_collisions = 0;
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE T
DsmcData<T>::sigma_t() const {
    return _sigma_t;
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE T
DsmcData<T>::cell_volume() const {
    return _cell_volume;
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE T
DsmcData<T>::number_weight() const {
    return _number_weight;
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE int
DsmcData<T>::total_collisions() const {
    return _total_collisions;
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcData<T>::set_total_collisions(int v) {
    _total_collisions = v;
}
}