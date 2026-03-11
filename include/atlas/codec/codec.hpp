#pragma once

namespace atlas::system {

template <typename T>
Codec<T>::Codec(DomainHostPtr<T> domain)
    : _domain(domain) {
    // The codec needs a domain because it allocates per-cell state and
    // all world->grid indexing (cell counts, linearization, etc.) is derived
    // from the domain's discretization.
    //
    // Without a valid domain, we cannot:
    // - compute number_of_cells(),
    // - size device buffers correctly,
    // - guarantee consistent indexing across kernels.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "Codec: domain must not be null.";

    // Bring the object into a valid, usable state immediately:
    // allocate/resize internal buffers based on the domain grid and
    // initialize them to a known value.
    reset();
}

template <typename T>
void
Codec<T>::update(const ParticleDeviceProbe<T>& particle_probe,
                 const DomainDeviceProbe<T>& domain_probe,
                 const SpatialHashingProbe<T>& searcher_probe,
                 CodecDeviceProbe<T>& codec_probe) {

    this->encode(particle_probe, domain_probe, searcher_probe, codec_probe);
    this->decode(particle_probe, domain_probe, searcher_probe, codec_probe);
}

template <typename T>
CodecDeviceProbe<T>
Codec<T>::make_device_probe() noexcept {
    // Produce a lightweight, device-friendly "view" of the codec state.
    //
    // The returned probe must be trivially copyable so it can be passed to
    // CUDA kernels / device lambdas without owning memory.
    //
    // Important: the probe stores raw pointers into this object's buffers,
    // so it is only valid as long as:
    // - this Codec instance remains particle_count, and
    // - internal buffers are not resized/reset in a way that changes addresses.

    // ------------------------------------------------------------
    // Diagnostic guard: enforce a single probe instance.
    // ------------------------------------------------------------
    // This mirrors the same safety rule used in other subsystems (e.g. searchers):
    // multiple probes increase the chance that some piece of code keeps an old
    // probe past a rebuild/reset and ends up reading/writing through stale pointers.
    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    // Default-construct an empty probe. Even if later code accidentally uses it,
    // it will contain a null pointer unless we fill it below.
    CodecDeviceProbe<T> probe {};

    // ------------------------------------------------------------
    // Attach raw device pointers for kernel-side access.
    // ------------------------------------------------------------
    // allocated_system is a per-cell array (size == domain->number_of_cells()).
    // Typical usage pattern in kernels:
    // - treat each cell as an independent slot,
    // - read/modify the allocation state quickly with O(1) indexing.
    //
    // raw_pointer_cast() removes any wrapper types and yields a plain pointer
    // suitable for device code. The pointer must remain valid for the lifetime
    // of the probe's use.
    probe.allocated_solver = atlas::raw_pointer_cast(d_allocated_system.data());

    // Set the codec type.
    probe.type = this->type();

    return probe;
}

template <typename T>
void
Codec<T>::reset() noexcept {
    // Reinitialize codec internal storage so it matches the current domain grid.
    //
    // number_of_cells() is typically:
    //   grid_size.x * grid_size.y * grid_size.z
    // (or the domain's chosen linearization scheme).
    //
    // We allocate one integer entry per cell. Initial value '0' means:
    // - "unallocated / no solver assigned" (as implied by the name),
    // - and provides a deterministic starting state for subsequent kernel passes.
    const auto num_of_cells = _domain->number_of_cells();

    // Resize the device buffer to exactly one element per cell and fill with zeros.
    // If the buffer already has the correct size, this acts as a clear/reset.
    d_allocated_system.resize(num_of_cells, 0);
}

} // namespace atlas::system
