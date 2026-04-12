#pragma once

namespace atlas::system {

template <typename T>
Codec<T>::Codec(DomainHostPtr<T> domain)
    : _domain(domain) {

    // Every codec instance must be attached to a valid simulation domain.
    // The domain provides the spatial resolution and topology information
    // required by codec-side buffers and update logic.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "Codec: domain must not be null.";

    // Initialize all internal codec state immediately after construction
    // so the object starts in a consistent, ready-to-use configuration.
    reset();
}

template <typename T>
void
Codec<T>::update(const FluidDeviceProbe<T>& particle_probe,
                 const DomainDeviceProbe<T>& domain_probe,
                 const SpatialHashingProbe<T>& searcher_probe,
                 CodecDeviceProbe<T>& codec_probe) {

    // The base codec update path performs a complete round-trip:
    //
    // 1. encode(...)
    //    Gather or transform runtime state into codec-managed representation.
    //
    // 2. decode(...)
    //    Reconstruct or apply codec-managed data back into the simulation view.
    //
    // Derived implementations may override the participating steps,
    // but the default orchestration is always encode followed by decode.
    this->encode(particle_probe, domain_probe, searcher_probe, codec_probe);
    this->decode(particle_probe, domain_probe, searcher_probe, codec_probe);
}

template <typename T>
CodecDeviceProbe<T>
Codec<T>::make_device_probe() noexcept {

    // Track how many times a device probe has been requested.
    // This object is designed around a single authoritative device-side probe.
    ++_probe_count;

    // Enforce the invariant that exactly one runtime-owned device probe exists.
    //
    // Rationale:
    // - Backend kernels should observe one canonical codec state handle.
    // - Multiple probes could imply duplicated or inconsistent device bindings.
    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    // Start from a zero-initialized probe object so every exposed field
    // has a defined default state before selective population.
    CodecDeviceProbe<T> probe {};

    // Publish raw device-accessible storage that marks per-cell system allocation state.
    //
    // This intentionally exposes only the minimal low-level pointer needed by
    // device kernels, rather than the higher-level container object itself.
    probe.allocated_system = atlas::raw_pointer_cast(d_allocated_system.data());

    // Record the concrete codec type so device-side or dispatch-side logic
    // can identify which codec behavior the probe corresponds to.
    probe.type = this->type();

    // Return the fully bound device probe view.
    return probe;
}

template <typename T>
void
Codec<T>::reset() noexcept {

    // Query the domain resolution in terms of total simulation cells.
    // Codec allocation state is maintained at cell granularity.
    const auto num_of_cells = _domain->number_of_cells();

    // Resize and reinitialize the device-side allocation-state buffer.
    //
    // Semantics:
    // - one entry per domain cell
    // - initial value 0 means "not allocated / inactive / cleared"
    //
    // This restores codec-managed state to a clean baseline.
    d_allocated_system.resize(num_of_cells, 0);
}

} // namespace atlas::system