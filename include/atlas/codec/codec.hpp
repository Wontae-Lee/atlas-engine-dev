#pragma once

namespace atlas::system {

template <typename T>
Codec<T>::Codec(DomainHostPtr<T> domain)
    : _domain(domain) {

    // All codec implementations depend on a concrete simulation domain.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "Codec: domain must not be null.";

    reset();
}

template <typename T>
void
Codec<T>::update(const FluidDeviceProbe<T>& particle_probe,
                 const DomainDeviceProbe<T>& domain_probe,
                 const SpatialHashingProbe<T>& searcher_probe,
                 CodecDeviceProbe<T>& codec_probe) {

    // The default codec pipeline is a full encode/decode round-trip.
    this->encode(particle_probe, domain_probe, searcher_probe, codec_probe);
    this->decode(particle_probe, domain_probe, searcher_probe, codec_probe);
}

template <typename T>
CodecDeviceProbe<T>
Codec<T>::make_device_probe() noexcept {

    ++_probe_count;

    // System owns the single authoritative codec probe for the runtime.
    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    CodecDeviceProbe<T> probe {};

    // Expose only raw device-side state to backend kernels.
    probe.allocated_system = atlas::raw_pointer_cast(d_allocated_system.data());

    probe.type = this->type();

    return probe;
}

template <typename T>
void
Codec<T>::reset() noexcept {

    const auto num_of_cells = _domain->number_of_cells();

    // Track allocation state at domain-cell resolution.
    d_allocated_system.resize(num_of_cells, 0);
}

}
