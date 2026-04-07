#pragma once

namespace atlas::system {

template <typename T>
Codec<T>::Codec(DomainHostPtr<T> domain)
    : _domain(domain) {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "Codec: domain must not be null.";

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

    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    CodecDeviceProbe<T> probe {};

    probe.allocated_system = atlas::raw_pointer_cast(d_allocated_system.data());

    probe.type = this->type();

    return probe;
}

template <typename T>
void
Codec<T>::reset() noexcept {

    const auto num_of_cells = _domain->number_of_cells();

    d_allocated_system.resize(num_of_cells, 0);
}

}
