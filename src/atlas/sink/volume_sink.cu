#include <atlas/sink/volume_sink.h>

#include <stdexcept>
#include <utility>

namespace atlas {

VolumeSink::Builder
VolumeSink::builder() noexcept {
    return Builder {};
}

VolumeSink::Builder&
VolumeSink::Builder::with_unit(Unit unit) {
    _unit = std::move(unit);
    return *this;
}

VolumeSink::Builder&
VolumeSink::Builder::with_tolerance(const float tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

VolumeSink
VolumeSink::Builder::build() {
    validate();

    VolumeSink sink(std::move(*_unit), _tolerance);

    // Consume the accumulated state so a reused builder cannot leak the moved-from
    // unit or a stale tolerance into a second sink.
    _unit.reset();
    _tolerance = 0.0f;

    return sink;
}

atlas::host_shared_ptr<VolumeSink>
VolumeSink::Builder::make_host_shared() {
    return atlas::make_host_shared<VolumeSink>(build());
}

void
VolumeSink::Builder::validate() const {
    if (!_unit) {
        throw std::runtime_error("VolumeSink::Builder: unit must not be null.");
    }

    // A NaN or negative tolerance would make the interior widening meaningless on the
    // device, where the test cannot report failure; reject it here on the host.
    if (!atlas::isfinite(_tolerance) || _tolerance < 0.0f) {
        throw std::runtime_error("VolumeSink::Builder: tolerance must be finite and non-negative.");
    }
}

}
