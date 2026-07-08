#include <atlas/sink/surface_sink.h>

#include <stdexcept>
#include <utility>

namespace atlas {

SurfaceSink::Builder
SurfaceSink::builder() noexcept {
    return Builder {};
}

SurfaceSink::Builder&
SurfaceSink::Builder::with_unit(Unit unit) {
    _unit = std::move(unit);
    return *this;
}

SurfaceSink::Builder&
SurfaceSink::Builder::with_tolerance(const float tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

SurfaceSink
SurfaceSink::Builder::build() {
    validate();

    SurfaceSink sink(std::move(*_unit), _tolerance);

    _unit.reset();
    _tolerance = 0.0f;

    return sink;
}

atlas::host_shared_ptr<SurfaceSink>
SurfaceSink::Builder::make_host_shared() {
    return atlas::make_host_shared<SurfaceSink>(build());
}

void
SurfaceSink::Builder::validate() const {
    if (!_unit) {
        throw std::runtime_error("SurfaceSink::Builder: unit must not be null.");
    }

    if (!atlas::isfinite(_tolerance) || _tolerance < 0.0f) {
        throw std::runtime_error("SurfaceSink::Builder: tolerance must be finite and non-negative.");
    }
}

}
