#include <atlas/sink/tracing_sink.h>

#include <stdexcept>
#include <utility>

namespace atlas {

TracingSink::Builder
TracingSink::builder() noexcept {
    return Builder {};
}

TracingSink::Builder&
TracingSink::Builder::with_unit(Unit unit) {
    _unit = std::move(unit);
    return *this;
}

TracingSink
TracingSink::Builder::build() {
    validate();

    TracingSink sink(std::move(*_unit));

    // Consume the accumulated state so a reused builder cannot leak the moved-from
    // unit into a second sink.
    _unit.reset();

    return sink;
}

atlas::host_shared_ptr<TracingSink>
TracingSink::Builder::make_host_shared() {
    return atlas::make_host_shared<TracingSink>(build());
}

void
TracingSink::Builder::validate() const {
    if (!_unit) {
        throw std::runtime_error("TracingSink::Builder: unit must not be null.");
    }
}

}
