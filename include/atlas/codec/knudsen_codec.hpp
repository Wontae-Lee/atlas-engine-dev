#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {

    // Return a default-initialized builder for fluent KnudsenCodec construction.
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(UniverseHostPtr<T> domain,
                              FluidHostPtr<T> fluid,
                              SpatialHashingSearcherHostPtr<T> searcher,
                              T characteristic_length)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher))
    , _characteristic_length(characteristic_length) {

    // The characteristic length is a required physical/model parameter and
    // must be strictly positive for Knudsen-related normalization.
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";

    // Allocate one Knudsen-related value per domain cell.
    //
    // The base Codec is expected to provide access to the associated universe,
    // whose cell count determines the storage size for the encoded field.
    const auto num_of_cells = this->_domain->number_of_cells();

    d_knudsen_values.resize(num_of_cells, T(0));
}

template <typename T>
void
KnudsenCodec<T>::encode() {
    // Encoding logic is not implemented yet.
    //
    // This function is intended to populate d_knudsen_values from the current
    // simulation/domain state in a future implementation.
}

template <typename T>
void
KnudsenCodec<T>::decode() {
    // Decoding logic is not implemented yet.
    //
    // This function is intended to interpret or reconstruct state from the
    // stored Knudsen-related field in a future implementation.
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {

    // Store the universe/domain dependency for later construction.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    // Store the fluid dependency for later construction.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    // Store the spatial hashing searcher dependency for later construction.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {

    // Store the characteristic length parameter for later validation and construction.
    _characteristic_length = characteristic_length;
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {

    // All required dependencies must be present before a valid codec can be built.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KnudsenCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "KnudsenCodec::Builder: searcher must not be null.";

    // The characteristic length must be strictly positive.
    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {

    // Validate the builder configuration before constructing a value object.
    validate();

    return KnudsenCodec<T>(_domain, _fluid, _searcher, _characteristic_length);
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {

    // Validate the builder configuration before constructing a shared instance.
    validate();

    return atlas::make_host_shared<KnudsenCodec<T>>(_domain, _fluid, _searcher, _characteristic_length);
}

} // namespace atlas::system