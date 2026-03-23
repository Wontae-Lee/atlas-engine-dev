#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
Matter<T>::Matter(T mass_)
    : mass(mass_) {

    atlas::check<std::invalid_argument>(mass > T(0))
        << "Matter: mass must be positive.";
}

template <typename T>
typename Matter<T>::Builder
Matter<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
Matter<T>
Matter<T>::Builder::build() const {

    validate();
    return Matter<T>(_mass);
}

template <typename T>
atlas::host_shared_ptr<Matter<T>>
Matter<T>::Builder::make_host_shared() const {

    validate();

    return atlas::make_host_shared<Matter<T>>(_mass);
}

template <typename T>
typename Matter<T>::Builder&
Matter<T>::Builder::with_mass(T mass) noexcept {

    _mass = mass;
    return *this;
}

template <typename T>
void
Matter<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(_mass > T(0))
        << "Matter::Builder: mass must be positive.";
}

}