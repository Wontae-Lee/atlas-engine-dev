#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/copy.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace atlas::python {

template <typename T>
using NumpyScalar = std::conditional_t<std::is_same_v<T, std::size_t>, std::uint64_t, T>;

template <typename Buffer>
nanobind::object
numpy_copy(const Buffer& source, const std::size_t count) {
    namespace nb = nanobind;
    using T = typename Buffer::value_type;
    if (count > source.size()) {
        throw nb::value_error("state buffer is shorter than the requested count");
    }
    const HostBuffer<T> host(source.begin(), source.begin() + count);
    if constexpr (std::is_same_v<T, Float3>) {
        auto buffer = std::make_unique<float[]>(3 * count);
        for (std::size_t i = 0; i < count; ++i) {
            buffer[3 * i] = host[i].x;
            buffer[3 * i + 1] = host[i].y;
            buffer[3 * i + 2] = host[i].z;
        }
        nb::capsule owner(buffer.get(), [](void* p) noexcept { delete[] static_cast<float*>(p); });
        auto* data = buffer.release();
        return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>(data, {count, 3}, owner));
    } else {
        using Scalar = NumpyScalar<T>;
        auto buffer = std::make_unique<Scalar[]>(count);
        for (std::size_t i = 0; i < count; ++i) {
            buffer[i] = static_cast<Scalar>(host[i]);
        }
        nb::capsule owner(buffer.get(), [](void* p) noexcept { delete[] static_cast<Scalar*>(p); });
        auto* data = buffer.release();
        return nb::cast(nb::ndarray<nb::numpy, Scalar, nb::shape<-1>>(data, {count}, owner));
    }
}

template <typename T>
HostBuffer<T>
numpy_to_host(nanobind::handle values) {
    namespace nb = nanobind;
    if constexpr (std::is_same_v<T, Float3>) {
        using Array = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
        const auto array = nb::cast<Array>(values);
        HostBuffer<T> host(array.shape(0));
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = Float3(array.data()[3 * i], array.data()[3 * i + 1], array.data()[3 * i + 2]);
        }
        return host;
    } else {
        using Array = nb::ndarray<const NumpyScalar<T>, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
        const auto array = nb::cast<Array>(values);
        HostBuffer<T> host(array.shape(0));
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = static_cast<T>(array.data()[i]);
        }
        return host;
    }
}

template <typename T>
void
write_array(DeviceBuffer<T>& destination, const HostBuffer<T>& values, const std::size_t offset) {
    if (offset > destination.size() || values.size() > destination.size() - offset) {
        throw nanobind::value_error("array exceeds the state buffer capacity");
    }
    if (!values.empty()) {
        copy_host_to_device(values.data(), raw_pointer_cast(destination.data()) + offset, values.size());
    }
}

}
