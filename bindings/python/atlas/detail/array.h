#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace atlas::python {

template <typename T>
using NumpyScalar = std::conditional_t<std::is_same_v<T, std::size_t>, std::uint64_t, T>;

static_assert(std::is_trivially_copyable_v<Float3>);
static_assert(std::is_standard_layout_v<Float3>);
static_assert(sizeof(Float3) == 3 * sizeof(float));
static_assert(alignof(Float3) == alignof(float));
static_assert(offsetof(Float3, x) == 0);
static_assert(offsetof(Float3, y) == sizeof(float));
static_assert(offsetof(Float3, z) == 2 * sizeof(float));

template <typename T, typename Copy>
nanobind::object
numpy_snapshot(const std::size_t count, Copy&& copy) {
    namespace nb = nanobind;
    if constexpr (std::is_same_v<T, Float3>) {
        auto buffer = std::make_unique<Float3[]>(count);
        copy(buffer.get());
        nb::capsule owner(buffer.get(), [](void* pointer) noexcept {
            delete[] static_cast<Float3*>(pointer);
        });
        auto* data = reinterpret_cast<float*>(buffer.release());
        return nb::cast(nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>(data, { count, 3 }, owner));
    } else {
        using Scalar = NumpyScalar<T>;
        auto buffer  = std::make_unique<Scalar[]>(count);
        if constexpr (std::is_same_v<T, Scalar>) {
            copy(buffer.get());
        } else {
            auto native = std::make_unique<T[]>(count);
            copy(native.get());
            std::transform(native.get(), native.get() + count, buffer.get(), [](const T value) {
                return static_cast<Scalar>(value);
            });
        }
        nb::capsule owner(buffer.get(), [](void* pointer) noexcept {
            delete[] static_cast<Scalar*>(pointer);
        });
        auto* data = buffer.release();
        return nb::cast(nb::ndarray<nb::numpy, Scalar, nb::shape<-1>>(data, { count }, owner));
    }
}

template <typename T>
nanobind::object
numpy_copy_device(const DeviceBuffer<T>& source, const std::size_t count) {
    if (count > source.size()) {
        throw nanobind::value_error("state buffer is shorter than the requested count");
    }
    return numpy_snapshot<T>(count, [&](T* destination) {
        copy_device_to_host(source, destination, count);
    });
}

template <typename T>
nanobind::object
numpy_copy_device(const T* source, const std::size_t count) {
    return numpy_snapshot<T>(count, [&](T* destination) {
        copy_device_to_host(source, destination, count);
    });
}

template <typename Buffer>
nanobind::object
numpy_copy_host(const Buffer& source, const std::size_t count) {
    using T = typename Buffer::value_type;
    if (count > source.size()) {
        throw nanobind::value_error("snapshot buffer is shorter than the requested count");
    }
    return numpy_snapshot<T>(count, [&](T* destination) {
        std::copy_n(source.begin(), count, destination);
    });
}

template <typename T>
std::size_t
numpy_size(nanobind::handle values) {
    namespace nb = nanobind;
    if constexpr (std::is_same_v<T, Float3>) {
        using Array = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
        return nb::cast<Array>(values).shape(0);
    } else {
        using Array = nb::ndarray<const NumpyScalar<T>, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
        return nb::cast<Array>(values).shape(0);
    }
}

inline void
write_float3_array(DeviceBuffer<Float3>& destination,
                   const float* values,
                   const std::size_t count,
                   const std::size_t offset) {
    if (offset > destination.size() || count > destination.size() - offset) {
        throw nanobind::value_error("array exceeds the state buffer capacity");
    }
    if (count == 0) return;
    auto* output = raw_pointer_cast(destination.data()) + offset;
    copy_host_to_device(reinterpret_cast<const std::uint8_t*>(values),
                        reinterpret_cast<std::uint8_t*>(output),
                        count * sizeof(Float3));
}

template <typename T>
void
write_array(DeviceBuffer<T>& destination, nanobind::handle values, const std::size_t offset) {
    namespace nb = nanobind;
    const std::size_t count = numpy_size<T>(values);
    if (offset > destination.size() || count > destination.size() - offset) {
        throw nb::value_error("array exceeds the state buffer capacity");
    }
    if (count == 0) return;

    T* output = raw_pointer_cast(destination.data()) + offset;
    if constexpr (std::is_same_v<T, Float3>) {
        using Array      = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
        const auto array = nb::cast<Array>(values);
        write_float3_array(destination, array.data(), count, offset);
    } else {
        using Scalar     = NumpyScalar<T>;
        using Array      = nb::ndarray<const Scalar, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
        const auto array = nb::cast<Array>(values);
        if constexpr (std::is_same_v<T, Scalar>) {
            copy_host_to_device(array.data(), output, count);
        } else {
            auto native = std::make_unique<T[]>(count);
            std::transform(array.data(), array.data() + count, native.get(), [](const Scalar value) {
                return static_cast<T>(value);
            });
            copy_host_to_device(native.get(), output, count);
        }
    }
}

}
