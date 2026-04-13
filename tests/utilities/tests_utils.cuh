#pragma once

#include <cuda/cuda_macros.cuh>

#include <atlas/atlas.h>
#include <utilities/tests_utils.h>

namespace atlas::test::cuda {

template <typename T>
ATLAS_FORCE_INLINE void
expect_vec3_near(const atlas::Vector3<T>& lhs,
                 const atlas::Vector3<T>& rhs,
                 T eps) {
    CUDA_EXPECT_NEAR(lhs.x, rhs.x, eps);
    CUDA_EXPECT_NEAR(lhs.y, rhs.y, eps);
    CUDA_EXPECT_NEAR(lhs.z, rhs.z, eps);
}

template <typename T>
ATLAS_FORCE_INLINE std::vector<T>
to_host_vector(const atlas::DeviceBuffer<T>& buffer) {
    return ::atlas::test::copy_device_buffer(buffer);
}

template <typename T>
ATLAS_FORCE_INLINE std::vector<T>
to_host_vector(const atlas::HostBuffer<T>& buffer) {
    return std::vector<T>(buffer.begin(), buffer.end());
}

template <typename Buffer>
ATLAS_FORCE_INLINE auto
to_host_value(const Buffer& buffer, const std::size_t index)
    -> typename Buffer::value_type {
    return to_host_vector(buffer).at(index);
}

template <typename T = double>
struct IdentityOp {
    ATLAS_ALL_DEVICE T
    operator()(const T value) const {
        return value;
    }
};

template <typename T = double>
struct SquareOp {
    ATLAS_ALL_DEVICE T
    operator()(const T value) const {
        return value * value;
    }
};

template <typename T = double>
struct IncrementOp {
    ATLAS_ALL_DEVICE T
    operator()(const T value) const {
        return value + T(1);
    }
};

template <typename T = double>
struct PlusOp {
    ATLAS_ALL_DEVICE T
    operator()(const T lhs, const T rhs) const {
        return lhs + rhs;
    }
};

template <typename T = double>
struct MultiplyOp {
    ATLAS_ALL_DEVICE T
    operator()(const T lhs, const T rhs) const {
        return lhs * rhs;
    }
};

struct AcceptAllSpawnPredicate {
    ATLAS_ALL_DEVICE bool
    operator()(const atlas::geometry::GeometryOperator<double>&,
               const atlas::Vector3<double>&,
               const double) const {
        return true;
    }

    ATLAS_ALL_DEVICE bool
    operator()(const atlas::geometry::BoxGeometryOperator<double>&,
               const atlas::Vector3<double>&,
               const double) const {
        return true;
    }
};

struct GreaterThanXGeometryToleranceSpawnPredicate {
    ATLAS_ALL_DEVICE bool
    operator()(const atlas::geometry::GeometryOperator<double>&,
               const atlas::Vector3<double>& sample,
               const double tolerance) const {
        return sample.x > tolerance;
    }

    ATLAS_ALL_DEVICE bool
    operator()(const atlas::geometry::BoxGeometryOperator<double>&,
               const atlas::Vector3<double>& sample,
               const double tolerance) const {
        return sample.x > tolerance;
    }
};

}
