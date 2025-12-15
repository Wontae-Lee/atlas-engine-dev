#pragma once

#include  <atlas/buffer/device_buffer.h>
#include <atlas/memory/memory.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas {
namespace system {
    template <typename T>
    struct DomainDeviceProbe {
        T* d_temperature;
        Vector3<T>* d_field_force;

        Vector3<T> lower_corner;
        Vector3<T> upper_corner;
        Vector3<int> grid_size;

        T cell_size;
        T cell_volume;
        T inv_h;

        int num_of_cells = 0;
    };

    template <typename T>
    class Domain {
    public:
        Domain(const Vector3<T>& lower_corner, const Vector3<T>& upper_corner, T cell_size)
            : _lower_corner(lower_corner),
              _upper_corner(upper_corner),
              _cell_size(cell_size) {
            _cell_volume  = _cell_size * _cell_size * _cell_size;
            _inv_h        = T(1) / _cell_size;
            _grid_size    = atlas::math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<int>{ 1, 1, 1 };
            _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;
            d_temperature.resize(_num_of_cells, T(300));
            d_field_force.resize(_num_of_cells, Vector3<T>{ T(0), T(0), T(0) });
        }

        ~Domain() = default;

        ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>
        make_device_probe() noexcept {
            DomainDeviceProbe<T> probe;
            probe.d_temperature = atlas::raw_pointer_cast(d_temperature.data());
            probe.d_field_force = atlas::raw_pointer_cast(d_field_force.data());
            probe.lower_corner  = _lower_corner;
            probe.upper_corner  = _upper_corner;
            probe.grid_size     = _grid_size;
            probe.cell_size     = _cell_size;
            probe.cell_volume   = _cell_volume;
            probe.inv_h         = _inv_h;
            probe.num_of_cells  = _num_of_cells;
            return probe;
        }

    private:
        DeviceBuffer<T> d_temperature;
        DeviceBuffer<Vector3<T>> d_field_force;

        Vector3<T> _lower_corner;
        Vector3<T> _upper_corner;
        Vector3<int> _grid_size{ 1, 1, 1 };
        T _cell_size      = T(1);
        T _cell_volume    = T(1);
        T _inv_h          = T(1);
        int _num_of_cells = T(1);
    };
}

template <typename T>
using Domain = system::Domain<T>;

template <typename T>
using DomainHostPtr = atlas::host_shared_ptr<system::Domain<T>>;

template <typename T>
using DomainDevicePtr = atlas::device_shared_ptr<system::Domain<T>>;
}