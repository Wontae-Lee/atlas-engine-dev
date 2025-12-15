#pragma once
#include <atlas/buffer/device_buffer.h>

namespace atlas::system {

template <typename T>
class Field {
public:
    Field()  = default;
    ~Field() = default;

private:
    DeviceBuffer<T> d_temperature;
};
}