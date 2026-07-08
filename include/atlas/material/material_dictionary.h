#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/memory/memory.h>

#include <cstddef>
#include <vector>

namespace atlas {

// Owns the device-side table of materials (indexed by species).
class MaterialDictionary final {
public:
    class Builder;

public:
    MaterialDictionary() = default;

    MaterialDictionary(const MaterialDictionary&) = delete;

    MaterialDictionary(MaterialDictionary&&) noexcept = default;

    ~MaterialDictionary() = default;

    MaterialDictionary&
    operator=(const MaterialDictionary&)
        = delete;

    MaterialDictionary&
    operator=(MaterialDictionary&&) noexcept = default;

    ATLAS_HOST explicit
    MaterialDictionary(DeviceBuffer<Material> materials) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Material>&
    materials() const noexcept {
        return _materials;
    }

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Material>&
    materials() noexcept {
        return _materials;
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept {
        return _materials.size();
    }

    ATLAS_NODISCARD ATLAS_HOST bool
    empty() const noexcept {
        return _materials.empty();
    }

private:
    DeviceBuffer<Material> _materials;
};

class MaterialDictionary::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_material(const Material& material);

    ATLAS_HOST Builder&
    with_materials(const std::vector<Material>& materials);

    ATLAS_NODISCARD ATLAS_HOST MaterialDictionary
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaterialDictionary>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    std::vector<Material> _materials;
};

using MaterialDictionaryHostPtr = atlas::host_shared_ptr<MaterialDictionary>;

using MaterialDictionaryDevicePtr = atlas::device_shared_ptr<MaterialDictionary>;

}
