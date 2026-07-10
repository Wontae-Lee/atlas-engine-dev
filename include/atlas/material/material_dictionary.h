#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/memory/memory.h>

#include <cstddef>

namespace atlas {

/**
 * @brief Owns the device-side per-species material table.
 *
 * A `MaterialDictionary` holds a `DeviceBuffer<Material>` indexed by species id:
 * element `i` is the `Material` for species `i`. Solvers read it on the device
 * (e.g. `materials[species_id].mass()`); consumers such as
 * `MaxwellBoltzmannGenerator` resolve per-species mass through it.
 *
 * It is **move-only**: copying is deleted because it owns a `DeviceBuffer`
 * (`thrust::device_vector`), whose copy would be an implicit host→device→host
 * round trip. Construct one with the nested `Builder`. It is a host-side type;
 * only the buffer it holds lives in device memory.
 *
 * @see MaterialDictionary::Builder, Material
 */
class MaterialDictionary final {
public:
    /** @brief Builder that accumulates host-side materials and uploads them. */
    class Builder;

public:
    /** @brief Constructs an empty dictionary holding no materials. */
    MaterialDictionary() = default;

    /** @brief Deleted: the owned `DeviceBuffer` is not cheaply copyable. */
    MaterialDictionary(const MaterialDictionary&) = delete;

    /** @brief Moves the material table out of `other`, leaving it empty. */
    MaterialDictionary(MaterialDictionary&&) noexcept = default;

    /** @brief Releases the device material table. */
    ~MaterialDictionary() = default;

    /** @brief Deleted: copy assignment is unavailable for the same reason as copy construction. */
    MaterialDictionary&
    operator=(const MaterialDictionary&)
        = delete;

    /**
     * @brief Move-assigns the material table from `other`.
     * @return Reference to this dictionary.
     */
    MaterialDictionary&
    operator=(MaterialDictionary&&) noexcept = default;

    /**
     * @brief Constructs a dictionary that takes ownership of an existing device table.
     *
     * Host-only. Typically called by `Builder::build()` rather than directly.
     *
     * @param materials Device buffer of per-species materials; moved in.
     */
    ATLAS_HOST explicit MaterialDictionary(DeviceBuffer<Material> materials) noexcept;

    /**
     * @brief Returns a fresh, empty `Builder`.
     * @return A default-constructed builder. Host-only.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Read access to the device material table.
     * @return Const reference to the species-indexed device buffer. Host-only handle.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Material>&
    materials() const noexcept {
        return _materials;
    }

    /**
     * @brief Mutable access to the device material table.
     * @return Reference to the species-indexed device buffer. Host-only handle.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Material>&
    materials() noexcept {
        return _materials;
    }

    /**
     * @brief Number of species (materials) in the table.
     * @return The element count of the device buffer. Host-only.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept {
        return _materials.size();
    }

    /**
     * @brief Whether the table holds no materials.
     * @return `true` if the device buffer is empty. Host-only.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    empty() const noexcept {
        return _materials.empty();
    }

private:
    DeviceBuffer<Material> _materials; ///< Species-indexed device material table.
};

/**
 * @brief Accumulates host-side `Material`s and uploads them into a `MaterialDictionary`.
 *
 * Materials are appended in order; their positions become their species ids.
 * `build()` validates (at least one material required), copies the staged
 * host buffer into device memory, and clears the staging buffer so the builder
 * can be reused. Follows the project-wide builder pattern.
 *
 * @see MaterialDictionary
 */
class MaterialDictionary::Builder final {
public:
    /** @brief Constructs a builder with no staged materials. */
    Builder() = default;

    /**
     * @brief Appends a single material, assigning it the next species id.
     * @param material The material to stage (copied into the host buffer).
     * @return `*this`, for call chaining.
     */
    ATLAS_HOST Builder&
    with_material(const Material& material);

    /**
     * @brief Appends a batch of materials in order.
     * @param materials Host buffer whose elements are copied onto the staged list.
     * @return `*this`, for call chaining.
     */
    ATLAS_HOST Builder&
    with_materials(const HostBuffer<Material>& materials);

    /**
     * @brief Validates and uploads the staged materials into a new dictionary.
     *
     * Copies the staged host buffer into a fresh `DeviceBuffer<Material>` and
     * clears the staging buffer, so the builder is left empty and reusable.
     *
     * @return A `MaterialDictionary` owning the uploaded table.
     * @throws std::runtime_error if no material has been staged.
     */
    ATLAS_NODISCARD ATLAS_HOST MaterialDictionary
    build();

    /**
     * @brief Builds and wraps the dictionary in a host-side shared pointer.
     *
     * Convenience for consumers (e.g.
     * `MaxwellBoltzmannGenerator::Builder::with_material_dictionary`) that take
     * a `MaterialDictionaryHostPtr`.
     *
     * @return A `host_shared_ptr` owning the built dictionary.
     * @throws std::runtime_error if no material has been staged.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaterialDictionary>
    make_host_shared();

private:
    /**
     * @brief Throws unless at least one material has been staged.
     * @throws std::runtime_error when the staging buffer is empty.
     */
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<Material> _materials; ///< Staged host-side materials awaiting upload.
};

/** @brief Shared-ownership handle to a host-side `MaterialDictionary`. */
using MaterialDictionaryHostPtr = atlas::host_shared_ptr<MaterialDictionary>;

/** @brief Shared-ownership handle to a device-resident `MaterialDictionary`. */
using MaterialDictionaryDevicePtr = atlas::device_shared_ptr<MaterialDictionary>;

}