#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class CylinderLayer final : public GeometryLayer<T> {
public:
    class Builder;

public:
    ATLAS_HOST explicit CylinderLayer(
        const atlas::UnitHostPtr<T>& unit,
        int slices = 32);

    static Builder
    builder() noexcept;

    ~CylinderLayer() override = default;

protected:
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    int _slices;
};

template <typename T>
class CylinderLayer<T>::Builder {
public:
    Builder() = default;

    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    Builder&
    with_slices(int slices) noexcept;

    CylinderLayer
    build() const;

    std::shared_ptr<CylinderLayer>
    make_shared() const;

private:
    void
    validate() const;

private:
    atlas::UnitHostPtr<T> _unit = nullptr;

    int _slices = 32;
};

}

#include <vizkit/layer/geometry/cylinder_layer.hpp>

#endif