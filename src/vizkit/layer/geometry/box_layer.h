#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class BoxLayer final : public GeometryLayer<T> {
public:
    class Builder;

public:
    ATLAS_HOST explicit BoxLayer(const atlas::UnitHostPtr<T>& unit);

    static Builder
    builder() noexcept;

    ~BoxLayer() override = default;

protected:
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    void
    append_local_box_lines(
        const Vector3<T>& min_corner,
        const Vector3<T>& max_corner,
        std::vector<Vector3<T>>& positions) const;
};

template <typename T>
class BoxLayer<T>::Builder {
public:
    Builder() = default;

    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    BoxLayer
    build() const;

    std::shared_ptr<BoxLayer>
    make_shared() const;

private:
    void
    validate() const;

private:
    atlas::UnitHostPtr<T> _unit = nullptr;
};
}

#include <vizkit/layer/geometry/box_layer.hpp>

#endif