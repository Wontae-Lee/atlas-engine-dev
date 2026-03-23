#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class CircleLayer final : public GeometryLayer<T> {
public:
    class Builder;

public:
    ATLAS_HOST explicit CircleLayer(
        const atlas::UnitHostPtr<T>& unit,
        int segments = 64);

    static Builder
    builder() noexcept;

    ~CircleLayer() override = default;

protected:
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    int _segments;
};

template <typename T>
class CircleLayer<T>::Builder {
public:
    Builder() = default;

    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    Builder&
    with_segments(int segments) noexcept;

    CircleLayer
    build() const;

    std::shared_ptr<CircleLayer>
    make_shared() const;

private:
    void
    validate() const;

private:
    atlas::UnitHostPtr<T> _unit = nullptr;
    int _segments = 64;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/circle_layer.hpp>

#endif
