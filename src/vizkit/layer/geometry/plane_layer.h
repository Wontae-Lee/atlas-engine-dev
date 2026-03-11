#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

    template <typename T>
    class PlaneLayer final : public GeometryLayer<T> {
    public:
        class Builder;

    public:
        ATLAS_HOST explicit PlaneLayer(
            const atlas::UnitHostPtr<T>& unit,
            T extent = T(5));

        static Builder builder() noexcept;

        ~PlaneLayer() override = default;

    protected:
        void
        build_geometry(std::vector<Vector3<T>>& positions) override;

    private:
        T _extent;
    };

    template <typename T>
    class PlaneLayer<T>::Builder {
    public:
        Builder() = default;

        Builder& with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;
        Builder& with_extent(T extent) noexcept;

        PlaneLayer build() const;
        std::shared_ptr<PlaneLayer> make_shared() const;

    private:
        void validate() const;

    private:
        atlas::UnitHostPtr<T> _unit = nullptr;
        T _extent = T(5);
    };

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/plane_layer.hpp>

#endif