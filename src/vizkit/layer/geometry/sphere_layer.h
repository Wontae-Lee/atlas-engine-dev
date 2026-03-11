#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

    template <typename T>
    class SphereLayer final : public GeometryLayer<T> {
    public:
        class Builder;

    public:
        SphereLayer(const atlas::UnitHostPtr<T>& unit,int slices,int stacks);

        static Builder builder() noexcept;

    protected:
        void build_geometry(std::vector<Vector3<T>>& positions) override;

    private:
        int _slices;
        int _stacks;
    };

    template <typename T>
    class SphereLayer<T>::Builder {
    public:
        Builder& with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;
        Builder& with_slices(int slices) noexcept;
        Builder& with_stacks(int stacks) noexcept;

        SphereLayer build() const;
        std::shared_ptr<SphereLayer> make_shared() const;

    private:
        void validate() const;

    private:
        atlas::UnitHostPtr<T> _unit=nullptr;
        int _slices=32;
        int _stacks=16;
    };

}

#include <vizkit/layer/geometry/sphere_layer.hpp>

#endif