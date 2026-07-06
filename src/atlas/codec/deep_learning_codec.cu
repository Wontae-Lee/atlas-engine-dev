#include <atlas/codec/deep_learning_codec.h>

#include <atlas/logging/logging.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

DeepLearningCodec::Builder
DeepLearningCodec::builder() noexcept {
    return Builder {};
}

DeepLearningCodec::DeepLearningCodec(UniverseHostPtr domain,
                                     FluidHostPtr fluid,
                                     SearcherHostPtr searcher)
    : Codec(std::move(domain), std::move(fluid), std::move(searcher)) { }

void
DeepLearningCodec::encode() {
    static_cast<void>(make_probe());
}

void
DeepLearningCodec::decode() {
    static_cast<void>(make_probe());
}

DeepLearningCodec::Builder&
DeepLearningCodec::Builder::with_domain(UniverseHostPtr domain) noexcept {
    _domain = std::move(domain);
    return *this;
}

DeepLearningCodec::Builder&
DeepLearningCodec::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

DeepLearningCodec::Builder&
DeepLearningCodec::Builder::with_searcher(SearcherHostPtr searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

DeepLearningCodec::Builder&
DeepLearningCodec::Builder::with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept {
    _fixed_solver = std::move(fixed_solver);
    return *this;
}

DeepLearningCodec::Builder&
DeepLearningCodec::Builder::with_fixed_region(DeviceBuffer<int> fixed_region) noexcept {
    _fixed_region = std::move(fixed_region);
    return *this;
}

void
DeepLearningCodec::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "DeepLearningCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "DeepLearningCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "DeepLearningCodec::Builder: searcher must not be null.";

    const auto cell_count = static_cast<std::size_t>(_domain->cell_count());
    atlas::check<std::invalid_argument>(_fixed_solver.empty() || _fixed_solver.size() == cell_count)
        << "DeepLearningCodec::Builder: fixed_solver size must match universe cell count.";
    atlas::check<std::invalid_argument>(_fixed_region.empty() || _fixed_region.size() == cell_count)
        << "DeepLearningCodec::Builder: fixed_region size must match universe cell count.";
}

DeepLearningCodec
DeepLearningCodec::Builder::build() const {
    validate();

    auto codec = DeepLearningCodec(_domain, _fluid, _searcher);
    if (!_fixed_solver.empty()) {
        codec.set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec.set_fixed_region(_fixed_region);
    }

    return codec;
}

atlas::host_shared_ptr<DeepLearningCodec>
DeepLearningCodec::Builder::make_host_shared() const {
    validate();

    auto codec = atlas::make_host_shared<DeepLearningCodec>(_domain, _fluid, _searcher);
    if (!_fixed_solver.empty()) {
        codec->set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec->set_fixed_region(_fixed_region);
    }

    return codec;
}

}
