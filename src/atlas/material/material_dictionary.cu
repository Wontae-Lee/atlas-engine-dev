#include <atlas/material/material_dictionary.h>

#include <stdexcept>
#include <utility>

namespace atlas {

MaterialDictionary::MaterialDictionary(DeviceBuffer<Material> materials) noexcept
    : _materials(std::move(materials)) {
}

MaterialDictionary::Builder
MaterialDictionary::builder() noexcept {
    return Builder {};
}

MaterialDictionary::Builder&
MaterialDictionary::Builder::with_material(const Material& material) {
    _materials.push_back(material);
    return *this;
}

MaterialDictionary::Builder&
MaterialDictionary::Builder::with_materials(const HostBuffer<Material>& materials) {
    // Append at the end so previously staged species keep their ids.
    _materials.insert(_materials.end(), materials.begin(), materials.end());
    return *this;
}

MaterialDictionary
MaterialDictionary::Builder::build() {
    validate();

    // Range-construct the device buffer from the staged host range: this is the
    // single host->device upload of the whole species table.
    MaterialDictionary dictionary(
        DeviceBuffer<Material>(_materials.begin(), _materials.end()));

    // Leave the builder empty so it can be reused without re-uploading these.
    _materials.clear();

    return dictionary;
}

atlas::host_shared_ptr<MaterialDictionary>
MaterialDictionary::Builder::make_host_shared() {
    return atlas::make_host_shared<MaterialDictionary>(build());
}

void
MaterialDictionary::Builder::validate() const {
    if (_materials.empty()) {
        throw std::runtime_error("MaterialDictionary::Builder: at least one material is required.");
    }
}

}
