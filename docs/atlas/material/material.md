# Material

The material module describes the physical properties of a species and stores
the per-species table used by the solvers. `Material` is a tagged-union like the
other modules; `MaterialDictionary` owns the device-side table.

## Files

| File | Role |
|---|---|
| `include/atlas/material/material.h` | leaf types + `Material` umbrella (`DeviceVariant`) + accessors |
| `include/atlas/material/material_type.h` | `enum class MaterialType { molecule, atom, ion, neutron, solid }` |
| `include/atlas/material/material_dictionary.h` | `MaterialDictionary` (owns `DeviceBuffer<Material>`) + `Builder` |
| `src/atlas/material/material_dictionary.cu` | `MaterialDictionary::Builder` implementation |

## `Material`

`Material` is a **`DeviceVariant`** over one leaf per `MaterialType`
(`MoleculeMaterial`, `AtomMaterial`, `IonMaterial`, `NeutronMaterial`,
`SolidMaterial`). Every leaf currently carries the same fields; keeping them as
distinct types leaves room for per-type physics to diverge later (for example an
atom dropping rotational/vibrational energy).

```cpp
class MoleculeMaterial {           // one such leaf per MaterialType
public:
    float mass {};
    std::optional<float> translational_energy;
    std::optional<float> rotational_energy;
    std::optional<float> vibrational_energy;
};

class Material {
public:
    MaterialType type;
    union { MoleculeMaterial molecule; AtomMaterial atom; /* … */ };
    float                mass() const;                  // visit accessors
    std::optional<float> translational_energy() const;
    std::optional<float> rotational_energy() const;
    std::optional<float> vibrational_energy() const;
};
```

Notes:

- The old `MaterialProperties` had both `mass` and `molecular_mass`; the field is
  now a single **`mass`** (the former `molecular_mass`, renamed). All other
  legacy fields were dropped in this pass.
- The leaves are trivially copyable (`float` + trivially-copyable
  `std::optional<float>`), so `Material` is trivially copyable and lives in a
  `DeviceBuffer<Material>`. The `mass()`/energy accessors dispatch via
  `DeviceVariant::visit` and work on the device (verified with nvcc, including
  the `std::optional` return).
- Construct from a leaf: `Material m(MoleculeMaterial{ .mass = ... });`.

## `MaterialDictionary`

`MaterialDictionary` owns the device-side species table as a
`DeviceBuffer<Material>` (indexed by species id). It is host-side and move-only
(it owns a device buffer). Build it from host-side `Material` values:

```cpp
auto dictionary = MaterialDictionary::builder()
    .with_material(Material(MoleculeMaterial{ .mass = m0 }))
    .with_materials(more_materials)
    .build();
// dictionary.materials()  -> const DeviceBuffer<Material>&
// dictionary.size(), dictionary.empty()
```

`MaterialDictionaryHostPtr` (a `host_shared_ptr<MaterialDictionary>`) is what
consumers such as `MaxwellBoltzmannGenerator::Builder::with_material_dictionary`
take to resolve per-species mass.

## Adding a leaf / field

- New material type: add a `MaterialType` value, a leaf type, a union member and
  `DeviceVariantCase`.
- New shared field: add it to every leaf and expose a `visit`-based accessor on
  `Material`.
