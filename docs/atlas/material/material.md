# Material

The material module describes the physical properties of a species and stores
the per-species table used by the solvers. `Material` is a tagged-union like the
other modules; `MaterialDictionary` owns the device-side table.

## Files

| File | Role |
|---|---|
| `include/atlas/material/material.h` | `ConceptMaterial` + `Material` umbrella (`DeviceVariant`) + accessors |
| `include/atlas/material/material_type.h` | `enum class MaterialType { molecule, atom, ion, neutron, solid }` |
| `include/atlas/material/molecule.h` | `Molecule` leaf |
| `include/atlas/material/atom.h` | `Atom` leaf |
| `include/atlas/material/ion.h` | `Ion` leaf |
| `include/atlas/material/neutron.h` | `Neutron` leaf |
| `include/atlas/material/solid.h` | `Solid` leaf |
| `include/atlas/material/material_dictionary.h` | `MaterialDictionary` (owns `DeviceBuffer<Material>`) + `Builder` |
| `src/atlas/material/material_dictionary.cu` | `MaterialDictionary::Builder` implementation |

## `Material`

`Material` is a **`DeviceVariant`** over one leaf per `MaterialType`
(`Molecule`, `Atom`, `Ion`, `Neutron`, `Solid`), each in its own header. Keeping
them as distinct types lets the per-type physics diverge — `Solid` already does.

Each leaf keeps its state private and exposes it through getters. Every property
a leaf carries is **always present** — a plain `float`, not
`std::optional<float>` — and is supplied through the leaf's constructor.

```cpp
class Molecule {                   // Atom / Ion / Neutron look the same
public:
    Molecule() = default;
    Molecule(float mass,
             float translational_energy,
             float rotational_energy,
             float vibrational_energy);

    float mass() const;
    float translational_energy() const;
    float rotational_energy() const;
    float vibrational_energy() const;

private:
    float _mass {};
    float _translational_energy {};
    float _rotational_energy {};
    float _vibrational_energy {};
};

class Solid {                      // no internal energy at all
public:
    explicit Solid(float mass);
    float mass() const;
    float translational_energy() const;   // throws std::runtime_error
    float rotational_energy() const;      // throws std::runtime_error
    float vibrational_energy() const;     // throws std::runtime_error
private:
    float _mass {};
};

template <typename M>
concept ConceptMaterial = requires(const M material) {
    { material.mass() } -> std::same_as<float>;
    { material.translational_energy() } -> std::same_as<float>;
    { material.rotational_energy() } -> std::same_as<float>;
    { material.vibrational_energy() } -> std::same_as<float>;
};

class Material {
public:
    MaterialType type;
    union { Molecule molecule; Atom atom; /* … */ };
    float mass() const noexcept;          // device-capable, DeviceVariant::visit
    float translational_energy() const;   // host-only, throws for Solid
    float rotational_energy() const;      // host-only, throws for Solid
    float vibrational_energy() const;     // host-only, throws for Solid
};
```

Notes:

- The old `MaterialProperties` had both `mass` and `molecular_mass`; the field is
  now a single **`mass`** (the former `molecular_mass`, renamed). All other
  legacy fields were dropped in this pass.
- The leaves are trivially copyable (plain `float`s), so `Material` is trivially
  copyable and lives in a `DeviceBuffer<Material>`.
- **`mass()` is the only device-side accessor.** It dispatches through
  `DeviceVariant::visit`, which is `noexcept` and `__host__ __device__` — so it
  can neither host a `throw` nor call a host-only getter. The three energy
  accessors are therefore `ATLAS_HOST` and dispatch with a plain `switch` on
  `type`, which lets `Solid`'s getters propagate their `std::runtime_error`.
- `ConceptMaterial` constrains every leaf to the four getters; a `static_assert`
  in `material.h` checks each one. `Solid` satisfies it by declaring the three
  energy getters and throwing from them.
- Construct from a leaf: `Material m(Molecule(mass, e_tra, e_rot, e_vib));` or
  `Material m(Solid(mass));`.

## `MaterialDictionary`

`MaterialDictionary` owns the device-side species table as a
`DeviceBuffer<Material>` (indexed by species id). It is host-side and move-only
(it owns a device buffer). Build it from host-side `Material` values:

```cpp
auto dictionary = MaterialDictionary::builder()
    .with_material(Material(Molecule(m0, e_tra, e_rot, e_vib)))
    .with_materials(more_materials)             // HostBuffer<Material>
    .build();
// dictionary.materials()  -> const DeviceBuffer<Material>&
// dictionary.size(), dictionary.empty()
```

`MaterialDictionaryHostPtr` (a `host_shared_ptr<MaterialDictionary>`) is what
consumers such as `MaxwellBoltzmannGenerator::Builder::with_material_dictionary`
take to resolve per-species mass.

## Adding a leaf / field

- New material type: add a `MaterialType` value, a leaf header, a union member
  and a `DeviceVariantCase`.
- New shared field: add it to every leaf (member + constructor parameter +
  getter), extend `ConceptMaterial`, and expose a `visit`-based accessor on
  `Material`.
