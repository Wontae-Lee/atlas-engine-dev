# Material

The material module describes the physical constants of a species and owns the
per-species table the solvers read on the device. `Material` is a tagged union
(one leaf per `MaterialType`); `MaterialDictionary` owns a
`DeviceBuffer<Material>` indexed by species id. It is not a step in
`System::update()` — it is reference data attached to a `Fluid`
(`Fluid::materials()` → `MaterialDictionaryHostPtr`) and consulted during the
**solve** phase (the DSMC kernels read collision parameters) and once at
**generate** time (`MaxwellBoltzmannGenerator` caches per-species mass).

## Files

| File | Role |
|---|---|
| `include/atlas/material/material_type.h` | `enum class MaterialType : int { molecule, atom, ion, neutron, solid }` — the discriminant |
| `include/atlas/material/molecule.h` | `Molecule` leaf — eight `float`s (polyatomic) |
| `include/atlas/material/atom.h` | `Atom` leaf — same eight fields (monatomic) |
| `include/atlas/material/ion.h` | `Ion` leaf — same eight fields (charged; charge not modelled) |
| `include/atlas/material/neutron.h` | `Neutron` leaf — same eight fields (neutral nuclear) |
| `include/atlas/material/solid.h` | `Solid` leaf — mass only; all other getters return `1.0f` |
| `include/atlas/material/material.h` | `ConceptMaterial`, the `Material` umbrella (`DeviceVariant`), the eight visitor functors, `MaterialVariant` |
| `include/atlas/material/material_dictionary.h` | `MaterialDictionary` (owns `DeviceBuffer<Material>`) + nested `Builder` |
| `src/atlas/material/material_dictionary.cu` | `MaterialDictionary` / `Builder` out-of-line definitions |

## The leaves

Four of the five leaves — `Molecule`, `Atom`, `Ion`, `Neutron` — are
**byte-for-byte identical**: eight `float`s with the same constructor and eight
getters. They are kept as distinct types purely so per-species physics can
diverge later; today only `Solid` diverges. Each holds nothing but `float`s, so
each is trivially copyable and can be captured by value into a device lambda.

```cpp
Molecule(float mass,                    // kg
         float translational_energy,    // J
         float rotational_energy,       // J
         float vibrational_energy,      // J
         float reference_diameter,      // m   (VHS/VSS d_ref)
         float reference_temperature,   // K   (VHS/VSS T_ref)
         float viscosity_index,         // -   omega; 0.5 = hard sphere
         float scattering_parameter);   // -   alpha; 1.0 = isotropic (VHS)
```

The last four fields are the Variable Hard Sphere / Variable Soft Sphere
(VHS/VSS) cross-section parameters the DSMC kernels consume. `_viscosity_index`
and `_scattering_parameter` carry in-class defaults `0.5f` / `1.0f` so a
default-constructed leaf is a hard sphere rather than a degenerate one.

`Solid` is the wall/boundary species. It stores only `mass()` and returns a
constant `1.0f` from every other getter (see **Deliberately absent**).

## `ConceptMaterial` and the umbrella

`ConceptMaterial<M>` requires all eight getters, each returning exactly `float`.
Five `static_assert`s in `material.h` check every leaf conforms — adding a
property means adding a getter to the concept *and* to every leaf, or the build
breaks.

```cpp
template <typename M>
concept ConceptMaterial = requires(const M m) {
    { m.mass() }                   -> std::same_as<float>;
    { m.translational_energy() }   -> std::same_as<float>;
    { m.rotational_energy() }      -> std::same_as<float>;
    { m.vibrational_energy() }     -> std::same_as<float>;
    { m.reference_diameter() }     -> std::same_as<float>;
    { m.reference_temperature() }  -> std::same_as<float>;
    { m.viscosity_index() }        -> std::same_as<float>;
    { m.scattering_parameter() }   -> std::same_as<float>;
};
```

`Material` is a **`DeviceVariant`** umbrella — the trivially-copyable variant,
*not* `HostVariant` — because no leaf owns a `DeviceBuffer`; the whole point is
that a `Material` lives in a `DeviceBuffer<Material>` and is read inside device
kernels. It holds `MaterialType type` plus a raw `union` of the five leaves; the
special members are `= default` and trivially copyable *only because* every leaf
is. The comment in the header is explicit: do not add a leaf that owns a
resource.

```cpp
class Material final {
public:
    MaterialType type = MaterialType::molecule;
    union { Molecule molecule; Atom atom; Ion ion; Neutron neutron; Solid solid; };

    ATLAS_ALL_DEVICE Material() noexcept;                 // activates molecule
    template <typename Payload> explicit Material(const Payload&) noexcept;

    float mass() const noexcept;                          // and the seven others
};
```

Every getter dispatches through `MaterialVariant::visit(*this, Functor{}, 0.0f)`:
a plain visitor functor (e.g. `MaterialMass`) is invoked on whichever leaf
`type` selects. They are functors rather than lambdas specifically because nvcc
forbids an extended `__host__ __device__` lambda in class scope. The `0.0f` is a
fallback for a tag that matches no case, which cannot happen for a normalized
tag — `DeviceVariant::normalize` folds any out-of-range `type` back to
`molecule`. The leaf-payload constructor deduces the tag from the argument type
via `MaterialVariant::construct_payload`, so `Material m(Solid(mass))` just
works; a non-leaf argument trips a `static_assert` inside `DeviceVariant`.

Which getters actually feed physics (all reads are in the DSMC kernels under
`include/atlas/solver/dsmc/kernel/`, plus mass in the generator):

| Property | Read by |
|---|---|
| `mass` | `dsmc_scatter.h`, `variable_hard_sphere_kernel.h`, `maxwell_boltzmann_generator.cu:124` |
| `reference_diameter` | `hard_sphere_kernel.h`, `variable_hard_sphere_kernel.h` |
| `reference_temperature`, `viscosity_index` | `variable_hard_sphere_kernel.h` |
| `scattering_parameter` | `variable_soft_sphere_kernel.h` |
| `translational/rotational/vibrational_energy` | **nothing** (see Not implemented) |

## `MaterialDictionary`

Owns the device species table as a `DeviceBuffer<Material>` where element `i` is
species `i`. It is **move-only** — copy is deleted because copying a
`thrust::device_vector` is a host→device→host round trip. It is a host-side
handle; only the buffer it wraps lives in device memory. Build it through the
nested `Builder`, which stages `Material`s in a `HostBuffer<Material>` (append
order = species id), then `build()` validates (at least one material, else
`std::runtime_error`), range-constructs the device buffer in a single upload,
and clears the staging buffer so the builder is reusable.

```cpp
auto dictionary = MaterialDictionary::builder()
    .with_material(Material(Molecule(m0, /* … */)))
    .with_materials(host_buffer)          // batch append, HostBuffer<Material>
    .build();                             // or .make_host_shared()
```

`Fluid` holds a `MaterialDictionaryHostPtr`
(`host_shared_ptr<MaterialDictionary>`); the DSMC solver reads
`fluid.materials()->materials()` and, if the fluid carries no dictionary,
refuses to solve (`DsmcSolver::solve` logs and returns).
`MaxwellBoltzmannGenerator` resolves per-species mass through the same handle.

## Deliberately absent

- **`Solid`'s non-mass getters return `1.0f`.** A wall never flows, collides as
  a gas particle, or carries internal energy, so `Solid` stores only `mass`.
  `1.0f` (not `0.0f`) is chosen because the VHS/VSS math divides by and takes
  powers of these values; a nonzero finite placeholder keeps any accidental use
  well-defined instead of producing NaN/Inf. The stub must be `noexcept` and
  `__host__ __device__` — throwing or returning a sentinel is not an option on
  the device — so a `Solid` reads as junk-but-finite rather than erroring. It is
  not expected to be picked as a collision partner.
- **Charge, nuclear state, per-mode structure are not modelled.** `Ion` stores
  no field coupling and `Neutron` no nuclear data; both are eight plain `float`s
  like `Molecule`. Consistent with the engine's large-domain / low-detail aim.
- **No `std::optional` properties.** Every field is an always-present `float`
  supplied at construction, so leaves stay trivially copyable for the device.

## Not implemented

Three items exist as declarations but are never produced or consumed by engine
code. Evidence gathered by grepping `src`, `include`, `benchmarks`, and `tests`,
excluding each type's own header and the serialization round-trip.

| What | Where | Evidence | Verdict |
|---|---|---|---|
| The three internal-energy fields `translational_energy` / `rotational_energy` / `vibrational_energy` on **every** leaf | `molecule.h:150-154` etc.; accessors `material.h:145-160` | Stored, exposed, and round-tripped through `protobuf_snapshot.cpp:210-238`, but **no DSMC kernel or solver ever reads them** — the only `.*_energy()` reads in the whole tree are the serialization getters. | Accepted but ignored — reserved for future collision-energy physics; carried and serialized, never consumed. |
| `Ion` and `Neutron` leaves | `ion.h`, `neutron.h`; union members `material.h:79,81` | Constructed **only** by deserialization (`protobuf_snapshot.cpp:247,249`) and unit tests. No engine algorithm, benchmark, or example produces one; behaviour is identical to `Molecule` (no divergent physics yet). | Deliberate extension point / union-completeness case — a user *can* supply one via the builder and it will round-trip, but nothing in the shipped engine writes one. |
| `MaterialDictionaryDevicePtr` alias | `material_dictionary.h:186` | Zero references anywhere outside its own declaration; only `MaterialDictionaryHostPtr` is used (by `Fluid` and serialization). | Dead / provided for symmetry with the host handle. |

Note for contrast: `Atom` is also constructed only in tests and serialization
within this repo, but it is a realistic monatomic species used by the DSMC
kernel tests, so it is not listed as dead. The `with_materials` batch appender
and `make_host_shared` *are* exercised (serialization restore and generator
tests respectively).

## Extending

- **New leaf type.** Add a `MaterialType` enumerator, a leaf header satisfying
  `ConceptMaterial`, a union member in `Material`, and a `DeviceVariantCase` in
  the `MaterialVariant` alias binding the tag to that member. The leaf must be
  trivially copyable and own no resource, or `Material` stops being trivially
  copyable and the `= default` members become ill-formed. The
  `static_assert(ConceptMaterial<NewLeaf>)` line catches a missing getter at
  compile time.
- **New shared property.** Add the field + constructor parameter + getter to
  every leaf, add the requirement to `ConceptMaterial` (this is what forces you
  to touch all five leaves), add a visitor functor and a `visit`-based accessor
  on `Material`, and extend the serialization schema/round-trip if it must
  persist.
- **Divergent physics for `Ion`/`Neutron`.** Give the leaf its own fields and
  getter bodies; no umbrella change is needed as long as it still satisfies
  `ConceptMaterial`.
