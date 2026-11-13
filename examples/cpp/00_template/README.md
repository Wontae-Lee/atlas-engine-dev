# 00 — Native C++ template

Copy this directory to start a native Atlas example. `main.cu` is a complete,
small simulation: two particles move at 10 m/s along x, with no collisions,
emission, or removal. Its structure separates construction (`build_system`),
stepping and reporting (`run`), and command-line handling (`main`).

## Files and build target

| File | Purpose |
|---|---|
| `main.cu` | Runnable starting point; edit its `build_system()` to define your simulation |
| `CMakeLists.txt` | Declares `atlas_example_cpp_00_template` and attaches it to `atlas_examples_cpp` |
| `reference.h` | Broader API construction examples for materials, geometries, optional states, generators, boundaries, solvers, and codecs |

`reference.h` is included so the compiler checks the API examples, but its
`atlas_examples::template_case::build_system()` is not called. It combines
alternative components for reference; copy the pieces you need into your own
`build_system()` rather than treating that combination as a physical benchmark.

## Configure, build, and run

Run from the repository root after initializing the repository submodules and
installing the [native dependencies](../../../docs/contributing/dependencies.md):

```bash
cmake -S . -B build/examples-tbb -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_HOST_COMPILER=native \
    -DATLAS_EXAMPLES=ON -DATLAS_PYTHON=OFF \
    -DATLAS_BENCHMARKS=OFF -DATLAS_GOOGLE_TEST=OFF -DBUILD_TESTING=OFF \
    -DATLAS_INTERACTIVE=OFF
cmake --build build/examples-tbb --target atlas_example_cpp_00_template
./build/examples-tbb/examples/cpp/00_template/atlas_example_cpp_00_template 40
```

The optional argument is the number of complete `System::update()` calls.
Omitting it means 40. `0` constructs and reports the initial state. `--help`
prints the invocation. Negative, non-integer, overflowing, and extra arguments
fail before constructing a system. No case-name argument is accepted.

After 40 steps, both particles remain alive at x approximately `-0.460000`;
their y coordinates remain `-0.1` and `0.1`, and z remains zero. Floating-point
rounding can affect the last printed digits. The timestep is `1e-4` seconds.
The template intentionally has no sink: extending a run does not automatically
remove particles that leave the grid.

The `.cu` source also builds as ordinary C++ in native TBB configurations, which
need no CUDA toolkit. For CUDA, configure a separate directory with
`ATLAS_DEVICE_SYSTEM=CUDA`; nvcc compiles the same source, and execution needs a
GPU. See the [build guide](../../../docs/contributing/build.md) for toolchains.

## How to write the simulation

1. **Choose materials.** Build a `MaterialDictionary` before constructing Fluid.
   The template has one molecule with mass `4.65e-26` kg. Its dictionary index is
   `0`; every supplied species value must refer to a material in that dictionary.
   Use physically meaningful properties when adding a collision solver.
2. **Allocate and initialize particles.** `buffer_size` is capacity, while
   `particle_count` is the live prefix. The template reserves 1024 slots and
   initializes two. `with_position`, `with_velocity`, and `with_species` accept
   host buffers containing exactly the live count. Builders copy those values
   into the configured backend's buffers. Keep position in metres, velocity in
   metres per second, and statistical weight positive. If a source will add
   particles, leave enough capacity; emission stops when capacity is exhausted.
3. **Define the grid.** Set the Universe bounds and positive cell size. The
   template uses `[-1, 1]^3` and `0.25` m cells. Alternatively use
   `with_geometry(geometry)` to derive bounds. Grid bounds organize spatial
   searches; add explicit boundaries for reflection or removal.
4. **Add optional physics.** Pass `DsmcSolver::builder().make_host_shared()` to
   `with_solver` when you need particle collisions. A `Source` defines spawn
   positions and a paired `Generator` defines velocity/species; add both with
   `with_emitter`. Build a `Unit` from `Geometry` and `Sync` for source, collider,
   and sink boundaries. Add colliders with `with_collider` and sinks with
   `with_sink`. `reference.h` shows the builder shapes, and
   [01_cylinder](../01_cylinder/README.md) supplies a complete working combination.
5. **Transfer ownership to System.** `Fluid` and `Universe` use unique host
   pointers; pass them with `std::move`. Those local pointers must not be used
   afterward. Access the owned objects through `system.fluid()` and
   `system.universe()`. Use `build()` once per configured System builder, because
   it moves the required objects into the resulting system.
6. **Advance and observe.** `with_dt` sets a positive timestep in seconds.
   Call `system.update()` once per desired step; `system.step()` counts completed
   updates. Read only the leading `particle_count` values of each Fluid state.
   Construct a `HostBuffer` from backend iterators for a bulk transfer before
   host analysis, as the template does. Do not dereference CUDA device pointers
   in host code. Keep output/export policy in your example; use
   `system.save(directory)` only when you want explicit snapshots.

The full contracts for [System](../../../docs/atlas/system/system.md),
[Fluid](../../../docs/atlas/fluid/fluid.md), and
[Universe](../../../docs/atlas/universe/universe.md) describe ownership,
validation, and state storage. Core builders report invalid configurations; the
entry point prints exceptions and exits with a nonzero status.

## Create the next numbered example

1. Copy the directory:

   ```bash
   cp -R examples/cpp/00_template examples/cpp/02_my_simulation
   ```

2. Change the first line of the copied `CMakeLists.txt` to:

   ```cmake
   set(example_target atlas_example_cpp_02_my_simulation)
   ```

   Keep the native/nvcc language selection, Core linkage, optional shared-library
   linkage, and `add_dependencies` call. This gives the example its own build
   target and includes it in `atlas_examples_cpp` automatically.

3. Add `add_subdirectory(02_my_simulation)` to
   `examples/cpp/CMakeLists.txt`. Keep the directories in numeric order and use
   the same number and name for corresponding Python/Interactive examples.
4. Edit `build_system()` and reporting in the copied `main.cu`; rename the
   `00_template` diagnostic label. Remove `reference.h` and its include if you
   do not need the API catalog. Replace the copied README with your model,
   parameters, assumptions, command, and expected output.
5. Re-run the configure command, then build and run the new target:

   ```bash
   cmake --build build/examples-tbb --target atlas_example_cpp_02_my_simulation
   ./build/examples-tbb/examples/cpp/02_my_simulation/atlas_example_cpp_02_my_simulation 40
   ```

Check a zero-step run for construction, a short positive run for state changes,
and a malformed argument for a controlled failure. Keep the default run small.
Update the [example index](../../README.md) when publishing the new example.
