# Scope and evidence for the common style

This reference distinguishes the owner's requested conventions from older
patterns still present in the tree. It is a source-reading guide, not an
inventory of every API or a claim that every file has already been normalized.
Paths below link to representative first-party code inspected for the style.

## C++/CUDA and class construction

- [.clang-format](../../../../.clang-format) specifies four-space indentation,
  type-side pointers/references, separate return types, attached braces, spaced
  braced lists, aligned assignments, and `ColumnLimit: 0`. The effective
  `BreakBeforeBraces: Attach` setting matters more than inactive custom
  `BraceWrapping` suboptions. Do not reinterpret the file as Allman braces.
- [Coding guide](../../../../docs/contributing/coding-style.md) defines concise
  names, `<noun>_count`, direct flow, meaningful helper roles, member organization,
  and comments about contracts and reasons. Follow the nearest family for more
  specific member ordering; do not reorder public value fields or ABI layouts
  mechanically to match a general type-grouping preference.
- [VolumeSink header](../../../../include/atlas/sink/volume_sink.h) and
  [implementation](../../../../src/atlas/sink/volume_sink.cu) show a leaf,
  nested Builder declaration, constructor initialization, `with_*` chaining,
  optional staged state, validation, and consuming build/reset behavior.
- [Unit header](../../../../include/atlas/unit/unit.h) and
  [implementation](../../../../src/atlas/unit/unit.cu) show direct optional motion
  members and public pose operations. The owner explicitly rejected an extra
  motion-wrapper hierarchy, mismatched Builder/constructor state, hardcoded
  smoothing, and unnecessary float/double conversions for this functionality.
- [TriangleMesh](../../../../include/atlas/geometry/triangle_mesh.h) has a
  reusable const builder; do not copy Unit's reset behavior into it by habit.
  [Float3](../../../../include/atlas/math/vector/float3.h) is a public value type;
  its fields need not become private to resemble an owning class.
- [Constants](../../../../include/atlas/math/constants.h), existing Unit/Sync
  operations, and [array conversion](../../../../bindings/python/atlas/detail/array.h)
  are reuse points. Reusing an API is a writing preference; the architecture
  skills describe its ownership and host/device constraints.
- [Camera](../../../../src/interactive/rendering/camera.h) and the standard
  [intake example](../../../../examples/interactive/02_intake/main.cpp) demonstrate
  constructor-based initial setup. Existing camera cursor state uses `double`;
  a single-precision physics preference does not change external-library types.

## Examples, bindings, and Python

The latest owner instruction makes the intake example's direct entry-point
style the standard across `examples/cpp`, `examples/interactive`, and
`examples/python`. Existing C++/Python parsers, `build_system` helpers, and
summary printing are legacy patterns, not separate approved example standards.
Preserve frontend APIs while applying the shared writing style.

- The bindings use **nanobind**. Follow [Unit registration](../../../../bindings/python/atlas/unit/unit.h):
  `register_*`, `namespace nb = nanobind`, fluent `.def` / `.def_prop_ro`, and
  named arguments such as `"axis"_a`. Keep Python class names PascalCase and
  methods/properties/enum members snake_case. Do not introduce a second binding
  library or change exported spellings as a style cleanup.
- Python implementations use four spaces, snake_case functions/locals,
  `_internal` names, uppercase constants, and normal separation of top-level
  definitions. See [engine selection](../../../../bindings/python/atlas/_engine.py),
  [package exports](../../../../bindings/python/atlas/__init__.py), and
  [version tool](../../../../scripts/bump_version.py). Preserve a file's established
  annotation/import style; there is no basis for a repository-wide type-hint or
  `os.path`-to-`pathlib` migration in a local edit.
- Keep array types meaningful: `np.float32` physical vectors, `np.uint64`
  species, and `np.int64` mesh indices as used by the bindings. Preserve actual
  ABI casts, shape validation, and ownership-transfer code rather than applying
  the minimal-example style to the binding implementation.
- [Interactive service](../../../../src/interactive/main.cpp), Python engine
  selection, and packaging tools deliberately contain CLI/error handling.
  These are applications or tools, not the simplified simulation examples.

## Tests and benchmarks

- [Core unit tests](../../../../tests/atlas/unit/unit_tests.cpp) and
  [rendering tests](../../../../tests/interactive/rendering_tests.cpp) use
  GoogleTest `TEST(Subject, Behavior)`. Use `ASSERT_*` for prerequisites and
  `EXPECT_*` for observations; choose float equality or tolerance by the contract.
  Fixtures and namespace-local helpers have a real reuse role here. They are
  not prohibited by the examples' direct-entry-point convention.
- [Python tests](../../../../tests/python/test_math.py) use `unittest.TestCase`,
  `test_snake_case`, `subTest`, and explicit exception assertions. Reuse
  [test support](../../../../tests/python/support.py) for repeated setup.
- [Benchmarks](../../../../benchmarks/atlas/smoke/main.cpp) retain Google Benchmark's
  `BM_*` naming and result-consumption APIs. Timers and reporting integral to a
  benchmark are not temporary debug instrumentation to remove.

## Build files, scripts, and configuration

- [CMake libraries](../../../../cmake/AtlasLibraries.cmake) and
  [example CMake](../../../../examples/interactive/02_intake/CMakeLists.txt) use
  lowercase commands, `ATLAS_*` project options, snake_case local variables,
  quoted paths, and target-scoped dependencies/properties. Follow neighboring
  multiline argument indentation and `if (...)` / `endif ()` spelling.
- [Presets](../../../../CMakePresets.json) and runtime JSON use two spaces and
  strict JSON. Use supported engine field names and meaningful target names;
  don't proliferate per-case presets. [reference.jsonc](../../../../examples/interactive/00_template/reference.jsonc)
  is an annotated reference, not a runtime file.
- [Wheel script](../../../../scripts/build_wheels.sh) uses a Bash shebang,
  `set -euo pipefail`, quoted expansions, arrays, snake_case functions/local
  variables, uppercase environment knobs, and temporary-directory cleanup.
  Meaningful shell helpers and user-facing diagnostics belong here.
- [CI YAML](../../../../.github/workflows/ci-core.yml) uses structured jobs/steps
  with two-space indentation and block scalars for shell commands.
  [Dockerfile](../../../../Dockerfile) uses uppercase instructions and named
  stages. Match these formats rather than applying C++ whitespace conventions.
- [Tools](../../../../tools/check_device_inline.py) include older utility styles
  and stale documentation links. Inspect actual behavior before copying a
  convention; do not treat an incidental old script as the universal standard.

Exclude `external/` vendor code, `assets/` CAD/material data, build outputs,
generated headers such as `include/atlas/atlas.h`, and generated artifacts from
style normalization. Their presence was considered when scoping this review;
they are not evidence of the owner's first-party coding preferences.
