# Plan 1 — Examples and Smoke Benchmark

## Goal
Rebuild `examples/` and Atlas smoke benchmark from current `main` so all frontends use the same default **cylinder** case and remain easy to extend.

## Target structure
```text
examples/
├── README.md
├── cpp/
│   ├── main.cu
│   ├── template.h
│   └── cases/
│       └── cylinder.h
├── interactive/
│   ├── main.cpp
│   ├── template.jsonc
│   └── cases/
│       └── cylinder.json
└── python/
    ├── main.py
    ├── template.py
    └── cases/
        └── cylinder.py

benchmarks/atlas/
└── smoke/
    └── main.cpp
```

## Responsibilities
- `main.*`: entry point only; parse/select case and run it. Default case = `cylinder`.
- `cases/cylinder.*`: compact, actually runnable cylinder simulation; include only settings required by that case.
- `template.*`: reference for creating new simulations. Cover the major settings currently supported by the engine, without turning the cylinder case into a full showcase.
- `examples/README.md`: single tutorial for C++, Interactive, and Python. Explain build/run, file roles, default cylinder case, use of templates, and how to add the same new case to all three frontends.

## Template coverage
Show the major current configuration/API branches where applicable:
- `dt`
- Fluid: capacity/count/weight, materials, initial particle states
- Materials: molecule, atom, ion, neutron, solid
- Universe: bounds/geometry, cell size, optional cell states
- DSMC solver/kernel settings
- Sources: surface/volume
- Generators: uniform, jittering, Maxwell-sigma, Maxwell-Boltzmann
- Geometry: box, circle, cylinder, plane, sphere, square, triangle, triangle mesh, polygonal prism
- Unit pose and optional linear/angular motion
- Collider settings: MAC, restitution, diffuse sampling
- Sinks: surface, volume, tracing
- Codec representative parameters

For `template.jsonc`, use comments to document alternatives. Keep `cylinder.json` strict JSON and directly executable.

## Interactive rules
- Remove hard-coded simulation construction from the native-window example.
- Load `cases/cylinder.json` by default.
- Custom config/case selection may override the default.
- Use existing `JsonCodec -> SimulationConfig -> Session/SystemFactory -> Atlas Core` path.
- Do not duplicate geometry/material/solver dispatch already implemented in `SystemFactory`.
- Rendering must consume the existing interactive/session view path, not construct physics objects.

## Case extension
New cases go only under `cases/`, e.g.:
```text
cpp/cases/channel.h
interactive/cases/channel.json
python/cases/channel.py
```
Add a small selector branch in each `main.*`. Do not introduce extra abstraction files until case selection becomes large enough to justify them.

## Smoke benchmark
Rewrite `benchmarks/atlas/smoke/main.cpp` to exercise a minimal cylinder/DSMC update path rather than the current trivial drifting-particle placeholder.

Requirements:
- Exercise Atlas core + DSMC solver + cylinder collider + `System::update()`.
- Keep workload small and deterministic.
- Avoid a benchmark whose workload grows continuously because of emitters; reset/recreate state as needed so iterations are comparable.
- Treat it as harness/execution smoke first, not a representative performance benchmark.

## Cleanup
Update all references to old example paths/names in CMake, tests, README/docs, Dockerfile, and workflows. Remove obsolete example files/directories after replacements are wired.

## Done when
- The three example frontends have matching directory semantics.
- Running each frontend with no case argument uses cylinder.
- Templates expose the major current engine configuration surface.
- Interactive uses JSON rather than hard-coded physics setup.
- Smoke benchmark runs a minimal cylinder DSMC path.
- `examples/README.md` is the single example tutorial/source of truth.
