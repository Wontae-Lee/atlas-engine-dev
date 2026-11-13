# 01 — Interactive cylinder flow

`config.json` emits a molecular freestream at 500 m/s and 300 K, advances it
around an open cylinder of radius 0.2, and removes particles using six outer
volume sinks. It uses a DSMC variable-hard-sphere solver, a 20,000-particle
capacity, and `dt = 5e-5`. The cylinder has velocity `(0, 0.02, 0)` to demonstrate
live geometry poses in the renderer; the C++ and Python cylinder examples keep
the cylinder stationary. This is a smoke example, not a validated benchmark.

After configuring with `ATLAS_EXAMPLES=ON`, `ATLAS_INTERACTIVE=ON`, and
`ATLAS_INTERACTIVE_RENDERING=ON`:

```bash
cmake --build build/interactive-tbb --target atlas_example_interactive_01_cylinder
./build/interactive-tbb/examples/interactive/01_cylinder/atlas_example_interactive_01_cylinder
```

This example follows [`02_intake`](../02_intake/README.md), the standard
Interactive example. It loads `config.json` and executes the lifecycle directly
in `main()`, with no CLI options, helper functions, progress output, or exception
wrapper. The window runs until closed, then the session and application close
without an extra simulation update. Inline comments explain each stage.
The camera is initialized with `Camera(position, target)` in `main.cpp`.

Edit the source `config.json` and rebuild to refresh the build-tree copy.
The same file can be loaded by the headless JSONL application:

```bash
./build/interactive-headless-tbb/src/interactive/atlas-interactive \
    --config examples/interactive/01_cylinder/config.json
```

See the [template guide](../00_template/README.md) for full configuration,
controls, installation, lifecycle details, and instructions for adding a new
numbered example.
