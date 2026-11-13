# 01 — Python cylinder flow

`main.py` independently constructs and runs a small molecular freestream around
an open cylinder. It has no case selector and does not import another example.
Install the Atlas package and NumPy, then run from the repository root:

```bash
ATLAS_DEFAULT_ENGINE=tbb python examples/python/01_cylinder/main.py
ATLAS_DEFAULT_ENGINE=tbb python examples/python/01_cylinder/main.py 40
```

The optional nonnegative decimal argument selects the number of simulation
updates (default 40). `0` builds and reports without stepping. Invalid counts
and extra arguments fail with a usage message; `--help` displays the syntax.
Select CUDA through `ATLAS_DEFAULT_ENGINE=cuda` before starting the script when
the CUDA extension and a working GPU are available.

The setup uses a 20,000-particle capacity, `0.1` m cells, a `5e-5` s time step,
300 K gas, and a 500 m/s stream in the positive x direction. A volume source
and Maxwell–Boltzmann generator inject particles, a DSMC solver advances
collisions, and an isothermal collider represents the cylinder with radius
`0.2` m. Six thin volume sinks remove particles near the domain boundaries.
Edit the constants and component construction in `run()` to adjust this setup.

Progress reports every ten steps and on the final step. The final summary
includes live particle count, particles downstream of the cylinder, and mean
streamwise velocity. The NumPy arrays used for analysis are owned snapshots;
they do not share live simulation storage. Floating-point and stochastic
results can differ across backends. This is a smoke example, not a validated
engineering benchmark.

For a smaller starting point, ownership rules, and instructions for creating
`02_<name>`, use the [Python template guide](../00_template/README.md).
