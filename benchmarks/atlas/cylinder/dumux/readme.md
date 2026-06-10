# DuMux - Flow Around A Cylinder

This case uses DuMux's unstructured DFG benchmark Navier-Stokes cylinder-channel setup.

- `run.sh --smoke` uses `cylinder_channel.msh` and disables indicator checks for a quick execution check.
- `run.sh --full` uses `cylinder_channel_quad.msh` and enables the DuMux drag/lift/pressure indicator checks.
