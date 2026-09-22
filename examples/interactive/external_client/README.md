# JSONL External Client Contract

The control channel is implemented as newline-delimited JSON over the
`atlas-interactive` process's stdin and stdout. Start it with the sibling
`simulation.json` file:

```bash
atlas-interactive --config examples/interactive/simulation.json
```

The startup response returns the created `session_id`. Each subsequent request
uses a caller-defined `request_id`; commands that target a simulation also
carry that `session_id`. The example `requests.jsonl` assumes the startup
session is ID 1.

The process supports `create`, `start`, `pause`, `step`, `status`,
`save`, `restart`, `close`, and `shutdown`. A `create` request carries
`payload.config` and optional `payload.output`. `step` carries
`payload.step_count`, and `save` carries `payload.path`.

Control and rendered frames remain separate channels. The future frame path is:

```text
Renderer
  -> OffscreenTarget
  -> FrameStream
  -> external client
```

`OffscreenTarget` and a concrete `FrameStream` are not implemented. A future
client will receive rendered frames rather than Fluid buffers, CUDA pointers,
OpenGL identifiers, or per-particle JSON.
