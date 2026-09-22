# Planned External Client Contract

Control and rendered frames are separate channels.

The control channel will carry generic `Command` values to `Server`. `Server`
forwards requests, and `Session` dispatches `start`, `pause`, `step`, `status`,
`save`, `restart`, `close`, and `shutdown`. The native process supplies a
`SystemFactory`; the current protocol does not transfer a `System` or an
initialization payload. No request serialization or transport listener exists
yet.

The frame channel will run independently:

```text
Renderer
  -> OffscreenTarget
  -> FrameStream
  -> external client
```

`OffscreenTarget` and concrete `FrameStream` transport are planned work. The
client will receive rendered frames rather than raw Fluid buffers, CUDA
pointers, OpenGL buffer identifiers, or per-particle JSON objects.

Camera commands are also future protocol work. They can eventually update the
native `Camera` without exposing simulation memory to the client.

`requests.jsonl` is a proposed textual representation of the current `Command`,
`Request::step_count`, and `Request::path` concepts. It is documentation, not a
finalized wire format or an implemented parser.
