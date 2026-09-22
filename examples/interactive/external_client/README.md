# Planned External Client Contract

Control and rendered frames are separate channels.

The control channel will carry generic `Command` values to `Server`. `Server`
dispatches `start`, `pause`, `step`, and `close` operations to `Session`.
`initialize` still needs a native `System` payload contract, and no request
serialization or transport listener exists yet.

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

`requests.jsonl` is a proposed textual representation of the current `Command`
and `Request::step_count` concepts. It is documentation, not a finalized wire
format or an implemented parser.
