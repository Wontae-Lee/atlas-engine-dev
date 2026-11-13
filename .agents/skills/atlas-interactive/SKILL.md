---
name: atlas-interactive
description: Modify Atlas Interactive configuration, commands, session execution, or rendering under src/interactive while preserving Core ownership and headless operation. Use for JSON-to-Core wiring, lifecycle changes, and rendering-state bugs; example-only edits belong to the examples workflow.
---

# Atlas Interactive

Read the repository [bootstrap](../../SKILL.md), then the task's relevant source
and [Interactive guide](../../../docs/frontends/interactive.md). This skill covers
the native frontend, not numerical algorithms inside Core.
Apply the common [coding style](../atlas-coding-style/SKILL.md) before editing;
this skill adds Interactive architecture and integration guidance.

## Find the owner

| Change | Read first |
|---|---|
| JSON fields, enum names, defaults, partial validation | [SimulationConfig](../../../src/interactive/config/simulation_config.h), [JsonCodec](../../../src/interactive/transport/json_codec.cpp), [CoreFactory](../../../src/interactive/config/system_factory.cpp) |
| New command or response | [protocol](../../../src/interactive/protocol), [Server](../../../src/interactive/server/server.cpp), [JsonLinesTransport](../../../src/interactive/transport/json_lines_transport.cpp) |
| Start, pause, step, restart, statistics, CSV | [Session](../../../src/interactive/session/session.cpp), [statistics](../../../src/interactive/statistics), [output](../../../src/interactive/output) |
| Render commands and window/session lifetime | [InteractiveApplication](../../../src/interactive/application/interactive_application.cpp), [RenderManager](../../../src/interactive/application/render_manager.cpp) |
| Missing or stale particles/meshes | [scene views](../../../src/interactive/view), [RawStateProvider](../../../src/interactive/rendering/state/raw_state_provider.cpp), [layers](../../../src/interactive/rendering/layer) |
| Backend upload or graphics lifetime | [StateBridge implementations](../../../src/interactive/rendering/backend), [OpenGL owners](../../../src/interactive/rendering/opengl), [render targets](../../../src/interactive/rendering/target) |
| Native executable or library boundary | [main.cpp](../../../src/interactive/main.cpp), [CMakeLists.txt](../../../src/interactive/CMakeLists.txt) |

## Configuration and commands

- Trace new settings through `SimulationConfig` → `JsonCodec` → `CoreFactory` →
  the existing Core builder. Do not add an example-only JSON key that the engine
  ignores. Check both complete simulation decoding and the partial `validate`
  target when changing a required field or geometry variant.
- Keep structural checks in the decoder: JSON types, vector dimensions, enum
  spellings, and required fields. Let Core builders decide numerical and physical
  validity. Use existing builder calls rather than duplicating their rules.
- Unit translation and mesh geometry translation are different stages. For a
  mesh, `geometry.translation` shifts local vertices before the Unit's pose;
  `unit.translation` positions the Unit in world space. Plane `offset` remains
  a scalar distance, not a mesh translation vector.
- Keep command scope explicit. `create`, `validate`, and `shutdown` are server
  scoped; session operations carry `session_id`. Rendering commands go through
  `InteractiveApplication`, which coordinates its server and optional renderer.
- The JSONL service in `src/interactive/main.cpp` intentionally has CLI parsing,
  exception-to-error handling, and a separate blocking-input reader. Keep stdout
  for protocol responses and diagnostics on stderr. The minimal, no-CLI style
  of `examples/interactive/02_intake` does not apply to this service.

## Execution and rendering

- `InteractiveApplication::update()` advances the server before updating
  rendering. Rendering may skip a frame independently of the simulation rate.
  Do not advance physics from `Renderer`, a layer, or a camera handler.
- Closing a window only releases rendering resources. Pausing retains state;
  restarting rebuilds the system from the factory's original configuration.
  Closing a session or shutting down through the application closes its attached
  renderer first. Preserve these distinctions when changing lifecycle commands.
- Obtain fresh `Session::scene_view()` data for a frame. Particle buffer views
  expose only the live prefix; geometry views pair configuration with current
  Core transforms. Borrowed pointers must not survive resize, restart, or session
  destruction. `RawStateProvider::finish_frame()` clears borrowed geometry views.
- OBJ files are loaded by `TriangleMesh::load_from_obj()` through `CoreFactory`.
  Relative `file` paths currently use the process working directory. The factory
  owns meshes and associates geometry descriptions with those owners; its
  rebuild path refreshes the lookup while reusing cached meshes for restart.
  Do not retain addresses of temporary configuration objects in a render view.
- File-backed meshes can have an empty JSON `triangles` array. Render the actual
  mesh supplied by `GeometryRenderView::mesh`, with its local translation already
  applied, then apply the live Unit transform. Do not reload files per frame or
  reapply the local translation. The inline-triangle fallback still needs that
  translation when no mesh owner is supplied.
- Preserve the CMake split: `atlas::interactive` has no graphics dependencies;
  `atlas::interactive-rendering` adds them. Exactly one `StateBridge` implementation
  is selected for TBB or CUDA. Keep the CUDA host-staging fallback and release
  registrations before reallocating or discarding OpenGL buffers.

## Finish the change

Update the [Interactive guide](../../../docs/frontends/interactive.md) for wire
fields or public behavior, and [frontend architecture](../../../docs/architecture/frontends.md)
when ownership changes. Relevant existing coverage lives in
[execution tests](../../../tests/interactive/interactive_tests.cpp) and
[rendering tests](../../../tests/interactive/rendering_tests.cpp). For mesh or
lifetime changes, cover local/world transforms and restart as well as first
creation. Follow bootstrap execution permissions; do not infer permission to
build, run a simulation, or test from this skill.
