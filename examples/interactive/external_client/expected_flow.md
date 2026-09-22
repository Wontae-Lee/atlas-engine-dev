# Expected External Client Flow

1. The client launches `atlas-interactive`, optionally with `--config`.
2. The process emits a startup response if a config file created a Session.
3. The client may create more sessions by sending configuration in a `create`
   request.
4. `Server` owns every live Session and routes commands by `session_id`.
5. `start` makes the selected Session advance continuously in the server loop.
6. `pause` stops automatic updates, while `step` advances an explicit count.
7. `status`, `save`, and `restart` operate on the selected Session.
8. `close` removes that Session; `shutdown` stops the process.
9. Every response carries the request's `request_id` for correlation.
10. A future rendering client obtains frames through
    `Renderer -> OffscreenTarget -> FrameStream`, independently of JSON
    control.

The live core System never crosses the protocol boundary, and particle arrays
are not serialized into JSON.
