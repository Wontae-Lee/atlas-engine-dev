# Frontend Boundaries

The dependency graph is one-way:

```text
bindings/python  -----> Atlas Core <-----  src/interactive
```

Atlas Core never depends on either consumer, and Python does not route through
Interactive.

The [Python binding](../frontends/python.md) mirrors core objects and adapts the
language boundary. NumPy conversion, Python lifetime anchors, and runtime forms
of C++ template state access remain under `bindings/python/`.

The [Interactive subsystem](../frontends/interactive.md) owns JSON protocol,
session control, application statistics, and native rendering. Its execution
library can be built without graphics. Rendering consumes non-owning Session
views and transfers core buffers directly to OpenGL through the selected backend.

Application policies such as CSV files, window presentation, and external
communication stay outside the core. Explicit core serialization remains
available to both consumers for state persistence.
