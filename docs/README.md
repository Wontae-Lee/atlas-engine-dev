# Atlas documentation

The root [README](../README.md) introduces Atlas and its quick starts. The guides below explain the APIs, implementation, and maintenance workflows in more detail.

| Topic | Start here | What it covers |
|---|---|---|
| Use Atlas | [Examples](../examples/README.md) | The maintained cylinder case in C++, Python, and Interactive |
| Core architecture | [Overview](architecture/overview.md) and [simulation pipeline](architecture/simulation-pipeline.md) | State ownership, builders, and step ordering |
| Core C++ API | [Core modules](atlas/) and [System](atlas/system/system.md) | Object construction, policies, backend views, and persistence |
| Python | [Python frontend](frontends/python.md) | Engine selection, bindings, and NumPy ownership |
| Interactive | [Interactive guide](frontends/interactive.md) | JSON configuration, commands, sessions, and native rendering |
| Frontend design | [Frontend boundaries](architecture/frontends.md) | How Python and Interactive consume Core independently |
| Build from source | [Build guide](contributing/build.md) and [dependencies](contributing/dependencies.md) | TBB, CUDA, headless, rendering, presets, and toolchains |
| Contribute | [Workflow](contributing/workflow.md), [coding style](contributing/coding-style.md), and [testing](contributing/testing.md) | Changes, documentation, conventions, and test suites |
| Operate and release | [Docker](operations/docker.md), [CI](operations/ci.md), and [releases](operations/releases.md) | Images, automation, publication, and versioning |
