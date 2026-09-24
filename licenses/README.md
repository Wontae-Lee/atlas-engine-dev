# Third-party notices

`third-party/` retains upstream license texts for the pinned dependencies in
`cmake/dependencies/CMakeLists.txt` and nanobind 2.13.0 from PyPI. nanobind uses
Tessil robin-map; Protobuf includes utf8_range. Their notices are included too.
Wheels include these notices under their distribution metadata. Docker runtime
images carry them under `/opt/atlas/licenses`.

Dependency versions, sources, and installation are documented in
[the dependency guide](../docs/contributing/dependencies.md). When updating a
release pin, review the corresponding license text as well as its archive hash.
