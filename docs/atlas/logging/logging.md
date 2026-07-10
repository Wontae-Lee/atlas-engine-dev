# Logging

A small, header-first diagnostics facility: a set of stream-into factories
(`atlas::warn()`, `atlas::info()`, `atlas::error()`, `atlas::debug()`) that emit
one timestamped, source-located line per streaming expression, plus a
throw-on-failure guard pair (`atlas::check` / `atlas::error_if`) that logs at
`error` level and throws. It is not part of the `System::update()` step pipeline;
it is a cross-cutting utility the runtime modules call from their host-side guard
branches (e.g. the searcher when a per-cell count state is missing, the DSMC
solver when a required view is incomplete). The whole feature is gated on
`ATLAS_ENABLE_LOGGING`: when the macro is absent the header supplies
same-spelling no-op stand-ins, so call sites never need an `#ifdef`.

## Files

| File | Role |
|---|---|
| `include/atlas/logging/logging.h` | The entire public surface: `LoggingLevel`, the `Logger` RAII accumulator (and its `NullLogger` disabled twin), the `Logging` config facade, the level factories, `should_log` / `emit_log_line`, and the `ErrorIfProxy` / `check` / `error_if` guards with their `ATLAS_CHECK` / `ATLAS_ERROR_IF` macros. |
| `src/atlas/logging/logging.cpp` | Out-of-line definitions compiled only under `ATLAS_ENABLE_LOGGING`: the file-scope stream pointers, threshold, and mutex; `should_log`, `emit_log_line`, `Logger::~Logger`, and the `Logging` setters. |

The CMake wiring (`cmake/AtlasLibraries.cmake`) builds `atlas-logging` and adds
`ATLAS_ENABLE_LOGGING` to `atlas-core` **only when `ATLAS_LOGGING=ON`** (the debug
presets; the release presets set it `OFF`). Linking `atlas::core` then pulls in
`atlas::logging` transitively.

## The shape

### `LoggingLevel` — one enum, two roles

```cpp
enum class LoggingLevel : uint8_t {
    all = 0, debug = 1, info = 2, warn = 3, error = 4, off = 5
};
```

Ordered by increasing severity so a single `uint8_t` comparison decides
visibility: a message of level `m` is emitted while the threshold `t` satisfies
`t <= m` (`is_leq` in the source). Smaller means *more verbose*. The type is a
threshold when stored globally (`all` admits everything, `off` suppresses
everything) and a per-message level when attached to a line (only `debug`,
`info`, `warn`, `error` are meaningful there). The fixed `uint8_t` underlying
type is load-bearing: `is_leq` casts through it.

### `Logger` — RAII line accumulator

`Logger` is meant to live exactly as long as one streaming expression. Each
`operator<<` appends to an internal `std::stringstream`; the destructor formats
and emits the finished line once — but only if the buffer is non-empty *and*
`should_log(_level)` passes. `operator<<` is `const` and the buffer `mutable`,
so a temporary returned by value from `warn()` et al. can still be streamed into.
Copy is deleted; move exists solely so the factories can return by value. The
captured `std::source_location` is the factory's call site (file/line/function),
not the destructor's, so lines point at user code.

`emit_log_line` produces `[LEVEL] YYYY-MM-DD HH:MM:SS file:line: function:
message`, selects the per-level stream, and flushes. A global `std::mutex`
serializes the whole write (and re-checks the threshold under the lock) so lines
from different threads never interleave; an individual `Logger` object is still
not safe to share across threads.

### `Logging` — process-wide config facade

All-static, never instantiated. Under the mutex it overrides the per-level
destination stream (`set_info_stream` / `set_warn_stream` / `set_error_stream` /
`set_debug_stream`, or `set_all_stream` for all four), sets the global threshold
(`set_level`), or flips it to the extremes (`mute` → `off`, `unmute` → `all`).
Passing `nullptr` to a stream setter restores the default: `info`/`warn`/`debug`
default to `std::cout`, `error` to `std::cerr`. Streams are not owned and must
outlive their use.

### The disabled build (`#else`)

When `ATLAS_ENABLE_LOGGING` is undefined, the header instead defines a
`NullLogger` whose `operator<<` discards everything, a `make_null_logger()`
returning a shared `constexpr` instance, an all-no-op `Logging` class, and
factory functions returning the null sink. Streamed *arguments are still
evaluated* (their side effects run); only formatting and output are elided. This
is the path the release presets compile.

### `check` / `error_if` — log-and-throw guards

```cpp
template <typename ExceptionT = std::runtime_error>
ErrorIfProxy<ExceptionT> error_if(bool cond, std::source_location loc = ...) noexcept;
template <typename ExceptionT = std::runtime_error>
ErrorIfProxy<ExceptionT> check(bool cond, std::source_location loc = ...) noexcept;
```

`ErrorIfProxy` collects an optional message via `operator<<` and, if armed,
throws `ExceptionT` from its destructor at the end of the full expression:
`atlas::check<std::invalid_argument>(x > 0) << "bad x";`. `error_if` arms on
`cond == true`; `check` arms on `cond == false` (assertion form). Before throwing
it emits an `error` line **only when `ATLAS_ENABLE_LOGGING` is defined** (guarded
by `#ifdef` inside the destructor); the throw happens either way, substituting
`"ATLAS error"` when no message was streamed. The destructor is
`noexcept(false)` by design — do not construct one while an exception already
propagates.

The `ATLAS_CHECK(cond)` / `ATLAS_ERROR_IF(cond)` macros expand to `check` /
`error_if` in debug builds and to the `NoopErrorProxy` (via `(void)sizeof(cond)`,
which type-checks but does not evaluate `cond`) when `NDEBUG` is defined. Note
this gate is `NDEBUG`, *independent* of `ATLAS_ENABLE_LOGGING`.

## Deliberately absent

- **No log-to-file, rotation, or async sink.** Emission is a locked, flushing
  write to a raw `std::ostream*`. The redirection hook is `set_*_stream`; anything
  richer is the consumer's stream to install.
- **No `printf`-style or `std::format` API.** Formatting is `operator<<` into a
  `std::stringstream`, i.e. whatever a type's `ostream` overload provides.
- **No thread-safe `Logger` object.** Only the final emission is serialized; a
  `Logger` is a single-expression temporary and is documented not to be shared.
- **`should_log`'s pre-check is deliberately lock-free** (a plain read of
  `s_level`); `emit_log_line` re-checks under the mutex so the visible decision
  stays consistent. The fast path is meant to be cheap.

## Not implemented

The facility is a general-purpose API, but the engine drives only a narrow slice
of it. Of the four message levels, **only `warn` is ever emitted through its
factory, and `error` only through the `check` guard** — `info` and `debug` are
never produced anywhere in `include/`, `src/`, `tests/`, or `benchmarks/`.
Everything below is a public extension point available to a library consumer, not
dead code to be removed; nothing here is written to by engine code, and no test
exercises it.

Confirmed callers that *do* exist: `atlas::warn()` at
`src/atlas/searcher/spatial_hashing_searcher.cu:170,178` and
`src/atlas/solver/dsmc/dsmc_solver.cu:248,257,264`; `atlas::check<...>()` at 11
sites across `spatial_hashing_searcher.cu`, `codec/knudsen_codec.cu`, and
`universe/universe.cu`. Those two are live.

| What | Where | Evidence | Verdict |
|---|---|---|---|
| `atlas::info()` factory + `LoggingLevel::info` as a message level | `logging.h:212` | No caller in `include`/`src`/`tests`/`benchmarks` (only a comment mentions `atlas::info()` at `logging.cpp:217`). | Declared, never called — public API extension point. |
| `atlas::debug()` factory + `LoggingLevel::debug` as a message level | `logging.h:245` | No caller anywhere. | Declared, never called — public API extension point. |
| `atlas::error()` factory | `logging.h:234` | No caller anywhere. The `error` *level* is still reached, but only via `ErrorIfProxy::~ErrorIfProxy` (the `check` path), never through this `Logger` factory. | Declared, never called — public API extension point. |
| `atlas::error_if()` + `ATLAS_ERROR_IF` macro | `logging.h:511`, `545` | No caller; every guard site uses `check` instead. | Declared, never called — public API extension point. |
| `ATLAS_CHECK` macro | `logging.h:565` | No caller; the 11 guard sites call `atlas::check<...>()` directly, not the macro. | Declared, never called. |
| `Logging` facade — `set_info_stream`, `set_warn_stream`, `set_error_stream`, `set_debug_stream`, `set_all_stream`, `set_level`, `mute`, `unmute` | `logging.h:132`–`202` | No `Logging::` call anywhere outside the module; no test constructs or configures it. Consequently the `off` threshold and the `mute()`/`unmute()` transitions are never exercised in-engine — `s_level` stays at its default `all`. | Declared, never called — the runtime-configuration surface for a library consumer. |
| `NoopErrorProxy` | `logging.h:482` | Only reachable through `ATLAS_CHECK` / `ATLAS_ERROR_IF` under `NDEBUG`; since those macros have no callers, this type is never instantiated. | Exists so the disabled-macro path has a return type — latent, tied to the unused macros. |

## Extending

- **Emit at a new level from engine code.** Call the matching factory and stream
  into it: `atlas::info() << "resolved regime " << name;`. Nothing else is
  required — the line appears whenever the global threshold admits that level,
  and compiles to nothing in an `ATLAS_LOGGING=OFF` build. Emitting `info` or
  `debug` is how you would give those currently-unused levels a producer.
- **Redirect or silence output** (consumer side): install a stream with
  `Logging::set_all_stream(&my_stream)` or one of the per-level setters, and
  raise the bar with `Logging::set_level(LoggingLevel::warn)` (or `mute()`).
  These are the intended entry points for the setters listed as never-called
  above.
- **Add a guarded precondition:** `atlas::check<SomeException>(cond) << "why";`.
  Pick the exception as the template argument (default `std::runtime_error`); the
  proxy logs at `error` and throws at the end of the full expression. Use the
  `ATLAS_CHECK` macro form only where you want the check to vanish under `NDEBUG`.
- **Add a new `LoggingLevel`:** insert it in severity order between `all` and
  `off` (the ordering is the contract `is_leq` relies on), then extend the
  `stream_for` and `level_to_string` switches in `logging.cpp` and add a factory
  plus its `NullLogger` twin in the header. There is no `static_assert` guarding
  the ordering — keep the enumerators monotonically increasing by severity by
  hand.
