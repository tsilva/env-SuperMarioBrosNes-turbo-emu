# Native architecture

The published `supermariobrosnes-turbo` package is a statically specialized
Super Mario Bros extension. It does not discover games at runtime.

## Binary boundaries

- `crates/nes-turbo-nrom-core` is unpublished. It owns the 6502 interpreter,
  PPU, mapper-0 cartridge parsing, controller, renderer, profiler, FCEU state
  loading, frame/event loop, and the `NromGame` contract.
- `crates/smb-turbo-driver` is compiled only into the SMB extension. It owns
  the canonical ROM digest, SMB signals and episode semantics, sprite-zero
  timing, signature capabilities, dispatch PCs, fast-path handlers, and the
  semantic descriptor/decoder catalog for opt-in research infos.
- The root crate owns the concrete `MarioVecEnv`, PyO3/NumPy boundary, Rayon
  lane stepping, preprocessing, construction-time info selection, optional
  batched staging buffer, and the existing Python module.

Future games must use a sibling driver and a sibling native extension/wheel.
They must not be normal dependencies of this root package. This keeps their
code, dispatch sites, and LTO layout outside the SMB binary.

## Static driver contract

`NromMachine<G>` is monomorphized for one `NromGame`. A driver supplies:

- `Options`, cached `Signals`, pre-step `StepContext`, and `FastPaths`;
- a `PpuTiming` type with `SPRITE0_HIT_DOT: Option<usize>`;
- construction-time signature detection;
- a direct-PC `dispatch_fast_path` implementation;
- `synchronize` after reset, state loading, cold boot, and every frame;
- pre-step context plus post-step reward and termination calculation.

There are no trait objects, registries, function-pointer tables, string
lookups, JSON/YAML profiles, or runtime game selection in the frame loop.

Research-info decoders are pure immutable RAM reads performed once after an
info-producing vector step or reset. They are deliberately separate from the
per-frame `synchronize` signal path. When no extra key is selected, the root
crate stores no descriptor IDs, allocates no staging buffer, and makes no
research-info extraction call.

The core owns ordinary interpretation, PPU ticking, NMI ordering, pending PPU
cycle delivery, event boundaries, and the frame guard. A handler may mutate
CPU, RAM, stack, controller, PPU status, and `FrameBudget`, but may not tick the
PPU. `Miss` is side-effect-free. An applied handler returns one of:

- `Continue` when its headroom proof permits deferred delivery;
- `FlushIfEventDue` when the central loop should conditionally deliver events;
- `FlushNow` when the central loop must deliver accumulated cycles now.

The profiled and unprofiled loops remain separate and structurally parallel.

## SMB fast paths

The SMB driver has 13 capability guards and 12 dispatch sites. Capabilities
and sites are deliberately separate: `sprite0_poll_exit`, for example, is a
dependency of the sprite-poll handler rather than another PC dispatch site.
The local `smb_fast_paths!` macro generates the capability fields, exact
construction-time detection, dependency checks, and direct `match` dispatch.

The checked dependency closure in `ci/smb-extension-dependencies.txt` is an
additional isolation guard. A future driver must not alter it. A deliberate
shared-core or dependency change must update it explicitly and still pass the
SMB correctness and exact-ref throughput gates.

## Adding another NROM game

1. Create a sibling unpublished driver crate implementing `NromGame`.
2. Create a sibling PyO3 extension/package that depends on the core and that
   driver, never on `smb-turbo-driver` or this root extension.
3. Keep ROM identity, signals, timing, reward/termination, signatures, PCs,
   and handlers entirely in the new driver.
4. Give every signature an exact-match and one-byte-rejection test. Check
   unique PCs, valid dependencies, side-effect-free misses, interpreter
   equivalence, event headroom, and frame-guard fallback.
5. Add enabled-versus-disabled seeded traces for representative saved states
   and preprocessing/reset configurations.
6. Build the new wheel independently. Confirm the SMB dependency-closure check
   is unchanged.
7. If the shared NROM core changes, run the complete SMB test/oracle suite and
   exact-ref benchmark acceptance. Do not waive a regression.

A non-NROM game should receive a mapper-specific sibling core rather than
widening this one or adding mapper dispatch to the SMB frame loop.

## Verification hooks

The core's non-release `test-support` feature exposes deterministic state
snapshots and fast-path disabling for equivalence tests. The external
`synthetic_driver` integration test proves that a consumer can define its own
signals, `None` sprite timing, signature-guarded handler, and interpreter
fallback without becoming an SMB release dependency.
