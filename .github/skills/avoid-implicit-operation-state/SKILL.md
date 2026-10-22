---
name: avoid-implicit-operation-state
description: Avoids mutable static or thread-local flags that implicitly change operation semantics. Use when designing copy modes, scoped behavior overrides, or cross-module operation contexts.
---

# Keep operation semantics explicit

Do not use mutable globals, function-local statics, or `thread_local` flags or
counters to change how constructors, copies, or other operations behave.
A scoped RAII guard does not make this design explicit: unrelated or reentrant
calls still depend on ambient state.

1. Identify where the operation's policy originates and every nested call that
   needs it.
2. Express the choice through a named operation, an explicit argument or tag,
   or an explicitly passed context. Propagate it through nested calls rather
   than consulting shared state.
3. Keep ordinary operations independent of special modes. In particular, a
   payload-sharing copy must not change the semantics of unrelated ordinary
   copies performed during that operation.
4. Preserve ownership, lifetime, validation and exception-safety guarantees.
   Do not replace an ambient flag with another hidden mutable singleton.
5. Cover nested operations, ordinary-operation isolation and relevant failure
   paths in focused tests.

Immutable constants and explicit per-object state are not prohibited. The
design to avoid is hidden mutable state controlling an operation's semantics.
