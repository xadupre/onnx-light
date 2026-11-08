---
name: never-request-root-access
description: Avoids blanket filesystem-root permission requests when running repository commands, formatters, builds and tests.
---

# Never request access to filesystem root

- Never request blanket access to `/` for repository work.
- Keep commands and file access scoped to the repository and the specific
  dependencies required by the task.
- Run formatters, builds and tests as separate, direct tool calls. Avoid shell
  wrappers, chained commands and dynamically constructed paths when they cause
  permission inference to request access to `/`.
- Determine build parallelism in a separate read-only call, then pass the
  calculated worker count literally to the build command.
- If a tool proposes access to `/`, do not submit or repeat that request.
  Reformulate the command with explicit repository-relative paths.
- If a required operation still cannot be scoped narrowly, explain the blocker
  rather than asking for filesystem-root access.
