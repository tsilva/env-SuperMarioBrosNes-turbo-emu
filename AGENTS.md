# SuperMarioBros-Nes-turbo Codex Notes

## Product Specifications

Before every task in this repository, use the `$specs-author` skill to read the entire root `SPECS.md`. Before finishing, reread it and check the task and conversation for new or changed stakeholder intent.

- Treat `SPECS.md` as the persistent source of stakeholder requirements that cannot be inferred reliably from code or remembered conversations.
- If the task, repository, or user request contradicts, omits, or ambiguously interprets the specification, tell the user. Continue safe exploration and work that does not depend on resolving the issue, but never silently choose an interpretation.
- Never edit `SPECS.md` from inference. Propose the exact change, explain why it reflects stakeholder intent, and edit the file only after the user explicitly approves that exact change.
- Keep `SPECS.md` complete, concise, and compacted. It must contain stakeholder intent rather than implementation, architecture, operations, or transient project detail.

- Use `/autoresearch-speed` for throughput optimization involving
  `scripts/benchmark_sps.py`, emulator hot paths, `env_steps_per_sec`, or
  autoresearch campaigns. Skill: `.codex/skills/autoresearch-speed/SKILL.md`.
- Use `/build-release` to tag a version and build the validated cross-platform
  PyPI wheel set plus source distribution. Skill:
  `.codex/skills/build-release/SKILL.md`.
- Use `/regenerate-mario-promo` to rebuild the verified Stable Retro versus
  SuperMarioBros-Nes-turbo Level 1-1 promotional comparison. Skill:
  `.codex/skills/regenerate-mario-promo/SKILL.md`.
