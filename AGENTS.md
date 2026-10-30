# env-SuperMarioBrosNes-turbo-emu Codex Notes

## Product Specifications

When specifications exist, use `$specs-author` for substantive work and
specification changes. Follow its shared read/reuse cadence, provisional-scenario
guidance, proportional detail, decision-based approval, and closing intent check.
The maintained procedure is `/Users/tsilva/.codex/skills/private-skills/specs-author/SKILL.md`.

- Use `/mario-emu_build-release` for local metadata-only tagging and GitHub Actions builds
  of the validated cross-platform
  PyPI wheel set plus source distribution. Skill:
  `.codex/skills/mario-emu_build-release/SKILL.md`.
- Use `/mario-emu_benchmark-latest-release` to benchmark the newest stable PyPI release
  with the canonical TurboBench workflow, publish verified official evidence
  to Hugging Face, and refresh the project benchmark documentation. Skill:
  `.codex/skills/mario-emu_benchmark-latest-release/SKILL.md`.

## Shared release procedure

The project `mario-emu_build-release` skill composes `$release-workflow` from
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md`.
Read both for release work; keep project commands, version policy, artifact
requirements, and approval gates in the project adapter.
