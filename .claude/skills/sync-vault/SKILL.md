---
name: sync-vault
description: "Propose updates to the Obsidian vault at C:\\Users\\joshc\\OneDrive\\Desktop\\Vault\\Projects\\Bet Bot so it stays current after commits, schema changes, or new findings. Triggers: \"sync vault\", \"update vault\", \"vault sync\", \"did we update the vault\", \"refresh the vault\". Should also be proactively suggested at the end of any session that touched code or surfaced findings worth recording (especially before /handoff)."
user-invocable: true
---

## Why this skill exists

The Bet Bot project keeps a substantive Obsidian vault as its second-brain: design decisions, market specs, progress log, open questions. Without active maintenance the vault decays — and once it's stale, agents start trusting the code over the vault, which kills the vault's value entirely.

This skill encodes which kinds of changes belong in which notes so vault updates are routine rather than judgment-driven (which is what fails — the agent forgets).

## Vault location

`C:\Users\joshc\OneDrive\Desktop\Vault\Projects\Bet Bot\`

Inventory:
- `BetBot_index.md` — project index, "Current status" section, "Open questions & future work" checklist, topic map
- `decisions_Bet.md` — architectural decisions with code + rationale, grouped by domain
- `progress.md` — chronological build log, dated entries with commit hashes
- `markets/{btts,ou15,ou25}.md` — per-market spec: models, thresholds, config, edge detection flow
- `data/{sources,team-mapping}.md` — data sources, freshness, name normalisation
- `ops/{dashboard,settlement}.md` — operational layer
- `CONTEXT.md` — ubiquitous-language doc (mirrors the repo copy)
- `to-do_Bet.md` — currently empty/placeholder; to-dos actually live in `BetBot_index.md` "Open questions & future work"
- `markets 1/btts.md` — appears to be a rename artefact, **do not touch**; flag to user

## Discrimination map: what kind of change → which note

| Change type | Vault note(s) | What to write |
|---|---|---|
| New commit (feature) | `progress.md` + `BetBot_index.md` Current status | Progress: dated entry, commit hash, what was built, why, wikilinks. Index: tick off any status item resolved; add new status line if ongoing. |
| New commit (fix) | `progress.md` + `BetBot_index.md` if hardening item resolved | Progress: dated entry referencing the original finding. Index: tick off if applicable. |
| New design decision | `decisions_Bet.md` (new subsection) + `BetBot_index.md` Decisions section (new bullet with wikilink) | Decisions: section with rationale + code. Index: one-line link. |
| Existing decision changed | `decisions_Bet.md` (don't delete — append "Updated DD MMM YYYY:" sub-paragraph) | Preserve original rationale; explain what changed and why. |
| New finding (open) | `BetBot_index.md` "Open questions & future work" + `~\.claude\projects\C--Users-joshc-OneDrive-Documents-Project\memory\pre_live_hardening.md` if money-adjacent | Index: one-line `- [ ]` entry, severity flag if critical. Memory: full detail with file:line citation. |
| Finding resolved | `BetBot_index.md` (tick `[x]`), `progress.md` (entry), `pre_live_hardening.md` if applicable | Mark `[x]`, dated, link to commit. |
| Market rule / threshold / multiplier change | `markets/<market>.md` + `decisions_Bet.md` if it warrants a rationale entry | Update the threshold table. Decisions: only if the change is non-obvious. |
| Data source added / changed / quality issue | `data/sources.md` | Source name, what it provides, update frequency, known limitations / blockers. |
| Team mapping change | `data/team-mapping.md` | Add/remove mapping entry; flag any silent-mismatch risk. |
| Dashboard structure change | `ops/dashboard.md` | New tab, removed feature, callback change, schema. |
| Settlement logic change | `ops/settlement.md` | New parser, new path, race conditions resolved, market additions. |
| Domain term added/clarified | `CONTEXT.md` (**both** vault copy and repo copy — keep them in lockstep) | Definition + example usage. |
| Architecture refactor | `progress.md` + sometimes `decisions_Bet.md` | Progress: what moved where. Decisions: only if the refactor encodes a new principle. |
| Off-season state change (e.g. retrain flags flipped) | `BetBot_index.md` Current status | Tick / add the status emoji. |
| Scheduled task added/changed | `progress.md` + `ops/settlement.md` if settlement-adjacent | Document the task name, trigger schedule, what it runs. |

## Procedure

When invoked:

1. **Identify the change window.** Ask the user what to sync (e.g. "this session", "since the last vault `updated:` date", "since commit X"). If unspecified, default to "this session".

2. **Collect changes from every source.**
   - `git log --oneline <ref>..HEAD` for commits in window
   - `git diff --name-status <ref>..HEAD` for files touched
   - Memory file additions (`~\.claude\projects\C--Users-joshc-OneDrive-Documents-Project\memory\*.md`) since the window started
   - Findings raised in conversation (open questions, severity flags, decisions)
   - Scheduled-task changes via `Get-ScheduledTask` if relevant

3. **Classify each change against the discrimination map.** Build a per-note edit list. One change can belong to multiple notes (e.g. a fix that also resolves a hardening item touches `progress.md` AND `BetBot_index.md` AND `pre_live_hardening.md`).

4. **Draft proposed updates.** For each affected note:
   - Read the current file
   - Find the right section (don't dump at the bottom — match the file's structure)
   - Draft the edit as an addition where possible (almost never delete)
   - Bump the `updated:` field in YAML frontmatter to today's date
   - Match the file's existing voice and link style

5. **Present as a checklist.** Show the user:
   - Per-note: file path, section, draft diff
   - Total notes affected
   - Any notes you considered but decided not to touch (with reason)

6. **Apply on approval.** Use `Edit` with surgical, minimal diffs. If a note has multiple updates, do them in one Edit call where possible to avoid drift.

7. **Confirm the result.** Summarise what landed, list any wikilinks that now resolve to aspirational (not-yet-written) notes so the user can decide whether to write them.

## Voice and formatting conventions (match these exactly)

- **Wikilinks:** `[[BetBot_index]]`, `[[decisions_Bet#Section name]]`, `[[markets/ou25]]`, `[[ops/dashboard]]`. Note that `markets 1/btts.md` exists as a duplicate and is wikilinked from index as `[[Projects/Bet Bot/markets 1/btts]]` — do not change that link without user approval.
- **Date format:**
  - In YAML frontmatter `updated:` field → ISO `2026-05-26`
  - In headings inside progress.md → human `26 May 2026`
- **Status emoji in `BetBot_index.md`:** `:white_check_mark:` (done), `:construction:` (in progress), `:no_entry_sign:` (blocked), `:calendar:` (scheduled / future).
- **Code examples** in `decisions_Bet.md` use realistic code from the actual project, not pseudocode. Strip imports/boilerplate.
- **Decision sections** follow the pattern: motivation → solution (with code) → trade-offs / caveats. Not just "we chose X".
- **Progress entries** start with the commit hash (or hashes), then a one-line context, then a `**What was built / fixed:**` bullet list with wikilinks.

## When NOT to use this skill

- Pure documentation sessions where no code changed and no findings surfaced.
- Trivial commits (typo fixes, comment-only changes) — vault doesn't need an entry for every commit, only for ones that materially change what someone needs to know about the system.
- Mid-session, before the work is actually done. Vault sync goes at the end so the entries reflect what *actually* shipped, not what was planned.
- If the user is explicitly in `/grill-with-docs` mode — that skill handles CONTEXT.md updates as part of its own workflow.

## Things this skill does NOT do

- Fabricate decisions or progress entries. Only describe what genuinely happened in the change window.
- Refactor or restructure existing notes. Only append within existing sections, or add new sub-sections where the file's structure invites it.
- Touch `markets 1/btts.md` (rename artefact — flag to user if it seems relevant).
- Sync CONTEXT.md changes in only one location — the vault copy and the repo copy must be updated together if either changes.
- Push to git or create commits. Vault is not version-controlled in this project's git repo; it lives in OneDrive and syncs separately.

## Map maintenance

If the vault gains a new note type, restructures, or a new project area is added (e.g. a `staking/` folder), the discrimination map in this skill needs updating. The map is the authoritative reference for "what goes where" — when the map and the vault disagree, the vault wins and the map gets updated. Flag any drift to the user.

## Relationship to other skills

- **`/handoff`** — vault sync should run before handoff so the next session starts from a current vault. Suggest `/sync-vault` proactively when the user invokes `/handoff` and the vault hasn't been touched in the current session.
- **`/grill-with-docs`** — handles `CONTEXT.md` updates as part of its own workflow. Do not duplicate that work here; just ensure the vault copy of `CONTEXT.md` matches the repo copy after `/grill-with-docs` finishes.
- **`/gstack-document-release`** — global skill that updates project READMEs and ARCHITECTURE.md post-ship. `/sync-vault` is the vault-specific complement; both can run after a release.
- **`pre-scan`** — project skill that runs at the start of data-pipeline sessions. Surfaces issues that often become vault-worthy findings.
