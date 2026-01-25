---
title: ChatGPT 5.2 Pro Dense Memory Signals Reference
description: Research prompt and notes on dense memory signals for agent context injection
tags:
  - reference
  - ai
  - memory
  - signals
created: 2026-01-25
---

My prompt:



  Research Prompt: Dense Memory Signals for Agent Context Injection

  Context

  I'm building an agent memory system for LLM-based coding assistants (like Claude Code, Cursor, Codex). The core problem: when an agent's context window is cleared or a new session starts, valuable context is lost—decisions made, gotchas discovered, work in progress, etc.

  Dense memory signals are compact, high-information-density snippets that can be injected at session start to restore context without consuming many tokens. The goal is to find signals with the best ratio of usefulness to token cost.

  What I Already Know

  I've researched existing tools and identified some signals:

  Tier 1 (High Value, Low Tokens):
  - Decisions made + rationale (~15 tokens, critical—LLM can't infer this)
  - Gotchas/warnings discovered (~10 tokens, prevents repeated mistakes)
  - User corrections (~10 tokens, direct mistake prevention)
  - Git staged files (~5 tokens, interrupted work signal)
  - Git stash contents (~5 tokens, parked work context)
  - Branch name (~3 tokens, feature context)
  - Build status (~3 tokens, "FAILING" changes everything)
  - Files read but NOT modified (~10 tokens, research context)

  Tier 2 (Medium Value):
  - Files touched with context (~15 tokens)
  - Summary of what was done (~25 tokens)
  - Questions the LLM asked (~15 tokens)
  - Tool usage patterns (~10 tokens)
  - Errors encountered (~15 tokens)
  - TODOs added/removed (~10 tokens)

  Existing tools' approaches:
  - claude-mem: Causal observations (what led to what)
  - Mem0: Extract facts, not transcripts (~90% token reduction)
  - A-Mem (NeurIPS 2025): Atomic notes with metadata, embeddings, links (85% reduction)
  - Letta/MemGPT: Message buffer + core memory blocks + archival

  What I Need Help With

  1. Identify non-obvious signals I'm missing:
  - What context does a developer carry in their head that an LLM loses between sessions?
  - What signals are available from git, IDE state, project structure, CI/CD, etc. that aren't commonly captured?
  - What "meta" signals exist (patterns across sessions, user habits, team conventions)?

  2. Evaluate signal properties:
  For each signal you identify, estimate:
  - Token cost: How many tokens to represent it?
  - Information value: High/Medium/Low—how much does this help the agent?
  - Extraction difficulty: Easy (automated) / Medium (heuristics) / Hard (requires LLM inference)
  - Uniqueness: Can the LLM infer this from code, or is it hidden context only humans know?

  3. Consider the "anti-signals":
  - What signals seem useful but actually waste tokens or mislead the agent?
  - What context should NOT be injected (too noisy, too stale, inferrable from code)?

  4. Explore scaling strategies:
  When an agent touches 50+ files across many sessions, raw signal lists explode. How should signals be:
  - Summarized hierarchically (file → directory → project level)
  - Decayed over time (recent = full detail, old = compressed)
  - Clustered semantically (group related signals)
  - Filtered by relevance to current task

  5. Cross-domain inspiration:
  - What do human developers do to restore context? (Notes, git commit messages, PR descriptions, etc.)
  - What do other domains do for context continuity? (Medical handoffs, shift changes, project management)
  - What research exists on human memory cues that could apply?

  Constraints

  - Token budget: The injection should be ~100-300 tokens for a "dense" format, expandable to ~500-1000 for "rich" format
  - Extraction source: Primarily git history, file system, session logs (parsed conversation transcripts), and project config
  - No user action required: Signals should be automatically extractable without user tagging or annotation

  Desired Output

  Please provide:

  1. A prioritized list of signals I haven't considered, with the properties above
  2. Critiques of my current signal list—what's overrated, underrated, or missing nuance?
  3. A proposed signal taxonomy—how would you categorize signals (by source? by value? by extraction method?)
  4. Scaling recommendations for when signal volume gets high
  5. Any relevant research, papers, or prior art I should investigate

  Example Output Format

  ## Signal: [Name]
  - **Description**: What this captures
  - **Token cost**: ~N tokens
  - **Value**: High/Medium/Low—why
  - **Extraction**: Easy/Medium/Hard—how
  - **Uniqueness**: Can LLM infer this? Why/why not?
  - **Example**: "Branch: feat/caching | Stash: 'WIP: redis connection pooling'"

  ---
  That prompt is self-contained and should elicit useful research contributions from other LLMs. You could also ask specific follow-up questions like:

  - "Focus specifically on git-derived signals I might be missing"
  - "What signals exist in IDE state (cursor position, open tabs, breakpoints)?"
  - "How do enterprise knowledge management systems handle context handoff?"



======++++++REPLY+++++=====
You’re already thinking in the right direction: most “lost context” isn’t *code*, it’s the invisible stuff a dev (or agent) just learned: **what we’re trying to accomplish, what we tried that failed, what we decided, what’s currently broken, and what the next step is**.

Below are (1) high-leverage signals you’re missing, (2) critiques of your list, (3) a taxonomy that’s practical for extraction + ranking, (4) scaling tactics when the signal list explodes, and (5) research/prior art worth mining.

---

## 1) Prioritized “missing” dense signals

I’m prioritizing by *usefulness ÷ tokens*, and by “can’t be inferred from code”.

### Signal: Next Step / Open Loop

* **Description**: The single most valuable handoff cue: what the agent was about to do next (or what remains undone).
* **Token cost**: ~10–25
* **Value**: **High** — eliminates “re-figure out what I was doing” time.
* **Extraction**: **Medium** — infer from final transcript turns + last tool calls + last git changes.
* **Uniqueness**: **High** — not reliably inferable from repo state.
* **Example**: “Next: fix failing `TestCacheEviction::ttl_expired` by adjusting clock mock in `cache_test.go`.”

---

### Signal: Current Blocker + Why

* **Description**: What’s preventing progress right now (missing secret, flaky test, unclear spec, reproducer unavailable).
* **Token cost**: ~10–25
* **Value**: **High** — prevents the classic “repeat same dead end”.
* **Extraction**: **Medium** — parse last error outputs + “I can’t because …” statements.
* **Uniqueness**: **High** — often external or transient.
* **Example**: “Blocked: can’t run integration tests—`REDIS_URL` missing locally.”

---

### Signal: Merge/Rebase/Conflict State

* **Description**: Whether git is mid-merge/rebase/cherry-pick and which files are conflicted.
* **Token cost**: ~8–20
* **Value**: **High** — changes everything about the next actions.
* **Extraction**: **Easy** — detect `.git/MERGE_HEAD`, `.git/rebase-apply`, `git status`.
* **Uniqueness**: **High** — not in code; pure workflow state.
* **Example**: “REBASE in progress; conflicts: `src/api.ts`, `package-lock.json`.”

---

### Signal: Dirty Working Tree Diffstat (Staged + Unstaged + Untracked)

* **Description**: A compact snapshot of “how much is in flight” + where, without listing every file.
* **Token cost**: ~10–20
* **Value**: **High** — signals incomplete work and risk.
* **Extraction**: **Easy** — `git status --porcelain`, `git diff --stat`, `git diff --cached --stat`.
* **Uniqueness**: **Medium** — partially inferable by looking at repo, but not from code alone.
* **Example**: “Dirty: staged 2 files (+45/-10), unstaged 5 (+120/-40), untracked: 1.”

---

### Signal: Ahead/Behind + Unpushed Commits Titles

* **Description**: Whether local work is unpublished + *what it is* (commit subject lines).
* **Token cost**: ~10–25 (top 1–3 commit subjects)
* **Value**: **High** — tells the agent “the story so far”.
* **Extraction**: **Easy** — `git status -sb`, `git log @{u}.. --oneline -n 3`.
* **Uniqueness**: **Medium** — code contains changes, but intent is compressed in subjects.
* **Example**: “Ahead 2: `Add redis pool`, `Fix TTL race in eviction`.”

---

### Signal: Last Command(s) Run + Exit Code

* **Description**: The most recent build/test/lint/dev-server command and whether it succeeded.
* **Token cost**: ~10–30
* **Value**: **High** — beats a vague “build failing”.
* **Extraction**: **Medium** — capture terminal/tool logs; normalize + redact paths.
* **Uniqueness**: **High** — not deducible from code.
* **Example**: “Last: `pytest -k cache` → FAIL (3), `ruff check` → OK.”

---

### Signal: Failure Signature (Canonical Error Fingerprint)

* **Description**: One-line normalized summary of the *dominant* failure (error type + location + key message).
* **Token cost**: ~8–18
* **Value**: **High** — rehydrates debugging context instantly.
* **Extraction**: **Medium** — parse logs; dedupe stack traces; pick top recurring signature.
* **Uniqueness**: **High** — ephemeral runtime info.
* **Example**: “FAIL: `KeyError: 'user_id'` at `src/auth.py:112` (in `LoginHandler`).”

---

### Signal: Active Working Set (Open Tabs + Primary Cursor Location)

* **Description**: The “mental desktop”: which files were open and where the cursor last was.
* **Token cost**: ~15–40 (top 3–8 files + 1–3 cursor spots)
* **Value**: **High** — humans use this as their fastest context cue.
* **Extraction**: **Easy/Medium** — easy if IDE exposes state; medium if inferred from recent reads/edits.
* **Uniqueness**: **High** — not in repo; purely session state.
* **Example**: “Focus: `src/cache/pool.ts:188`; tabs: `pool.ts`, `redis.ts`, `cache.test.ts`.”

*Why I’m so confident this matters:* task-context tooling in IDEs (e.g., Eclipse Mylyn / Mylar) explicitly models “task context” from developer interaction and uses it to focus what you see. That’s basically a human “dense memory signal” system. ([Eclipse][1])

---

### Signal: Navigation Trace (Symbols/Paths Investigated)

* **Description**: A compressed “investigation trail”: key symbols, call chain, or files traversed during debugging.
* **Token cost**: ~15–40
* **Value**: **High** — prevents re-reading the same rabbit hole; restores the mental model.
* **Extraction**: **Medium** — from IDE navigation events / “go to definition” logs / conversation + file reads.
* **Uniqueness**: **Medium/High** — code exists, but “what we already checked” isn’t.
* **Example**: “Trace: `RequestHandler→AuthSvc→TokenRepo`; checked `JWTClaims` parsing; not culprit.”

---

### Signal: Environment / Toolchain Fingerprint

* **Description**: The minimal environment facts that commonly break builds: language versions, package manager, OS/arch, container vs host.
* **Token cost**: ~10–30
* **Value**: **High** (conditional) — huge when mismatch caused the bug.
* **Extraction**: **Easy/Medium** — read `.tool-versions`, `.nvmrc`, `pyproject`, plus runtime introspection.
* **Uniqueness**: **High** — cannot infer from code reliably.
* **Example**: “Env: Node 20, pnpm 9; Python 3.12 venv active; running in Docker.”

---

### Signal: Local Services / Containers / Ports Running

* **Description**: What’s currently running locally (db, redis, queues) + which compose profile.
* **Token cost**: ~10–25
* **Value**: **High** — avoids “why can’t I connect” loops.
* **Extraction**: **Medium** — `docker ps`, compose state, dev scripts logs.
* **Uniqueness**: **High** — pure runtime state.
* **Example**: “Services: `docker compose up redis,postgres` (ports 6379/5432).”

---

### Signal: “Hidden Config Changed” Flag

* **Description**: Detect if developer touched config files that are often excluded from commits (`.env`, `config/local.*`) and summarize *that* without leaking secrets.
* **Token cost**: ~8–20
* **Value**: **High** — explains behavior differences.
* **Extraction**: **Easy** for “changed/not changed”; **Hard** to safely summarize content (needs redaction rules).
* **Uniqueness**: **High**
* **Example**: “Local config changed: `.env` edited (secrets redacted); `FEATURE_X=true` toggled.”

---

### Signal: Migration / Schema State

* **Description**: Pending migrations, last applied migration, or “DB schema drift”.
* **Token cost**: ~10–30
* **Value**: **High** for backend work — prevents subtle runtime failures.
* **Extraction**: **Medium** — framework-specific (Django/Alembic/Prisma/Flyway/etc.).
* **Uniqueness**: **High** — not deducible from code alone.
* **Example**: “Pending: `20260114_add_user_idx.sql`; not applied locally.”

---

### Signal: Dependency Delta Summary

* **Description**: “What dependencies changed” in one line (esp. lockfile changes), or “this PR is mostly dependency churn”.
* **Token cost**: ~10–30
* **Value**: **High** — clarifies risk + why diffs are huge.
* **Extraction**: **Medium** — parse lockfile diff; choose top N packages by impact.
* **Uniqueness**: **Medium/High** — code shows lockfile changes but not the intention/impact.
* **Example**: “Deps: bump `redis` 5.0→5.2, `pydantic` 2.6→2.7.”

---

### Signal: Repo Workflow Contract (How to Build/Test/Lint Here)

* **Description**: The “rules of the repo” distilled from configs: how to run tests, formatters, linters, pre-commit, CI entrypoints.
* **Token cost**: ~20–60
* **Value**: **High** — prevents agent doing the wrong thing or inventing commands.
* **Extraction**: **Easy/Medium** — read `Makefile`, `package.json`, `pyproject`, `.pre-commit-config.yaml`, CI YAML.
* **Uniqueness**: **Medium** — inferable by reading files, but costly to rediscover each session.
* **Example**: “Workflow: `make test`, `make lint`; fmt via `ruff format`; CI uses `pytest -m 'not slow'`.”

---

### Signal: “Tried & Failed” Attempts (Dead Ends Register)

* **Description**: A small list of approaches attempted and why they were rejected (e.g., “tried caching at layer X; broke invariants”).
* **Token cost**: ~20–60
* **Value**: **High** — huge loop-prevention payoff.
* **Extraction**: **Hard** — requires transcript understanding + causality extraction.
* **Uniqueness**: **High** — not in code; prevents repeated exploration.
* **Example**: “Tried: memoize `getUser()` (broke auth refresh). Rejected: global singleton cache (tests flaky).”

This is very aligned with “causal observations” approaches, and with research emphasizing structured, consolidated memory rather than transcripts. ([GitHub][2])

---

### Signal: Assumptions Ledger (Validated vs Unvalidated)

* **Description**: Short list of key assumptions and whether they were confirmed.
* **Token cost**: ~15–40
* **Value**: **Medium/High** — helps agent reason without hallucinating constraints.
* **Extraction**: **Hard** — needs inference + confidence tagging.
* **Uniqueness**: **High**
* **Example**: “Assumptions: Redis available in prod ✅; TTL must be per-tenant ✅; clock monotonic ❓.”

---

### Signal: Compatibility / Constraints Snapshot

* **Description**: Non-negotiables: supported versions, API compatibility, performance budget, security constraints.
* **Token cost**: ~15–50
* **Value**: **High**
* **Extraction**: **Medium** — configs + CI matrix + transcript.
* **Uniqueness**: **High** — often not explicit in code.
* **Example**: “Constraints: must support Python 3.10–3.12; no new deps; keep API stable.”

---

### Signal: Ownership / Review Surface (CODEOWNERS + “touches risky modules”)

* **Description**: A one-line hint about who owns touched areas, or “this change affects auth/billing”.
* **Token cost**: ~10–25
* **Value**: **Medium/High** — improves decisions about scope and safety.
* **Extraction**: **Easy/Medium** — parse CODEOWNERS + changed paths.
* **Uniqueness**: **Medium** — not inferable from code logic.
* **Example**: “Touches `auth/*` (owned by @security-team).”

---

### Signal: Work Item / PR Anchor (If Derivable)

* **Description**: Issue/PR number and title, if encoded in branch/commit messages (or local PR metadata).
* **Token cost**: ~8–25
* **Value**: **Medium/High** — restores “why this exists”.
* **Extraction**: **Medium** — regex branch names, parse commit subjects.
* **Uniqueness**: **High** — intent signal.
* **Example**: “Work item: `#4821 Cache token introspection results`.”

---

### Signal: “Task Context Model” (Ranked code elements, not file lists)

* **Description**: Instead of “50 touched files”, store a *ranked* set of relevant elements (files + symbols) with a “degree-of-interest” score.
* **Token cost**: ~30–80
* **Value**: **High** at scale — fights token blowups.
* **Extraction**: **Medium/Hard** — requires interaction logging + scoring.
* **Uniqueness**: **High**
* **Example**: “Context(top): `CachePool.acquire`, `RedisClient.connect`, `TTLClock.now`…”

This is directly inspired by task-focused interface research/tooling (Mylyn/Mylar) and newer “code context model” work (e.g., the ICSE’23 “TASK CONTEXT” tool). ([Eclipse][1])

---

### Signal: “Gist Episode” for a Session (Ultra-compressed)

* **Description**: 1–3 sentence gist of what happened, plus pointers/IDs to full details if needed.
* **Token cost**: ~40–90
* **Value**: **Medium/High** — good as a fallback when you have many sessions.
* **Extraction**: **Hard** — needs summarization that preserves decisions + blockers.
* **Uniqueness**: **Medium** — overlaps with other signals, but good for retrieval.
* **Example**: “Session gist: Implemented redis pool; tests failing due to mocked time; next fix in `cache_test.go`.”

This resembles “gist memory” ideas (compress episodes and retrieve details when needed). ([arXiv][3])

---

## 2) Critiques of your current signal list (overrated / underrated / nuance)

### What’s underrated or missing nuance

* **Build status (~3 tokens)**: totally right that “FAILING changes everything”, but **“FAILING” without *which command/target + top failing signature*** is often too lossy. I’d promote:

  * “Last command + exit code”
  * “Canonical failure signature”
  * “Top failing tests”
* **Git staged files / stash**: excellent, but incomplete without:

  * **unstaged + untracked counts** (people often forget they exist)
  * **diffstat** (size matters a lot for risk + mental rehydration)
* **Decisions + rationale**: best Tier 1 item. Small addition that increases value a lot:

  * include **“alternatives rejected”** (even 3–6 tokens) and whether decision is **reversible**.
* **Errors encountered (~15 tokens)**: make it *one canonical signature*, not a rotating list of stack traces.

### What’s potentially overrated (or easy to accidentally bloat)

* **Files read but NOT modified (~10 tokens)**: useful, but **very easy to explode** and often too low-signal if the agent skimmed a bunch of stuff. I’d compress it as:

  * “investigation trace” (top 3–5 files/symbols)
  * or directory-level cluster (“looked in `src/auth/*`”)
* **Questions the LLM asked (~15 tokens)**: keep only **unanswered** questions or “open unknowns”, not every question.
* **Tool usage patterns (~10 tokens)**: usually low value unless it encodes something actionable (“we can’t use docker here”, “no network”, “tests too slow so we used unit-only”).

### One subtle “gotcha” with dense signals

Some signals become harmful if they’re **stale but sound authoritative**. Any inferred signal (“we think cause is X”) should carry:

* a timestamp or commit anchor, and
* a confidence flag (“hypothesis” vs “confirmed”).

---

## 3) Proposed signal taxonomy

A taxonomy that stays useful under scaling usually needs *two axes*: **(A) what the signal means** and **(B) how stable it is**.

### Axis A: Semantic category

1. **Intent & Progress**

   * goal, next step, done/remaining, blockers
2. **Decisions & Constraints**

   * decisions + rationale, rejected alternatives, invariants, compatibility requirements
3. **Execution State**

   * last commands, failing signature/tests, running services, CI status
4. **Repo / Workflow State**

   * branch/upstream, ahead/behind, diffstat, merge/rebase state, stash
5. **Task Context Model**

   * ranked files/symbols, navigation trace, open tabs/cursor, search terms
6. **Risk Register**

   * known pitfalls, security/perf risks, “touches auth/billing”
7. **Human/Team Conventions**

   * lint/format rules, codeowners, review norms, user preferences/corrections

### Axis B: Stability / half-life

* **Ephemeral (minutes–hours)**: cursor position, open tabs, running containers, last command
* **Session-stable (hours–days)**: current blocker, failing tests, “next step”, diffstat
* **Task-stable (days–weeks)**: decisions, constraints, gotchas, rejected approaches
* **Project-stable (weeks–months)**: conventions, build workflow, codeowners

Why this matters: you can allocate token budget by half-life. Old ephemeral stuff should decay to zero aggressively.

### Bonus axis that helps ranking: “inferability”

* **Non-inferable** (inject aggressively): decisions, blockers, failed attempts, runtime errors
* **Inferable** (inject only if cheap or high relevance): file lists, summaries of changes, tool usage

---

## 4) Scaling recommendations when signal volume explodes

When you hit “50+ files across many sessions”, you need to stop thinking “list” and start thinking **index + retrieval + salience** (your own notes already point here; claude-mem also emphasizes progressive disclosure and layered retrieval). ([GitHub][4])

### A. Budget by category, not by raw count

Example dense budget (≈200–250 tokens total):

* 60 tokens: Decisions/Constraints (non-inferable, durable)
* 60 tokens: State (git + build/test + blockers)
* 60 tokens: Next step + top failure signature + open unknowns
* 30 tokens: Task context model (top 3 files/symbols)

This prevents “files touched” from eating the whole injection.

### B. Hierarchical summarization: file → directory → subsystem

For touched/read files:

* **Level 0**: “Touched: 47 files”
* **Level 1**: “`src/cache/*` 18, `src/auth/*` 9, `infra/*` 6…”
* **Level 2 (only for top clusters)**: list top 3–5 files *with a 3–6 token role label*:

  * “`src/cache/pool.ts` (pool logic), `redis.ts` (client), `cache.test.ts` (TTL tests)”

Role labels are cheap and massively improve usefulness.

### C. Decay by time *and* by stability

Use half-lives:

* cursor/open tabs: half-life ~2 hours
* build failures: half-life until next green run
* decisions: half-life weeks, but supersede if contradicted

MemoryBank explicitly explores time-based updating inspired by forgetting curves (and more broadly, time/decay is a common memory-system lever). ([arXiv][5])

### D. Cluster semantically + store “gist episodes”

Store per-session “gist” plus structured atomic notes. A-Mem is a concrete example of structuring notes with attributes/tags/keywords and then linking them into a network that can evolve as new memories arrive. ([arXiv][6])

Practical implementation for your case:

* Create **atomic notes** for decisions, gotchas, blockers, failed attempts.
* Create **episodes** for each work session (1–3 sentence gist + pointers).
* Link episodes ↔ atomic notes ↔ files/directories.

### E. Relevance filtering: don’t inject until you know the user’s next request

If your system can inject after the first user prompt (or can re-rank right after it), do:

1. Inject only **project-stable + last-known critical state** at session start (≈80–150 tokens).
2. After user prompt, fetch task-relevant notes (top-K by similarity + recency + type weight).

This is essentially “progressive disclosure”: start with an index and pull details when needed (claude-mem’s MCP workflow is explicitly built around this idea). ([GitHub][4])

### F. Scoring function for what makes it into the 100–300 token capsule

A simple salience score you can implement without an LLM:

* **BaseWeight(type)** × **RecencyDecay** × **UniquenessBoost** × **RiskBoost** × **TaskMatch**

Where:

* **UniquenessBoost** high for decisions/blockers/errors/failed attempts
* **RiskBoost** high for auth/billing/migrations/infra
* **TaskMatch** from branch name + current prompt + current file focus

Then pick top-K per category to preserve diversity.

### G. Explicit supersession + confidence tagging

Every memory item should support:

* `supersedes=<id>` and/or `status=deprecated`
* `confidence={hypothesis|likely|confirmed}`
* `anchor={commit sha | timestamp | build id}`

This prevents stale cues from becoming misinformation.

---

## 5) Cross-domain inspiration (human handoffs) you can steal shamelessly

### Use SBAR / I-PASS as the *format* of your injection capsule

Medical handoff tools exist because handoffs are exactly the “context loss” problem you’re describing.

* **SBAR** = *Situation, Background, Assessment, Recommendation* — an intentionally compressed structure for fast, high-signal transfer. ([Springer][7])
* **I-PASS** is another evidence-based handoff bundle associated with reduced medical errors in studies (often cited ~23% reduction in medical errors / ~30% in preventable adverse events, depending on study/design). ([Springer][8])

Mapping to coding sessions:

* **Situation**: current objective + severity (is the build broken? is prod impacted?)
* **Background**: decisions, constraints, relevant context
* **Assessment**: what’s true now (state + failures + what we tried)
* **Recommendation**: next step(s) + contingencies

### Safety-critical “shift handover” checklists

Industries like chemical plants treat shift handover as a high-risk moment and use structured logs/checklists emphasizing “ongoing tasks”, “abnormal conditions”, and “explicitly acknowledge nothing to report.” ([Process Solutions][9])

Your analog checklist items:

* ongoing WIP (diffstat/staged/stash)
* abnormal conditions (failing tests, merge conflicts, migrations pending)
* safety issues (security concerns, secrets, data handling constraints)
* operational limits (compatibility constraints)

### Human memory cue research: context cues + resumption cues

* **Context-dependent memory / encoding specificity**: recall improves when retrieval cues match encoding context (classic example: Godden & Baddeley divers study; and broader encoding specificity principle). ([app.nova.edu][10])
  *Implication*: injecting **the same cues the agent used while working** (file names, symbols, error signatures, command lines) is disproportionately effective for recall.
* **Interrupted-task effects** are more nuanced than pop-psych suggests; a 2025 meta-analysis reports no reliable memory advantage for unfinished tasks but does find a general tendency to resume (Ovsiankina effect). ([Nature][11])
  *Implication*: your “Next step / open loop” cue is still powerful—but treat it as a **resumption trigger**, not as guaranteed enhanced recall.

---

## 6) Research / prior art to investigate (highly relevant to your design)

### Agent / LLM memory systems

* **MemGPT** (hierarchical memory / paging-inspired) and the associated Letta architecture docs (core vs archival, etc.). ([arXiv][12])
* **Mem0 (2025)**: focuses on extracting + consolidating salient memory (and a graph variant), reporting big token savings and latency reductions. ([arXiv][13])
* **A-Mem (NeurIPS 2025)**: Zettelkasten-inspired structured notes + dynamic linking + memory evolution as new notes arrive. ([arXiv][6])
* **MemoryBank (AAAI)**: long-term memory mechanism + time-based updating inspired by forgetting curves. ([arXiv][5])
* **ReadAgent (2024)**: “gist memory” episodes + the ability to look up details in the original text. ([arXiv][3])
* **LoCoMo benchmark** for evaluating very long-term conversational memory across sessions. ([ACL Anthology][14])

### Coding-context continuity tooling (pre-LLM, but directly applicable)

* **Eclipse Mylyn / Mylar**: task context as a first-class object; DOI (“degree of interest”) ranking; restore task context when switching. ([Eclipse][1])
* **TASK CONTEXT (ICSE 2023 tool paper)**: predicting code context models for dev tasks based on navigation patterns. ([Zhiyuan Wan’s Home Page][15])
* **IDE session managers** like ContextKeeper (open documents, layout, breakpoints, bookmarks) as concrete examples of what humans need to restore “where was I?”. ([Visual Studio Marketplace][16])

### Practical memory plugin approaches in the LLM coding world

* **claude-mem**: progressive disclosure + layered retrieval (search → timeline → fetch details), plus privacy controls (exclude sensitive segments). ([GitHub][4])
* **Claude “memory tool” docs** (client-side persistent files) as another pattern for persistence. ([Claude][17])

---

## A “dense capsule” template (100–300 tokens) you can steal

Structured formats tend to compress well and reduce hallucination:

```
S: Goal=<…> | Severity=<GREEN/RED> | Branch=<…> | Ahead/Behind=<…>
B: Decisions=<…>; Constraints=<…>; Gotchas=<…>
A: State=Dirty(staged X/unstaged Y/untracked Z); LastCmd=<…>; FailSig=<…>
R: Next=<…>; BlockedOn=<…>; Focus=<file:line>; Trace=<sym→sym→sym>
```

(You can swap SBAR for I-PASS-style “Action list / Contingencies” if you prefer.)

---

If you want one very targeted follow-up direction (without adding user effort): I’d focus your next iteration on **Task Context Model signals** (ranked symbols/files) because that’s the cleanest way to beat “50-file explosion” while staying token-efficient—and there’s strong prior art in Mylyn/Mylar + newer code-context-model work to guide the scoring/representation. ([Eclipse][1])

[1]: https://eclipse.dev/mylyn/publications/2006-11-mylar-fse.pdf?utm_source=chatgpt.com "Using Task Context to Improve Programmer Productivity"
[2]: https://github.com/thedotmack/claude-mem?utm_source=chatgpt.com "GitHub - thedotmack/claude-mem: A Claude Code plugin that automatically ..."
[3]: https://arxiv.org/abs/2402.09727?utm_source=chatgpt.com "A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts"
[4]: https://github.com/thedotmack/claude-mem "GitHub - thedotmack/claude-mem: A Claude Code plugin that automatically captures everything Claude does during your coding sessions, compresses it with AI (using Claude's agent-sdk), and injects relevant context back into future sessions."
[5]: https://arxiv.org/abs/2305.10250?utm_source=chatgpt.com "MemoryBank: Enhancing Large Language Models with Long-Term Memory"
[6]: https://arxiv.org/abs/2502.12110?utm_source=chatgpt.com "A-MEM: Agentic Memory for LLM Agents"
[7]: https://link.springer.com/content/pdf/10.1186/s40886-018-0073-1.pdf?utm_source=chatgpt.com "Situation, Background, Assessment, Recommendation (SBAR) Communication ..."
[8]: https://link.springer.com/article/10.1186/s12909-024-05880-7?utm_source=chatgpt.com "Secondary analysis of hand-offs in internal medicine using the I-PASS ..."
[9]: https://process.honeywell.com/content/dam/process/en/documents/document-lists/doc_asm-consortium/white-papers/July%2031%202011%20-%20Effective%20Shift%20Handover.pdf?utm_source=chatgpt.com "Microsoft Word - Effective Shift HAndover_HCI2011 ... - Honeywell"
[10]: https://app.nova.edu/toolbox/instructionalproducts/edd8124/fall11/1975GoddenBaddeley.pdf?utm_source=chatgpt.com "CONTEXT-DEPENDENT MEMORY IN TWO NATURAL ENVIRONMENTS: ON LAND AND ..."
[11]: https://www.nature.com/articles/s41599-025-05000-w.pdf?utm_source=chatgpt.com "Interruption, recall and resumption: a meta-analysis of the Zeigarnik ..."
[12]: https://arxiv.org/pdf/2310.08560?utm_source=chatgpt.com "MemGPT: Towards LLMs as Operating Systems - arXiv.org"
[13]: https://arxiv.org/abs/2504.19413?utm_source=chatgpt.com "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"
[14]: https://aclanthology.org/2024.acl-long.747/?utm_source=chatgpt.com "Evaluating Very Long-Term Conversational Memory of LLM Agents"
[15]: https://zhiyuan-wan.github.io/assets/publications/icse_tool_23_task_context.pdf?utm_source=chatgpt.com "TASK CONTEXT: A Tool for Predicting Code Context Models for Software ..."
[16]: https://marketplace.visualstudio.com/items?itemName=ContextKeeper.ContextKeeperVsix&utm_source=chatgpt.com "Favorites Tabs and Files Manager - Visual Studio Marketplace"
[17]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool?utm_source=chatgpt.com "Memory tool - Claude Docs"
