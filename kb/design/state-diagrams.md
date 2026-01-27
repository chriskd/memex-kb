---
title: MX State Diagrams
description: State diagrams for major memex CLI flows
tags: [memex, architecture, flows, state-diagrams]
created: 2026-01-25
---

# MX State Diagrams

These diagrams capture the primary MX 0.2.0 user journeys and the key states that
govern behavior across CLI, storage, and publishing.

## Init and KB Discovery

### mx init (project or user scope)

```mermaid
stateDiagram-v2
    [*] --> ParseArgs
    ParseArgs --> InvalidOptions: --user + --path
    InvalidOptions --> [*]
    ParseArgs --> SelectTarget
    SelectTarget --> ExistsCheck
    ExistsCheck --> ErrorExists: exists and !--force
    ErrorExists --> [*]
    ExistsCheck --> CreateKB: not exists or --force
    CreateKB --> WriteReadme
    WriteReadme --> WriteConfig
    WriteConfig --> Done
    Done --> [*]
```

Notes:
- Project scope writes `.kbconfig` at the repo root with `kb_path`.
- User scope writes `.kbconfig` inside `~/.memex/kb/`.

### KB discovery (reads) + context overlays

```mermaid
stateDiagram-v2
    [*] --> NeedKB
    NeedKB --> CheckProject: find .kbconfig up tree (unless MEMEX_SKIP_PROJECT_KB)
    CheckProject --> ProjectFound: kb_path resolved
    CheckProject --> CheckUser: not found
    CheckUser --> UserFound: MEMEX_USER_KB_ROOT or ~/.memex/kb with .kbconfig
    CheckUser --> NoKB: none
    ProjectFound --> UseProject
    UserFound --> UseUser
    NoKB --> Error
```

```mermaid
stateDiagram-v2
    [*] --> WalkUp
    WalkUp --> LoadContext: .kbconfig found
    WalkUp --> NoContext: none
    LoadContext --> [*]
    NoContext --> [*]
```

Notes:
- Context overlays supply `primary`, `boost_paths`, `default_tags`, and `publish_base_url`.

## Add, Update, Patch

### mx add (create entry)

```mermaid
stateDiagram-v2
    [*] --> ResolveKB
    ResolveKB --> DiscoverContext
    DiscoverContext --> ChooseTarget
    ChooseTarget --> UseDirectory: --directory
    ChooseTarget --> UseCategory: --category
    ChooseTarget --> UsePrimary: context.primary
    ChooseTarget --> ErrorMissingTarget: none
    UseDirectory --> ValidateDirectory
    UseCategory --> EnsureCategoryDir
    UsePrimary --> EnsurePrimaryDir
    ValidateDirectory --> ValidateTags
    EnsureCategoryDir --> ValidateTags
    EnsurePrimaryDir --> ValidateTags
    ValidateTags --> SlugifyTitle
    SlugifyTitle --> ExistsCheck
    ExistsCheck --> ErrorExists: file exists
    ExistsCheck --> WriteEntry: available
    WriteEntry --> IndexEntry
    IndexEntry --> MaybeSemanticLinks
    MaybeSemanticLinks --> Done
    Done --> [*]
```

Notes:
- `--category` (or `--directory`) is required unless `primary` is set in `.kbconfig`.
- Writes rebuild backlinks and reindex the new entry.

### mx replace / patch / append (update entry)

```mermaid
stateDiagram-v2
    [*] --> ResolvePath
    ResolvePath --> ParseEntry
    ParseEntry --> ApplyEdits
    ApplyEdits --> WriteEntry
    WriteEntry --> ReindexEntry
    ReindexEntry --> Done
    Done --> [*]
```

Notes:
- Paths accept `@project/` and `@user/` prefixes.
- Edits update frontmatter metadata and reindex the entry.

## Search + Neighbors (Relations Graph)

```mermaid
stateDiagram-v2
    [*] --> ValidateKB
    ValidateKB --> RunSearch
    RunSearch --> ApplyMinScore
    ApplyMinScore --> IncludeNeighbors: --include-neighbors
    ApplyMinScore --> FormatOutput: no neighbors
    IncludeNeighbors --> ExpandFromSemanticLinks
    IncludeNeighbors --> ExpandFromTypedRelations
    IncludeNeighbors --> ExpandFromWikilinks
    ExpandFromSemanticLinks --> MergeResults
    ExpandFromTypedRelations --> MergeResults
    ExpandFromWikilinks --> MergeResults
    MergeResults --> FormatOutput
    FormatOutput --> [*]
```

Notes:
- `--scope` limits the KB roots used for indexing and search; single-KB mode returns unscoped paths.
- Neighbor expansion uses semantic links, typed relations, and wikilinks.

## Typed Relations + Publish Rendering

### mx relations-add / relations-remove

```mermaid
stateDiagram-v2
    [*] --> ResolveEntryPath
    ResolveEntryPath --> ParseEntry
    ParseEntry --> UpdateRelations
    UpdateRelations --> WriteFrontmatter
    WriteFrontmatter --> ReindexEntry
    ReindexEntry --> Done
    Done --> [*]
```

### mx publish

```mermaid
stateDiagram-v2
    [*] --> ResolveKBSource
    ResolveKBSource --> UseKbRoot: --kb-root
    ResolveKBSource --> UseScope: --scope
    ResolveKBSource --> UseContext: .kbconfig/.kbconfig project_kb or kb_path
    ResolveKBSource --> ErrorNoKB: none
    UseKbRoot --> LoadEntries
    UseScope --> LoadEntries
    UseContext --> LoadEntries
    LoadEntries --> BuildIndices
    BuildIndices --> RenderPages
    RenderPages --> BuildGraph
    BuildGraph --> WriteSearchIndex
    WriteSearchIndex --> Done
    Done --> [*]
```

Notes:
- Typed relation targets are normalized per-scope; cross-scope targets are skipped during publish.
- Publish outputs HTML pages, tag indexes, search index, and graph data.
