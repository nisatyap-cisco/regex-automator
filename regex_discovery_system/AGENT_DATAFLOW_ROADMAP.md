# Full Pipeline Agent & Data Flow

## Table of Contents

1. [Agent 1P — URL Discovery & Policy Extraction](#agent-1p)
2. [Full Agent & Data Flow (All 5 Agents)](#full-data-flow)
3. [Config & Environment](#config)

---

## Agent 1P — URL Discovery & Policy Extraction

Agent 1P discovers policy URLs through Tavily search (10 diverse queries → up to
20 URLs), with DuckDuckGo fallback when Tavily is unavailable.

### URL Discovery Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AGENT 1P — URL DISCOVERY                         │
│                                                                      │
│  STAGE 1: Tavily Search (breadth — curated relevance)                │
│  ─────────────────────────────────────────────                       │
│  10 diverse queries (format spec, wikipedia, ISO, regex, etc.)       │
│  → max 20 candidate URLs                                             │
│  Each URL comes with: title, snippet, relevance_score from Tavily    │
│                                                                      │
│  STAGE 2: Rank + Quality Filter                                      │
│  ──────────────────────────────                                      │
│  1. Deduplicate by normalized URL                                    │
│  2. Score each URL:                                                  │
│     • Domain authority tier     (0–10)  — gov, edu, standards, wiki  │
│     • Path keyword relevance    (0–3)   — format, spec, rules, etc.  │
│  3. Cap at 2 URLs per domain (avoid over-representing one source)    │
│  4. Block social media, e-commerce, video platforms                  │
│  5. Prefer HTTPS over HTTP                                           │
│  6. Select top 15 by composite score                                 │
│                                                                      │
│  OUTPUT: 15 ranked, vetted URLs → proceed to scrape + LLM extract   │
└──────────────────────────────────────────────────────────────────────┘
```

### Fallback Chain

```
Tavily    Behavior
───────   ──────────────────────────────────────
  ✓       Tavily search (best)
  ✗       DDG fallback (current fallback path)
```

---

## Full Data Flow (All 5 Agents)

### High-Level Pipeline

```
USER INPUT: --condition "India PAN number"
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    main.py (Orchestrator)                           │
│                                                                     │
│   ThreadPoolExecutor(max_workers=2)                                │
│   ┌──────────────────────┐     ┌──────────────────────────────────┐│
│   │  AGENT 1             │     │  AGENT 1P                        ││
│   │  Data Collector      │     │  Policy Researcher               ││
│   │                      │     │                                  ││
│   │  1. Check db/        │     │  URL DISCOVERY:                  ││
│   │  2. Scrape web       │     │  ┌────────────────────────────┐  ││
│   │     (Tavily+Camoufox │     │  │ Tavily → 20 URLs           │  ││
│   │      → DDG+httpx     │     │  │ Rank+Dedupe → top 15       │  ││
│   │      → LLM synth)    │     │  └────────────────────────────┘  ││
│   │  3. Sanity filter    │     │  4. Scrape top 15 pages          ││
│   │                      │     │  5. LLM extracts policies        ││
│   │  4. Train/test split │     │  6. Vendor URL search+scrape     ││
│   │                      │     │  7. LLM extracts vendor patterns ││
│   │  OUTPUTS:            │     │  8. Fallback: LLM knowledge      ││
│   │  • train.txt         │     │                                  ││
│   │  • test.txt          │     │  OUTPUTS:                        ││
│   │  • scraped.xlsx      │     │  • policies[]                    ││
│   │                      │     │  • numbering_authority            ││
│   │                      │     │  • vendor_patterns[]              ││
│   │                      │     │  • vendor_keywords[]              ││
│   │                      │     │  • sources[] (URLs consulted)     ││
│   └──────────┬───────────┘     └──────────────┬───────────────────┘│
│              │                                │                     │
│              ▼                                ▼                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  source_of_truth.json  (merge of stats + policies)          │  │
│   └─────────────────────────────────┬───────────────────────────┘  │
│                                     │                              │
│              ┌──────────────────────▼────────────────────┐         │
│              │  AGENT 2 — Pattern Analyzer               │         │
│              │                                           │         │
│              │  INPUTS:                                  │         │
│              │  • train.txt     (real examples)          │         │
│              │  • policies[]    (official rules)         │         │
│              │  • vendor_patterns[] (DLP regex/rules)    │         │
│              │  • vendor_keywords[] (detection labels)   │         │
│              │                                           │         │
│              │  PROCESS:                                 │         │
│              │  1. Statistical pre-analysis (Python):    │         │
│              │     lengths, char classes, prefixes,      │         │
│              │     segments                              │         │
│              │  2. LLM deep analysis: stats + 200        │         │
│              │     samples + policies + vendor info      │         │
│              │                                           │         │
│              │  OUTPUTS:                                 │         │
│              │  • format_rules[]                         │         │
│              │  • range_restrictions[]                   │         │
│              │  • contextual_keywords[]                  │         │
│              │  • keyword_proximity                      │         │
│              │  • numbering_policy (summary)             │         │
│              │  → results/patterns.json                  │         │
│              └──────────────────┬────────────────────────┘         │
│                                 │                                  │
│              ┌──────────────────▼────────────────────┐             │
│              │  AGENT 3 — Regex Generator            │             │
│              │                                       │             │
│              │  INPUTS:                              │             │
│              │  • patterns.json  (from Agent 2)      │             │
│              │  • policies[]     (from Agent 1P)     │             │
│              │                                       │             │
│              │  PROCESS:                             │             │
│              │  1. For each format_rule +             │             │
│              │     range_restriction:                 │             │
│              │     prompt LLM → regex                 │             │
│              │  2. For contextual_keywords:           │             │
│              │     build proximity regex              │             │
│              │     programmatically                   │             │
│              │  3. re.compile() every regex           │             │
│              │  4. On failure: re-prompt with error   │             │
│              │                                       │             │
│              │  OUTPUTS:                             │             │
│              │  • patterns[] with regex strings       │             │
│              │  → results/regex_patterns.json         │             │
│              └──────────────────┬────────────────────┘             │
│                                 │                                  │
│              ┌──────────────────▼────────────────────┐             │
│              │  AGENT 4 — Validator                  │             │
│              │                                       │             │
│              │  INPUTS:                              │             │
│              │  • test.txt          (held-out data)  │             │
│              │  • regex_patterns.json (from Agent 3) │             │
│              │  • policies[]        (from Agent 1P)  │             │
│              │  • data_source flag                   │             │
│              │                                       │             │
│              │  PROCESS:                             │             │
│              │  NET mode: LLM generates 1000 test    │             │
│              │    cases (balanced valid/invalid)      │             │
│              │  LLM mode: 40% held-out test.txt      │             │
│              │  Per-pattern: precision, recall,       │             │
│              │    accuracy, F1                        │             │
│              │  Auto-remediation: if recall < 0.80   │             │
│              │    → feed mismatches back to Agent 3   │             │
│              │    (up to 2 rounds)                    │             │
│              │                                       │             │
│              │  OUTPUTS:                             │             │
│              │  → results/validation_report.json      │             │
│              │  → results/final_report.md             │             │
│              └───────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Agent 1P Internal Flow

```
research_policies(condition, config)
│
├──► PHASE 1: Official Policies
│    │
│    ├──► _find_policy_urls(condition, config)
│    │    │
│    │    │   ┌─────────────────────────────────────────────────────┐
│    │    │   │  Tavily Search                                      │
│    │    │   │  ─────────────                                      │
│    │    │   │  Queries (10):                                      │
│    │    │   │  • "{cond} format specification rules official"     │
│    │    │   │  • "{cond} numbering policy allocation government"  │
│    │    │   │  • "{cond} valid range structure check digit"       │
│    │    │   │  • "{cond} format wikipedia"                        │
│    │    │   │  • "{cond} validation rules structure format guide" │
│    │    │   │  • "{cond} format rules digits length example"      │
│    │    │   │  • "{cond} official format documentation"           │
│    │    │   │  • "{cond} numbering system structure ISO standard" │
│    │    │   │  • "{cond} regex pattern format definition"         │
│    │    │   │  • "{cond} identifier format allocation rules"      │
│    │    │   │                                                     │
│    │    │   │  → Up to 20 URLs with titles + snippets             │
│    │    │   └────────────────────────┬────────────────────────────┘
│    │    │                            │
│    │    │   ┌────────────────────────▼────────────────────────────┐
│    │    │   │  Rank + Quality Filter                              │
│    │    │   │  ─────────────────────────────────────              │
│    │    │   │                                                     │
│    │    │   │  1. SCORE (composite)                               │
│    │    │   │     Domain authority tier    (0–10)                 │
│    │    │   │     Path keyword relevance   (0–3)                 │
│    │    │   │                                                     │
│    │    │   │  2. FILTER                                          │
│    │    │   │     Block: social, video, e-commerce, forums        │
│    │    │   │     Cap: 2 URLs per domain                          │
│    │    │   │                                                     │
│    │    │   │  3. SELECT TOP 15                                   │
│    │    │   │     Sort descending by composite score               │
│    │    │   └────────────────────────┬────────────────────────────┘
│    │    │                            │
│    │    └───► returns 15 ranked URLs ◄┘
│    │
│    ├──► _scrape_and_extract(top_15_urls)
│    │    • Playwright+Camoufox scrape each page
│    │    • _extract_rule_blocks() → focused text
│    │    • Fallback: _html_to_text()
│    │
│    ├──► LLM Policy Extraction (_POLICY_PROMPT)
│    │    • Feed scraped text (capped 60KB)
│    │    • Returns: policies[], numbering_authority
│    │
├──► PHASE 2: Vendor DLP Patterns
│    │    (Tavily/DDG for vendor docs)
│    │
├──► PHASE 3: LLM Fallback
│    │    (if web yields nothing)
│    │
└──► RETURN
     {
       condition, policies[], numbering_authority,
       vendor_patterns[], vendor_keywords[], sources[]
     }
```

### Data Flow Between All Agents — Summary Table

```
┌─────────┬────────────────────┬──────────────────────────┬──────────────────────────────┐
│  Agent  │  Reads             │  Writes                  │  Passes to                   │
├─────────┼────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Agent 1 │ condition          │ data/train.txt           │ Agent 2 (train data)         │
│         │ config             │ data/test.txt            │ Agent 4 (test data)          │
│         │ db/db_index.json   │ data/scraped.xlsx        │ main.py (stats)              │
│         │ (web via scraper)  │                          │                              │
├─────────┼────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Agent   │ condition          │ (in-memory dict)         │ Agent 2 (policies,           │
│ 1P      │ config             │                          │   vendor_patterns,           │
│         │ (web via Tavily    │                          │   vendor_keywords)           │
│         │  + Camoufox)       │                          │ Agent 3 (policies)           │
│         │                    │                          │ Agent 4 (policies)           │
│         │                    │                          │ main.py (→ source_of_truth)  │
├─────────┼────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Agent 2 │ data/train.txt     │ results/patterns.json    │ Agent 3 (patterns.json)      │
│         │ policies (1P)      │                          │                              │
│         │ vendor_patterns    │                          │                              │
│         │ vendor_keywords    │                          │                              │
├─────────┼────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Agent 3 │ patterns.json      │ results/regex_patterns   │ Agent 4 (regex_patterns)     │
│         │ policies (1P)      │   .json                  │                              │
├─────────┼────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Agent 4 │ data/test.txt      │ results/validation_      │ (terminal — report to user)  │
│         │ regex_patterns.json│   report.json            │                              │
│         │ policies (1P)      │ results/final_report.md  │                              │
│         │ data_source flag   │                          │                              │
└─────────┴────────────────────┴──────────────────────────┴──────────────────────────────┘
```

### Test Conditions

| Condition | Notes |
|-----------|-------|
| US Medicaid Number | Government numbering, state-specific prefixes |
| India PAN | Alphanumeric with check digit, well-documented |
| Russian Military Identity Number | Regional formatting, sparse English docs |
| SWIFT/BIC Code | International standard, ISO 9362 |

---

## Config

### `.env`

```bash
# AWS — set your named profile for local dev; leave EMPTY on EC2/ECS
AWS_PROFILE=
AWS_REGION=us-east-1

# Tavily web-search API key (optional — falls back to DuckDuckGo)
TAVILY_API_KEY=
```

### `load_config()` in `utils/bedrock_client.py`

Reads `config.yaml` and merges environment variables for:
- `aws_profile` / `AWS_PROFILE`
- `region` / `AWS_REGION`
- `tavily_api_key` / `TAVILY_API_KEY`
- `proxies` / `PROXIES`

### Sequence Diagram

```
User            main.py         Agent 1P          Tavily          Camoufox
 │                │                │                │                │
 │ --condition    │                │                │                │
 │───────────────►│                │                │                │
 │                │ research_      │                │                │
 │                │ policies()     │                │                │
 │                │───────────────►│                │                │
 │                │                │                │                │
 │                │                │  10 queries    │                │
 │                │                │───────────────►│                │
 │                │                │  ≤20 URLs      │                │
 │                │                │◄───────────────│                │
 │                │                │                │                │
 │                │                │ ┌──────────────────────────────┐│
 │                │                │ │ Rank + Dedupe → top 15       ││
 │                │                │ └──────────────────────────────┘│
 │                │                │                │                │
 │                │                │  scrape top 15 │                │
 │                │                │───────────────────────────────►│
 │                │                │  HTML content  │                │
 │                │                │◄──────────────────────────────│
 │                │                │                │                │
 │                │                │ ┌──────────────────────────┐   │
 │                │                │ │ Extract rule blocks      │   │
 │                │                │ │ → LLM policy extraction  │   │
 │                │                │ └──────────────────────────┘   │
 │                │                │                │                │
 │                │  policies dict │                │                │
 │                │◄───────────────│                │                │
 │                │                │                │                │
```
