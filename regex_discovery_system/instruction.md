# Agent Instructions — Regex Discovery System

This document lists the exact LLM instructions (prompts) passed to each agent
in the 4-agent pipeline, plus the supporting scraper LLM hints module.

---

## Pipeline Overview

```
                  ┌──────────────────────────┐
  Input ──────────┤  Agent 1: Data Collector  ├───────────┐
  (condition)     └──────────────────────────┘           │
                  ┌──────────────────────────┐           ▼
                  │ Agent 1P: Policy Researcher├──► Agent 2: Pattern Analyzer
                  └──────────────────────────┘           │
                                                          ▼
                                                 Agent 3: Regex Generator
                                                          │
                                                          ▼
                                                 Agent 4: Validator
```

Agents 1 and 1P run **in parallel**. Their outputs feed into Agent 2.

---

## Agent 1 — Data Collector

**File:** `agents/data_collector.py`
**Role:** Collects real-world example values via DB lookup → web scraping → LLM synthesis (fallback).

The LLM prompt is only used when DB and scraper both fail (LLM synthesis fallback).

### Prompt

```
You are a data-generation assistant.
Generate {batch_size} UNIQUE, REALISTIC examples of: {condition}.
Rules:
- One value per line, no numbering, no bullet points, no extra text.
- Values must resemble real-world data actually in use.
- Do not repeat values from previous batches: {previous_values_sample}
Output ONLY the values.
```

### Dynamic Placeholders

| Placeholder               | Source                                                     |
|---------------------------|------------------------------------------------------------|
| `{batch_size}`            | Number of values to generate per batch                     |
| `{condition}`             | User-supplied input string                                 |
| `{previous_values_sample}`| Last 50 generated values (or "N/A (first batch)")         |

---

## Agent 1P — Policy Researcher

**File:** `agents/policy_researcher.py`
**Role:** Searches the web for official format rules, policies, and vendor DLP patterns. Runs in parallel with Agent 1.

### Phase 1 — Policy Extraction Prompt

Used when government/official or **any reputable third-party** pages are scraped.

```
You are a numbering-policy expert.

CONDITION: {condition}

I have scraped the following web pages that may describe the format rules,
allocation policies, or structural constraints for the identifier above.
These pages may include government (.gov) sites, official regulatory body
websites, reputable third-party references (Wikipedia, open-data portals,
educational institutions), or well-known industry sites.

IMPORTANT: Accept and extract rules from ANY reputable, legitimate source —
not just government sites. If no government data is available but a credible
third-party site provides format rules, use that data confidently.

--- START OF SCRAPED TEXT ---
{texts}
--- END OF SCRAPED TEXT ---

TASKS — return a JSON object with these keys:

1. "policies": an array of concise, factual strings.  Each string is ONE
   structural rule or constraint (e.g. "First digit is always 9",
   "Length is exactly 5 digits", "Positions 1-2 encode the state").
   Include at least 5 rules if the text supports it.
   CRITICAL: Only include rules that are verifiable from the scraped text
   or that you know to be established by the official governing body.
   Cite the source where possible (government, Wikipedia, industry ref).

2. "numbering_authority": the name of the official government body or
   regulatory authority that governs this numbering system
   (e.g. "USPS", "CMS/HHS", "ITU", "India Post", "Ofcom").

Return ONLY valid JSON, no markdown fences.
```

### Phase 2 — Vendor/Competitor Extraction Prompt

Used when vendor DLP documentation pages are successfully scraped.

```
You are a DLP/data classification expert.

CONDITION: {condition}

I have scraped pages from major security and networking vendors that may
describe how they detect or classify this identifier type in their DLP,
CASB, or data classification products.

--- START OF SCRAPED TEXT ---
{texts}
--- END OF SCRAPED TEXT ---

TASKS — return a JSON object with these keys:

1. "vendor_patterns": an array of objects, each with:
   - "vendor": vendor name (e.g. "Microsoft Purview", "Netskope DLP")
   - "regex": the regex pattern they use (if found in the text), or null
   - "keywords": array of keywords/phrases they associate with this data type
   - "rules": array of format rules or validation logic they describe

2. "common_keywords": array of ALL unique keywords/phrases across vendors
   that are used to identify or label this data type. Include form field
   labels, DLP policy names, classification labels, etc.

Return ONLY valid JSON, no markdown fences.
```

### Phase 3 — LLM Fallback Prompt

Used when no useful web pages are found for either policies or vendors.

```
You are an expert on data classification and numbering systems.

CONDITION: {condition}

I could not find web pages with format rules for this identifier.
Using your training knowledge, provide:

1. "policies": array of concise structural rules/constraints for this
   identifier type. Include format, length, character types, ranges,
   check digits, and any allocation rules you know.

2. "numbering_authority": the official governing body.

3. "vendor_patterns": array of objects describing how major DLP vendors
   (Microsoft Purview, Netskope, Broadcom/Symantec, Zscaler, Skyhigh,
   Forcepoint, Palo Alto) classify this data type. For each vendor you
   know about, include:
   - "vendor": name
   - "regex": the regex they use (if known), or null
   - "keywords": keywords they associate with this data
   - "rules": validation rules they apply

4. "common_keywords": array of ALL keywords/phrases that any vendor or
   official source uses to label or detect this identifier. Be exhaustive.

Return ONLY valid JSON, no markdown fences.
```

### Dynamic Placeholders

| Placeholder    | Source                                              |
|----------------|-----------------------------------------------------|
| `{condition}`  | User-supplied input string                          |
| `{texts}`      | Combined scraped text from up to 3 pages (max 30K chars) |

---

## Agent 2 — Pattern Analyzer

**File:** `agents/pattern_analyzer.py`
**Role:** Analyzes collected data samples + policies to discover structural patterns, format rules, and contextual keywords.

### Main Prompt

```
You are a pattern-analysis expert.

CONDITION: {condition}

STATISTICAL SUMMARY:
{stats_json}

SAMPLE VALUES (200 of {total}):
{sample_values}
{policies}
TASKS — return a JSON object with these keys:

1. "format_rules": array of objects, each with:
   - "rule_id": short slug
   - "description": plain-English description of the structural rule
   - "applies_to": "all" | "subset"
   - "example_matches": [3+ examples from the sample]

2. "range_restrictions": array of objects, each with:
   - "position": which digit(s) or segment
   - "allowed_values": explicit set or range description
   - "source": cite the real-world policy or numbering authority if known

3. "contextual_keywords": array of strings — words/phrases that commonly
   appear near this identifier in documents (case-insensitive).
   IMPORTANT: Include as MANY relevant keywords as possible — aim for 8-15.
   Cast a wide net but do NOT include generic words that would cause false
   positives (e.g. avoid "number", "code", "ID" alone — too ambiguous).
   Every keyword must be specific enough that its presence near a value
   is a strong signal that the value is this identifier type.

   CRITICAL — MULTILINGUAL KEYWORDS: You MUST include keywords in the
   LOCAL LANGUAGE of the country this identifier belongs to. For example:
     - Latvia → include Latvian terms: "adrese", "pasta indekss",
       "dzīvesvieta", "pilsēta", "pasta indeksa sektors"
     - Czech Republic → include Czech: "poštovní směrovací číslo", "adresa"
     - Mexico → include Spanish: "dirección", "código postal", "ciudad"
     - Israel → include Hebrew terms
     - Sweden → include Swedish: "personnummer", "postnummer"
   Use actual Unicode characters (ā, ē, ī, č, š, ñ, ö, etc.) — not ASCII.

   Think broadly across these categories:
     - Form field labels (e.g. "ZIP code", "postal code", "mailing address")
     - Column headers in spreadsheets/databases
     - Surrounding text in official documents, forms, invoices
     - Abbreviations and acronyms used in the industry
     - LOCAL LANGUAGE terms used in official forms of that country
     - Related field names in software systems and APIs
     - Labels used by DLP/security vendors (Microsoft, Netskope, Broadcom,
       Zscaler, Skyhigh) for this same identifier type

4. "keyword_proximity": integer — how many terms away a keyword can be
   from the identifier and still indicate a match (default 10).

5. "numbering_policy": free-text summary of the real-world authority,
   allocation rules, check-digit algorithms, or geographic mapping that
   governs this identifier.

Return ONLY valid JSON, no markdown fences.
```

### Retry Prompt (invalid JSON)

```
Your previous response was not valid JSON. Please fix it.
Previous response:
{response}

Return ONLY valid JSON, no markdown fences.
```

### Dynamic Placeholders

| Placeholder        | Source                                                     |
|--------------------|------------------------------------------------------------|
| `{condition}`      | User-supplied input string                                 |
| `{stats_json}`     | Statistical summary (lengths, char distribution, etc.)     |
| `{total}`          | Total number of sample values                              |
| `{sample_values}`  | Up to 200 sampled values, one per line                     |
| `{policies}`       | Injected section from Agent 1P (rules + vendor patterns + vendor keywords) |

---

## Agent 3 — Regex Generator

**File:** `agents/regex_generator.py`
**Role:** Generates PCRE2/JavaScript-compatible regex patterns from the discovered format rules.

### Key Behaviors (enforced in code, not just prompts)

| Rule | Implementation |
|------|---------------|
| **No `^` or `$` anchors** | `_strip_anchors_add_boundaries()` strips them post-LLM |
| **`\b` word boundaries** | Added automatically to start and end of every value regex |
| **No `XXX` placeholders** | Replaced with `[A-Z0-9]{3}` in post-processing |
| **No proximity suffix on keyword regex** | `build_keyword_regex()` returns `(?i)(?:kw1\|kw2\|...)` only |
| **Spaces not escaped in keywords** | Custom `_escape_keyword()` preserves spaces and Unicode |
| **Diacritic tolerance in keywords** | `_escape_keyword()` expands every accented char into `[base\|accented]` class (e.g. `ú` → `[uú]`) so plain-ASCII typing also matches |
| **Language-exclusivity warning** | `build_keyword_regex()` calls `_is_language_exclusive()` on every keyword and logs a warning for pure-ASCII single-token keywords (e.g. `"KT"`, `"DNI"`) that could fire in unrelated foreign-language documents |
| **Combined value regex** | All individual value regexes merged into one unified pattern |
| **`ensure_ascii=False`** | JSON output preserves Unicode characters (ā, ē, ñ, etc.) |

### Main Prompt (per rule)

```
You are a regex engineering expert.

Write a single PCRE2/JavaScript-compatible regex that matches values satisfying this rule:

RULE: {rule_description}
RANGE CONSTRAINTS: {range_restrictions}
EXAMPLE MATCHES: {example_matches}

Requirements:
- Use raw string notation.
- Do NOT use ^ or $ anchors. The regex will be used for search within text.
- Use \b (word boundary) at the start and end of the pattern to prevent
  partial matches within longer strings.
- Avoid catastrophic backtracking.
- Prefer character classes and quantifiers over alternation where possible.
- NEVER include literal placeholder strings like "XXX" or "NNN" in the regex.
  If an optional part can be any characters, use the appropriate character
  class (e.g. [A-Z0-9]{3} not XXX).
- If the rule involves check digits, encode the structural pattern only
  (check-digit validation is done in code, not regex).

Return ONLY the regex string, no explanation.
```

### Refinement Prompt (on compile error)

```
The regex you provided is invalid.

ORIGINAL REGEX: {regex}
ERROR: {error}

Fix the regex so it is valid in both PCRE2 and JavaScript (ES2018+).
Do NOT use Python-specific syntax (e.g. (?P<name>...), (?P=name)).
Return ONLY the corrected regex string, no explanation.
```

### Post-Processing Pipeline

Every value regex goes through `_strip_anchors_add_boundaries()`:
1. Strip leading `^` if present
2. Strip trailing `$` if present
3. Replace any `XXX` with `[A-Z0-9]{3}`
4. Add `\b` at start if not already present
5. Add `\b` at end if not already present

### Combined Value Regex

After all individual value patterns are generated, a unified combined regex is
built by joining them: `\b(?:(?:pattern1)|(?:pattern2)|...)\\b`

This combined regex is added with `rule_id: "combined-value"` and is the
primary pattern used for comparison with user-provided regexes.

### Keyword Regex Format

Keywords are joined into: `(?i)(?:keyword1|keyword2|keyword with spaces|local_lang_term)`

**What it does NOT include:**
- No escaped spaces (`\ ` → just use ` `)
- No proximity suffix (`(?:\W+\w+){0,10}\W+`)

**Diacritic / special-character tolerance (implemented in `_escape_keyword`):**

Users often type keywords without diacritics because the accented character is
not on their keyboard.  Every accented character in a keyword is automatically
expanded into a `[base_char|accented_char]` character class so that both the
proper Unicode form and the plain-ASCII fallback match.

| Unicode keyword character | Regex character class produced |
|---------------------------|-------------------------------|
| `ú` (e.g. "N**ú**mero")   | `[uú]`                        |
| `é` (e.g. "biom**é**trico") | `[eé]`                      |
| `ó` (e.g. "Identificaci**ó**n") | `[oó]`                 |
| `á` (e.g. "Tel**é**fono")  | `[aá]`                        |
| `ñ` (e.g. "espa**ñ**ol")  | `[nñ]`                        |
| `ā`, `ē`, `ī` (Baltic)    | `[aā]`, `[eē]`, `[iī]`       |
| `č`, `š`, `ž` (Slavic)    | `[cč]`, `[sš]`, `[zž]`       |

**Example — Argentina passport keyword before/after:**

| Input keyword (from Agent 2) | Produced regex fragment |
|------------------------------|-------------------------|
| `Número de pasaporte`        | `N[uú]mero de pasaporte` |
| `Identificación nacional`    | `Identificaci[oó]n nacional` |
| `Dirección Nacional de Migraciones` | `Direcci[oó]n Nacional de Migraciones` |
| `Pasaporte electrónico`      | `Pasaporte electr[oó]nico` |
| `Pasaporte biométrico`       | `Pasaporte biom[eé]trico` |

Full keyword regex (excerpt):
```
(?i)(?:Pasaporte argentino|N[uú]mero de pasaporte|DNI|Documento Nacional de Identidad|Direcci[oó]n Nacional de Migraciones|Ministerio del Interior|Visa argentina|Documento de viaje|Identificaci[oó]n nacional|Registro Nacional de las Personas|RENAPER|Pasaporte electr[oó]nico|Pasaporte biom[eé]trico)
```

**Coverage:** All characters in `_DIACRITIC_MAP` (a–u variants, ñ, ç, č, š, ž,
ř, ł, ķ, ģ, ğ, ď, ț and their uppercase equivalents) are handled.

### Language-Exclusivity (`_is_language_exclusive`)

A keyword is **language-exclusive** if it can physically only appear in a
document written in the target language.  If a keyword passes the
exclusivity test, it prevents false-positive keyword matches in unrelated
foreign-language documents (e.g. English text that happens to contain "KT").

**The two tests (either is sufficient):**

| Test | Passes | Fails |
|---|---|---|
| **Contains ≥1 non-ASCII char** | `"Número"` (ú), `"Identificación"` (ó), `"dzīvesvieta"` (ī) | `"KT"`, `"DNI"`, `"RENAPER"` |
| **Multi-word phrase (≥2 tokens)** | `"pasaporte argentino"`, `"pasta indekss"`, `"Argentina passport"` | `"KT"` (1 token), `"DNI"` (1 token) |

**What happens when a keyword fails both tests (pure-ASCII single token):**
- It is **kept** in the regex (not silently dropped).
- A `WARNING` is logged by Agent 3 so the operator can review.

**Non-Latin scripts are automatically language-exclusive — no warning:**

| Script | Example keyword | `isascii()` | Verdict |
|---|---|---|---|
| Cyrillic | `"КТ"` (CT scan), `"МРТ"` (MRI) | `False` | ✅ exclusive — included, no warning |
| Cyrillic | `"больница"` (hospital) | `False` | ✅ exclusive — included, no warning |
| Arabic | `"رقم"` (number) | `False` | ✅ exclusive — included, no warning |
| Hebrew | `"מס׳"` | `False` | ✅ exclusive — included, no warning |
| Latin ASCII | `"KT"` | `True` | ⚠️ warned — same bytes appear in any language |
| Latin ASCII | `"MRT"` | `True` | ⚠️ warned — ambiguous in English context |

Python's `re` engine treats Unicode code points exactly: Cyrillic `К` (U+041A) and
Latin `K` (U+004B) are **different characters**. `(?i)(?:КТ)` matches `КТ`/`кт`
but will **never** match Latin `KT`, even with `IGNORECASE` enabled.  
Agent 2 is instructed to generate all non-Latin-script abbreviations freely.

**Why the diacritic form IS language-exclusive:**

`"Número de pasaporte"` → `"N[uú]mero de pasaporte"`

The expanded class `[uú]` matches both `u` (plain ASCII) and `ú` (accented).
Even the plain-ASCII form `"Numero de pasaporte"` is a **Spanish phrase** that
cannot appear verbatim in an English document, so the keyword is still
language-exclusive even after diacritic expansion.

**Example — Argentina passport keyword list quality (Latin script):**

| Keyword | Language-exclusive? | Reason |
|---|---|---|
| `"Pasaporte argentino"` | ✅ | multi-word Spanish phrase |
| `"Número de pasaporte"` | ✅ | contains ú (non-ASCII) |
| `"Dirección Nacional de Migraciones"` | ✅ | contains ó + multi-word Spanish |
| `"RENAPER"` | ✅ (marginal) | acronym unique to Argentina, no English meaning |
| `"DNI"` | ⚠️ warned | pure ASCII, 3-char — used in Spanish but no diacritic |
| `"KT"` | ⚠️ warned | pure ASCII, 2-char — appears in any language |

**Example — Russian medical record keyword list quality (Cyrillic script):**

| Keyword | Language-exclusive? | Reason |
|---|---|---|
| `"больница"` | ✅ | Cyrillic — isascii() False, physically cannot appear in English |
| `"медицинская справка"` | ✅ | Cyrillic multi-word phrase |
| `"КТ"` | ✅ | Cyrillic К+Т — `(?i)(?:КТ)` will never match Latin `KT` |
| `"МРТ"` | ✅ | Cyrillic М+Р+Т — `(?i)(?:МРТ)` will never match Latin `MRT` |
| `"ПТСР"` | ✅ | Cyrillic abbreviation for PTSD — not ASCII |

**Example output (Latvian postal code):**
```
(?i)(?:adrese|dzīvesvieta|pils[eē]ta|pasta indekss|pasta rajons|pasta indeksa teritorija|pasta indeksa vien[iī]ba|pasta indeksa sektors|Latvian postal code|LV postcode)
```

### Dynamic Placeholders

| Placeholder             | Source                                         |
|-------------------------|------------------------------------------------|
| `{rule_description}`    | `description` from each format rule            |
| `{range_restrictions}`  | JSON string of range restrictions              |
| `{example_matches}`     | Comma-separated example values                 |
| `{regex}`               | The regex that failed to compile               |
| `{error}`               | Python `re.compile()` error message            |

---

## Agent 4 — Validator

**File:** `agents/validator.py`
**Role:** Validates generated regex patterns against test data. Uses LLM to generate test cases when data is from DB/scraper.

### Test Case Generation Prompt

Used when data source is DB or scraper (not LLM-synthesized).

```
You are a test-case generation expert.

CONDITION: {condition}

KNOWN VALID EXAMPLES (from real data):
{sample_values}
{policy_section}
Generate exactly {count} test values — a mix of:
  - ~50% values that SHOULD match (realistic valid examples, including edge cases)
  - ~50% values that SHOULD NOT match (plausible but invalid — wrong range,
    wrong length, wrong format, boundary violations)

For each value, output ONE line in this exact format:
  VALID|<value>
  INVALID|<value>

No extra text, no numbering, no blank lines. Output ONLY the test lines.
```

### Remediation Prompt (auto-fix failing regex)

Used when a regex has recall < 80% — up to 2 remediation rounds.

```
You are a regex engineering expert.

The following regex was intended to match values for: {condition}

REGEX: {regex}
RULE: {description}

It FAILED to match these known-valid values:
{mismatches}

Fix the regex so it matches ALL the values above while still being precise.
Return ONLY the corrected regex string, no explanation.
```

### Dynamic Placeholders

| Placeholder        | Source                                                      |
|--------------------|-------------------------------------------------------------|
| `{condition}`      | User-supplied input string                                  |
| `{sample_values}`  | Up to 200 real values for context                           |
| `{policy_section}` | Format rules from policies (if available)                   |
| `{count}`          | Batch size for test generation (250 per batch, up to 1000)  |
| `{regex}`          | The failing regex pattern                                   |
| `{description}`    | The rule description this regex implements                  |
| `{mismatches}`     | Up to 30 false-negative values                              |

### Validation Modes

| Data Source         | Validation Approach                                          |
|---------------------|--------------------------------------------------------------|
| DB / Scraper        | LLM generates 1000 test cases (50% valid / 50% invalid)     |
| LLM Synthesis       | Uses the 40% held-out `test.txt` from Agent 1               |

---

## Scraper — LLM Hints

**File:** `tools/scraper/llm_hints.py`
**Role:** Asks the LLM to suggest search queries, country info, and a value extraction regex for the scraper tool. Called by Agent 1 before web scraping.

### Main Prompt

```
You are helping a web scraper find data for: "{condition}"

Return a JSON object with exactly these keys:
{
  "country": "the country this data belongs to",
  "gov_domains": ["gov.xx", "official-authority.xx"],
  "search_queries": ["query1", "query2", "query3", "query4", "query5"],
  "value_regex": "regex_pattern_here",
  "description": "brief description of the data format"
}

Rules for country and gov_domains:
- Identify WHICH COUNTRY this data type belongs to
- List 2-3 government or official authority domains for THAT country
- These domains help prioritize official results, but DO NOT reject
  legitimate third-party sites — accept any reputable source with data

Rules for search_queries (5 queries, ASCII only):
- Queries should find web pages that LIST many real examples of this data
- Include terms like "list", "all", "complete", "database", "directory"
- Prioritize government/official sites first but DO NOT limit to them:
  * 2 queries SHOULD use "site:<domain>" targeting the gov_domains
  * 2-3 queries MUST target third-party data aggregators, open-data portals,
    Wikipedia, or well-known reference sites
- Use ASCII characters only (no accented characters, no Unicode)

Rules for value_regex:
- A Python regex that matches a SINGLE value of this data type WITHIN text
- NEVER use anchors (^ or $) — the regex is used with re.findall on HTML
- Use \b word boundaries instead
- Must be specific enough to avoid false positives from random text

Return ONLY valid JSON, no markdown fences, no explanation.
```

### Post-Processing Rules

- Strips `^` and `$` anchors from `value_regex` if the LLM includes them
- Sets `value_regex` to `None` if it fails `re.compile()`
- Falls back to built-in `_FALLBACK_PATTERNS` in the runner if `value_regex` is `None`

---

## Comparison — LLM Analysis

**File:** `web/app.py` (function `_llm_compare`)
**Role:** Compares user-provided regex patterns against system-generated ones using Claude Sonnet analysis.

### Prompt

```
IDENTIFIER TYPE: {condition}
TRUTH-SET SIZE : {total} known-valid values

─── KNOWN FORMAT RULES ───────────────────────────────────────────────────────
{rules_block}

─── RANGE / POSITION RESTRICTIONS ───────────────────────────────────────────
{ranges_block}

─── REGEXES UNDER COMPARISON ─────────────────────────────────────────────────
{regex_block}

System keyword regex : `{sys_kw}`
User keyword regex   : `{user_kw}`

─── YOUR TASK ────────────────────────────────────────────────────────────────
Produce a concise Markdown analysis with EXACTLY the following sections.
Keep the TOTAL response under 350 words. Be precise, no filler sentences.

---

## Condition Comparison

For EVERY known format rule and range restriction above (plus any you can
infer from the regexes), one row per condition:

| Condition | {col_header} | Winner |
|-----------|{col_sep}|--------|
| <condition> | ✅ / ❌ / ⚠️ | ... | **Label** or Tie |

Symbols: ✅ fully enforced · ❌ not enforced · ⚠️ partially enforced

Cover at minimum (where applicable):
length · character type · segment structure · per-position restrictions ·
optional parts · word boundaries · check-digit encoding

---

## Scores

| Metric | {col_header} |
|--------|{col_sep}|
| fullmatch % | ... |
| search % | ... |
| Conditions ✅ | X/{n_rules} | ... |

---

## Key Differences

3 bullet points maximum. Cite actual character classes / quantifiers.
For keyword regexes, note language coverage and missing/extra terms in one bullet.

---

## Winner

One sentence: which regex wins and the single most important reason why.
If neither is ideal, give an improved regex in a fenced code block (no explanation needed).
```

**System message:** `"You are a regex accuracy analyst."`

**Design notes:**
- Total LLM response is capped at ~350 words — the table carries the detail.
- The Condition Comparison table is the centrepiece: every known format rule
  and range restriction gets its own row with ✅ / ❌ / ⚠️ for each regex.
- Key Differences is a max-3-bullet digest; keyword comparison goes here too.
- Winner is a single sentence (+ optional improved regex in a code block).

---

## Summary of Regex Formatting Rules

| Rule | Before | After |
|------|--------|-------|
| Strip `^` anchor | `^LV-\d{4}$` | `LV-\d{4}` |
| Strip `$` anchor | (same) | (same) |
| Add `\b` boundaries | `LV-\d{4}` | `\bLV-\d{4}\b` |
| No `XXX` literals | `[A-Z]{4}US[A-Z0-9]{2}XXX` | `[A-Z]{4}US[A-Z0-9]{2}[A-Z0-9]{3}` |
| Keyword: no escaped spaces | `Latvian\ postal\ code` | `Latvian postal code` |
| Keyword: no proximity suffix | `(?i)(?:kw)(?:\W+\w+){0,10}\W+` | `(?i)(?:kw)` |
| Keyword: local language | English only | English + Latvian/Czech/Hebrew/etc. |
| Keyword: diacritic tolerance | `Número` | `N[uú]mero` (matches "Numero" too) |
| Keyword: lang-exclusivity warning | `"KT"` pure-ASCII single token | kept + WARNING logged (not dropped) |
| Combined value regex | Individual patterns only | All merged into `\b(?:p1\|p2\|...)\b` |
| JSON output | ASCII-escaped Unicode | `ensure_ascii=False` preserves ā, ē, ñ |

---

## Summary of LLM Calls Per Pipeline Run

| Stage              | Agent    | LLM Call Purpose                              | When Used                          |
|--------------------|----------|-----------------------------------------------|------------------------------------|
| Data collection    | 1        | Synthesize example values                     | Only if DB + scraper fail          |
| Scraper hints      | (tool)   | Generate search queries + value regex         | Before web scraping                |
| Policy research    | 1P       | Extract policies from scraped text            | If policy pages scraped            |
| Vendor research    | 1P       | Extract DLP vendor patterns from scraped text | If vendor pages scraped            |
| Fallback knowledge | 1P       | Generate policies from training data          | If no pages scraped at all         |
| Pattern analysis   | 2        | Discover format rules + keywords              | Always                             |
| Regex generation   | 3        | Write regex per format rule                   | Always (1 call per rule)           |
| Regex refinement   | 3        | Fix regex compile errors                      | Only if regex fails `re.compile()` |
| Test generation    | 4        | Generate valid/invalid test cases             | If data from DB/scraper            |
| Regex remediation  | 4        | Fix regex with low recall                     | If recall < 80% (up to 2 rounds)  |
| Comparison         | (UI)     | Analyze system vs user regexes                | User-triggered from UI             |
