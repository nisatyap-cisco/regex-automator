Proposed Solution
Guiding Principle

Policy-backed rules are hard constraints.
Statistical patterns are soft evidence unless strongly validated.

1. Evidence Hierarchy
Evidence Type	Trust Level	Usage
Official policy / spec	High	Hard constraints
Vendor regex / DLP	High	Hard constraints
Large-scale data	Medium	Conditional rules
Small samples	Low	Observations only
2. Minimum Sample Size Guard
Current (unsafe)
absent = sorted(universe - observed)
if absent:
    position_constraints.append(...)
Fixed (baseline)
MIN_SAMPLES_FOR_CONSTRAINT = 200

if absent and len(chars_at_pos) >= MIN_SAMPLES_FOR_CONSTRAINT:
    position_constraints.append(...)
3. Add Confidence Gating

Even with 200 samples, absence ≠ invalidity.

Improved logic
if (
    absent
    and len(chars_at_pos) >= 200
    and is_high_confidence(position, absent, chars_at_pos)
):
    position_constraints.append(...)
else:
    statistical_observations.append(...)
4. Separate Hard vs Soft Constraints
Current (problematic)
{
  "range_restrictions": [...]
}
Proposed
{
  "format_rules": [...],
  "range_restrictions": [...],          // policy-backed only
  "statistical_observations": [...]     // sample-derived
}
5. Prompt Fix (Critical)
Current

"You MUST capture every entry in position_constraints..."

Replace with

Only convert position_constraints into hard range_restrictions if they are policy-backed or explicitly marked high-confidence. Otherwise, treat them as statistical observations.

learn: keep the patterns seen from sample data to learn "present" patterns that is seen across all- eg. length, starting with "LV-" cases


6. Policy-First Precedence Rule

If policy and data conflict:

✅ Keep policy rule
❌ Do NOT narrow using sample data
Example
Source	Rule
Policy	First digit ≠ 0
Sample	First digit ∈ [2-9]
Output
Keep: ≠ 0
Ignore: [2-9]
7. Rarity Safeguard

Do not exclude a character solely because it is unseen.

Only exclude when:

supported by policy
or extremely strong statistical evidence
8. Domain Plausibility Check

Before enforcing a rule:

Ask:

Is this restriction known for this identifier?
Would a domain expert expect this?
Examples
Identifier Type	Likely Rules
Bank account numbers	Length, prefix
Passport numbers	Prefix patterns
Random IDs	Usually none

If not plausible → discard

9. Recommended Enforcement Policy
Hard Constraints

Apply only if:

policy-backed
vendor-backed
officially documented
Soft Constraints

Keep as observations if:

sample size ≥ 200
statistically strong
domain plausible
not contradicting policy
Ignore

Discard if:

derived from absence in small/moderate samples
unsupported by documentation
likely to cause false negatives

11. Recommended Full Fix
MIN_SAMPLES_FOR_CONSTRAINT = 200

if (
    absent
    and len(chars_at_pos) >= MIN_SAMPLES_FOR_CONSTRAINT
    and is_policy_backed_or_high_confidence(...)
):
    position_constraints.append(...)   # hard constraint
else:
    statistical_observations.append(...)  # soft observation
12. Key Design Principles
Search broadly, enforce narrowly
Prefer documentation over data
Absence ≠ restriction
Statistical patterns require validation
Never let weak signals become hard constraints
Final Recommendation

Retain only policy-backed positional restrictions as hard regex constraints.
Statistical restrictions should require ≥200 samples and remain soft unless independently corroborated.