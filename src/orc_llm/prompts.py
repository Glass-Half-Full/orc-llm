from __future__ import annotations

from collections.abc import Sequence


def policy_prompt(text: str, rules: Sequence[dict[str, str]]) -> str:
    rule_lines = "\n".join(
        f"- {rule.get('id', 'rule')}: {rule.get('description', '')}" for rule in rules
    )
    return f"""Assess the text against the policy rules.

Return exactly one of:
- meets_policy
- violates_policy
- needs_review

Policy rules:
{rule_lines}

Text:
{text}

Decision:"""


def normalize_verdict(text: str) -> str:
    lowered = text.strip().lower()
    if "violates_policy" in lowered or "violates policy" in lowered:
        return "violates_policy"
    if "meets_policy" in lowered or "meets policy" in lowered:
        return "meets_policy"
    return "needs_review"

