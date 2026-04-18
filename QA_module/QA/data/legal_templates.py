"""
LEDGAR label -> question templates.

Spec says ambiguous categories should be discarded. We keep a curated
set of LEDGAR labels whose question form is unambiguous; any label
not in this dict makes the example get dropped by
preprocess_ledgar.py.

Keys are matched case-insensitively against LEDGAR labels; the
canonical LEDGAR label set has ~100 entries but many are rarely
useful for span extraction.
"""

LEDGAR_TEMPLATES = {
    # --- Core commercial / contractual clauses ---
    "payments":                 "What are the payment obligations?",
    "termination":              "Under what conditions can this be terminated?",
    "termination for convenience": "Can this agreement be terminated for convenience?",
    "termination for cause":    "Under what causes can this be terminated?",
    "term":                     "What is the term of this agreement?",
    "renewal":                  "How does this agreement renew?",
    "notices":                  "How must notices be given under this agreement?",
    "indemnification":          "What indemnification obligations exist?",
    "indemnity":                "What indemnity obligations exist?",
    "confidentiality":          "What are the confidentiality requirements?",
    "non-disclosure":           "What are the non-disclosure obligations?",
    "governing law":            "What law governs this agreement?",
    "venue":                    "Where must disputes be filed?",
    "jurisdiction":             "Which jurisdiction applies to disputes?",
    "warranties":               "What warranties are provided?",
    "representations":          "What representations are made?",
    "assignment":               "Can this agreement be assigned?",
    "amendments":               "How can this agreement be amended?",
    "waiver":                   "How can provisions of this agreement be waived?",
    "entire agreement":         "What does the entire agreement clause state?",
    "severability":             "What happens if a provision is unenforceable?",
    "force majeure":            "What events qualify as force majeure?",
    "insurance":                "What insurance is required?",
    "liability":                "What are the liability limits?",
    "limitation of liability":  "What limits apply to liability?",
    "arbitration":              "How are disputes arbitrated?",
    "dispute resolution":       "How are disputes resolved?",
    "counterparts":             "How may this agreement be executed?",
    "headings":                 "What does the headings clause state?",
    "survival":                 "Which provisions survive termination?",
    "effectiveness":            "When does this agreement become effective?",
    "expenses":                 "Which party bears expenses?",
    "fees":                     "What fees are payable under this agreement?",
    "taxes":                    "How are taxes allocated?",
    "audit":                    "What audit rights are provided?",
    "compliance with laws":     "What compliance obligations exist?",
    "export control":           "What export control obligations exist?",
    "anti-corruption":          "What anti-corruption obligations exist?",
    "intellectual property":    "What intellectual property rights are addressed?",
    "licenses":                 "What licenses are granted?",
    "grant of rights":          "What rights are granted?",
    "publicity":                "What publicity restrictions apply?",
    "non-compete":              "What non-compete obligations apply?",
    "non-solicitation":         "What non-solicitation obligations apply?",
    "subcontracting":           "Can obligations be subcontracted?",
    "successors":               "Who are bound as successors?",
    "relationship of parties":  "What is the relationship between the parties?",
    "interpretation":           "How is this agreement interpreted?",
    "definitions":              "How are key terms defined?",
    "effective date":           "When does this agreement take effect?",
}


def template_for(label: str) -> str:
    """Return the question template for a LEDGAR label, or None."""
    if not label:
        return None
    key = label.strip().lower()
    return LEDGAR_TEMPLATES.get(key)
