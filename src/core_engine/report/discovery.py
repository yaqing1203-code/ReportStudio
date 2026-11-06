"""Query-only discovery vocabulary; related topics never replace the user's subject."""
from __future__ import annotations

import re

from core_engine.report.text import normalize_text


def equivalent_terms(topic: str) -> list[str]:
    normalized = normalize_text(topic)
    semi = any(term in normalized for term in ("半固态", "准固态", "semi-solid", "quasi-solid"))
    all_terms = ["all-solid-state batteries", "all-solid-state battery", "all solid state batteries", "all solid state battery", "全固态电池"]
    generic_terms = ["solid-state batteries", "solid-state battery", "solid state batteries", "solid state battery", "固态电池"]
    # These are members of the broad battery topic. They do not permit a lithium
    # metal result to establish another chemistry's performance or mass production.
    chemistries = ("lithium", "lithium-ion", "lithium metal", "lithium-metal")
    specific = [term for term in chemistries[1:] if term in normalized]
    if not specific and re.search(r"\blithium\b", normalized):
        specific = ["lithium"]
    if "锂金属" in normalized:
        specific = ["lithium metal", "lithium-metal"]
    elif "锂离子" in normalized:
        specific = ["lithium-ion"]
    # Explicit chemistry questions keep only that chemistry's aliases.
    narrow_all, narrow_generic = [], []
    for chemistry in chemistries:
        for ending in ("batteries", "battery"):
            all_terms.append(f"all-solid-state {chemistry} {ending}")
            generic_terms.append(f"solid-state {chemistry} {ending}")
            if chemistry in specific:
                narrow_all.append(all_terms[-1])
                narrow_generic.append(generic_terms[-1])
    all_solid = (any(term in normalized for term in all_terms)
                 or "全固态" in normalized and "电池" in normalized)
    semi_terms = ["semi-solid-state batteries", "semi-solid-state battery", "quasi-solid-state batteries", "半固态电池"]
    if semi and all_solid:
        return (narrow_all or all_terms) + semi_terms
    if semi:
        return semi_terms
    if all_solid:
        return narrow_all or all_terms
    if (any(term in normalized for term in generic_terms)
            or specific and ("固态" in normalized or "solid-state" in normalized)):
        return narrow_generic or generic_terms
    return []


def matches_subject(topic: str, title: str, text: str, *, allow_lead: bool = False) -> bool:
    """Known subjects need their actual identity, beyond shared generic vocabulary."""
    aliases = [normalize_text(term) for term in equivalent_terms(topic)]
    if not aliases:
        return True  # Unknown subjects still use the ordinary relevance scorer.
    if any(alias in normalize_text(title) for alias in aliases):
        return True
    if allow_lead and any(alias in normalize_text(text[:400]) for alias in aliases):
        return True
    from core_engine.report.text import split_sentences

    return sum(any(alias in normalize_text(sentence) for alias in aliases)
               for sentence in split_sentences(text[:20000])) >= 2


def discovery_query(question: str, deficit: str, attempt: int, proposed_terms=()) -> tuple[str, str]:
    """Progress through distinct retrieval strategies instead of repeating a long prompt."""
    equivalents = equivalent_terms(question)
    scientific = bool(equivalents) or bool(re.search(r"electrolyte|conductivity|材料|电解质|论文|机理|催化", question, re.IGNORECASE))
    # Never send the entire compound question to every source. Equivalents are
    # curated; proposed translations are only search leads, not verified aliases.
    terms = equivalents[:1] or list(proposed_terms[:1]) or [question[:280]]
    focus = terms[0]
    from core_engine.report.passages import CONCEPTS

    # Keep each question's actual requirements. Previously every scientific gap
    # collapsed to the same "interfaces electrolyte" query and deduped away.
    context = normalize_text(question + " " + deficit)
    specific = [next((t for t in group if t.isascii()), group[0]) for group in CONCEPTS
                if any(t in context for t in group)]
    detail = " ".join(dict.fromkeys(specific))[:180] or deficit[:100]
    if scientific:
        strategies = [
            ("scholarly", f"{focus} {detail} research paper evidence"),
            ("primary", f'"{focus}" {detail} technical report experimental conditions'),
            ("targeted", f"{question[:250]} {deficit} 原始数据 实验条件"),
            ("counterevidence", f"{focus} limitations reproducibility correction retraction"),
        ]
    else:
        strategies = [
            ("targeted", f"{question[:280]} {deficit} original source 原始数据"),
            ("primary", f"{focus} {deficit} annual report technical specification"),
            ("scholarly", f"{focus} {deficit} research paper evidence"),
            ("counterevidence", f"{focus} {deficit} limitations contrary evidence"),
        ]
    return strategies[attempt % len(strategies)]
