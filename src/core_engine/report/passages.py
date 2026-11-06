"""Bounded, question-directed windows of the original text, preserving offsets."""
from __future__ import annotations

import re

from core_engine.report.text import content_terms, normalize_text

# Retrieval vocabulary only. A translation/related keyword never proves a claim.
CONCEPTS = (
    ("硫化物", "sulfide", "sulphide"), ("氧化物", "oxide", "nasicon", "garnet"),
    ("聚合物", "polymer"), ("卤化物", "halide"), ("界面", "interface", "interfacial"),
    ("阻抗", "impedance", "resistance"), ("电导率", "conductivity"),
    ("制造", "manufacturing", "fabrication", "processing"), ("良率", "yield"),
    ("瓶颈", "限制", "挑战", "风险", "challenge", "limitation", "barrier"),
    ("枝晶", "dendrite", "filament"), ("压力", "pressure", "stack pressure"),
    ("循环寿命", "cycle life", "cycling", "capacity retention"),
    ("能量密度", "energy density", "specific energy"), ("倍率", "rate capability", "c-rate"),
    ("安全", "safety", "thermal runaway"), ("液态", "liquid", "gel"),
    ("测试条件", "实验条件", "test conditions", "temperature", "loading", "protocol"),
    ("硫化氢", "hydrogen sulfide", "h2s"), ("湿度", "moisture", "humidity"),
    ("烧结", "sintering"), ("量产", "商业化", "scale-up", "mass production", "commercialization"),
    ("导电剂", "导电添加剂", "conductive additive", "carbon-conductive", "carbon additive"),
    ("面容量", "面载量", "areal capacity", "areal loading", "cathode loading"),
    ("副反应", "side reaction", "interfacial reaction", "decomposition"),
)
STOP = {"what", "which", "how", "the", "and", "for", "with", "from", "this", "that", "are", "does",
        "固态", "电池", "solid", "state", "batteries", "battery", "哪些", "如何", "什么", "需要"}


def focus_terms(question: str) -> set[str]:
    normalized = normalize_text(question)
    terms = {term for term in content_terms(question, STOP) if len(term) > 1 and not term.isdigit()}
    for equivalents in CONCEPTS:
        if any(term in normalized for term in equivalents):
            terms.update(equivalents)
    return terms


def relevance(question: str, text: str) -> int:
    normalized = normalize_text(text)
    return sum(bool(re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", normalized))
               if term.isascii() else term in normalized for term in focus_terms(question))


def select_research_claims(rows, questions: list[str], limit: int):
    """Retain distinct requested concepts before repeated background observations."""
    if len(rows) <= limit or not questions:
        return rows[:limit]
    focus = normalize_text(" ".join(questions))
    concepts = [group for group in CONCEPTS if any(term in focus for term in group)]
    if not concepts:
        return rows[:limit]
    features = [{i for i, group in enumerate(concepts) if any(term in normalize_text(row.text) for term in group)}
                for row in rows]
    selected: list[int] = []
    covered: set[int] = set()
    while len(selected) < min(limit, len(rows)):
        index = max((i for i in range(len(rows)) if i not in selected), key=lambda i:
                    (len(features[i] - covered), len(features[i]), -i))
        selected.append(index)
        covered.update(features[index])
    return [rows[i] for i in selected]


def extraction_windows(text: str, questions: list[str], size: int, limit: int) -> list[tuple[int, int]]:
    """Select diverse original windows; include later methods/limitations, not just the preface."""
    references = re.search(r"(?im)^\s*(?:references|bibliography|参考文献)\s*$", text[max(1000, size // 2):])
    if references:
        text = text[:max(1000, size // 2) + references.start()]
    if len(text) <= size or limit <= 1:
        return [(0, min(len(text), size))]
    # Prefer paragraph boundaries, with bounded overlap for cross-paragraph context.
    windows = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            if boundary > start:
                end = boundary + 1
        windows.append((start, end))
        if end == len(text):
            break
        start = max(start + size // 2, end - min(300, size // 10))
    chosen = [windows[0]]
    scores = [[relevance(question, text[a:b]) for a, b in windows] for question in questions]
    covered = [scores[i][0] for i in range(len(scores))]
    while len(chosen) < min(limit, len(windows)):
        def score(window, covered=covered):
            index = windows.index(window)
            gains = [max(0, row[index] - covered[i]) for i, row in enumerate(scores)]
            return (sum(gains), sum(row[index] for row in scores), window[0])
        candidate = max((window for window in windows if window not in chosen), key=score)
        chosen.append(candidate)
        index = windows.index(candidate)
        covered = [max(value, scores[i][index]) for i, value in enumerate(covered)]
    return sorted(chosen)


def gap_window(text: str, deficit: str, size: int) -> tuple[int, int, int] | None:
    """Best bounded original window for a specific deficit, outside bibliography."""
    windows = extraction_windows(text, [deficit], size, 3)
    scored = [(relevance(deficit, text[a:b]), a, b) for a, b in windows]
    score, start, end = max(scored, default=(0, 0, 0))
    return (start, end, score) if score > 0 else None
