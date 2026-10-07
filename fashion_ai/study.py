"""Stable, blinded candidate sets for human outfit ratings."""

from __future__ import annotations

import hashlib
import json
import random
from itertools import combinations

from fashion_ai.recommender import _color_score, rank_outfits


STUDY_VERSION = "outfit-rules-v4-pinterest-occasion"


def outfit_key(items: tuple[dict, ...]) -> str:
    return ",".join(str(item["id"]) for item in items)


def study_candidates(items: list[dict], occasion: str, trend_tags: set[str], user_id: int) -> tuple[str, list[dict]]:
    """Compare rules, a colour-only baseline, and random valid outfits."""
    ranked = rank_outfits(items, occasion, trend_tags, limit=None)
    if not ranked:
        return "", []
    keys = [outfit_key(result["items"]) for result in ranked]
    selected: list[dict] = []
    selected_keys: set[str] = set()

    def add(index: int, method: str) -> None:
        if keys[index] not in selected_keys:
            selected.append({
                "items": ranked[index]["items"], "outfit_key": keys[index],
                "model_rank": index + 1, "model_score": ranked[index]["score"],
                "method": method, "candidate_pool_size": len(ranked),
            })
            selected_keys.add(keys[index])

    for index in range(min(2, len(ranked))):
        add(index, "rules")
    colour_order = sorted(
        range(len(ranked)),
        key=lambda index: sum(
            _color_score(first["color"], second["color"])
            for first, second in combinations(ranked[index]["items"], 2)
        ),
        reverse=True,
    )
    for index in colour_order:
        if sum(choice["method"] == "colour_baseline" for choice in selected) == 2:
            break
        add(index, "colour_baseline")
    remaining = [index for index in range(len(ranked)) if keys[index] not in selected_keys]
    random.Random(f"{STUDY_VERSION}:{user_id}:{occasion}").shuffle(remaining)
    for index in remaining[:2]:
        add(index, "random")
    signature = json.dumps({
        "version": STUDY_VERSION, "user_id": user_id, "occasion": occasion,
        "choices": [(choice["outfit_key"], choice["method"], choice["model_rank"]) for choice in selected],
    }, sort_keys=True)
    study_id = hashlib.sha256(signature.encode()).hexdigest()[:20]
    random.Random(study_id).shuffle(selected)
    for display_position, choice in enumerate(selected, 1):
        choice["display_position"] = display_position
    return study_id, selected
