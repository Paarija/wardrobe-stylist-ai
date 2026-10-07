"""Explainable outfit and wardrobe-gap recommendations."""

from __future__ import annotations

from itertools import combinations, product
from urllib.parse import quote


NEUTRALS = {"black", "white", "grey", "cream", "beige", "brown", "navy"}
COLORS = (
    "black", "white", "grey", "cream", "beige", "brown", "navy", "blue",
    "teal", "green", "red", "burgundy", "pink", "purple", "yellow", "orange",
)
STYLES = ("casual", "smart", "sporty", "formal")
OCCASIONS = ("everyday", "college", "work", "party")
OCCASION_LABELS = {"everyday": "everyday", "college": "college", "work": "office", "party": "party"}
OCCASION_PROFILES = {
    "everyday": {
        "colors": {"blue": 1.6, "navy": 1.5, "beige": 1.4, "brown": 1.3, "white": 1.2, "teal": 1.1},
        "styles": {"casual", "sporty"}, "separates_bonus": 0.5, "dress_bonus": 0.4,
        "keywords": {"tee": 1.2, "denim": 1.1, "jeans": 1.1},
        "pinterest_query": "everyday casual outfit ideas",
    },
    "college": {
        "colors": {"blue": 1.8, "navy": 1.6, "white": 1.5, "grey": 1.3, "beige": 1.1},
        "styles": {"casual", "sporty"}, "separates_bonus": 1.2, "dress_bonus": 0.2,
        "shoe_colors": {"white", "black", "brown"},
        "keywords": {"tee": 1.5, "denim": 1.8, "jeans": 1.6, "sneakers": 1.5},
        "pinterest_query": "college outfit casual sneakers",
    },
    "work": {
        "colors": {"black": 2.0, "navy": 1.9, "grey": 1.8, "beige": 1.6, "white": 1.5, "brown": 1.2},
        "styles": {"smart", "formal"}, "separates_bonus": 2.0, "dress_bonus": 1.0,
        "shoe_colors": {"black", "brown", "beige"}, "outerwear_bonus": 1.2,
        "keywords": {"trousers": 1.8, "pinstripe": 1.6, "plaid": 1.2, "skirt": 1.0, "blazer": 2.0},
        "avoid_keywords": {
            "tee": 1.4, "denim": 1.2, "jeans": 1.2, "leopard": 2.0,
            "track": 4.0, "jogger": 4.0, "sweatpant": 4.0,
        },
        "avoid_styles": {"sporty": 3.5},
        "pinterest_query": "office outfit smart workwear",
    },
    "party": {
        "colors": {"burgundy": 2.2, "red": 2.1, "black": 1.9, "pink": 1.8, "purple": 1.8, "teal": 1.4},
        "styles": {"formal", "smart"}, "separates_bonus": 0.2, "dress_bonus": 2.5,
        "shoe_colors": {"black", "red", "burgundy", "beige"}, "statement_bonus": 1.4,
        "keywords": {"dress": 2.2, "floral": 1.8, "burgundy": 1.5, "leopard": 1.0, "pinstripe": 0.6},
        "avoid_keywords": {"track": 3.5, "jogger": 3.5, "sweatpant": 3.5},
        "avoid_styles": {"sporty": 3.0},
        "pinterest_query": "party outfit evening fashion",
    },
}


def occasion_pinterest_url(occasion: str) -> str:
    return "https://www.pinterest.com/search/pins/?q=" + quote(OCCASION_PROFILES[occasion]["pinterest_query"])


def _color_score(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if a == b:
        return 2.0
    if {a, b} in ({"blue", "brown"}, {"green", "beige"}, {"pink", "navy"}):
        return 1.5
    if a in NEUTRALS or b in NEUTRALS:
        return 1.3
    return 0.2


def _score(outfit: tuple[dict, ...], occasion: str, trend_tags: set[str]) -> tuple[float, list[str]]:
    pairs = list(combinations(outfit, 2))
    color = 3 * sum(_color_score(first["color"], second["color"]) for first, second in pairs) / len(pairs)
    styles = [item["style"] for item in outfit]
    style = 1.5 if len(set(styles)) == 1 else 0.5 if len(set(styles)) == 2 else 0
    profile = OCCASION_PROFILES[occasion]
    profile_colors = profile["colors"]
    occasion_score = sum(profile_colors.get(item["color"].lower(), 0) for item in outfit) / len(outfit)
    occasion_score += 0.7 * sum(item["style"] in profile["styles"] for item in outfit) / len(outfit)
    occasion_score += 0.5 * sum(item["occasion"] == occasion for item in outfit) / len(outfit)
    categories = {item["category"] for item in outfit}
    if {"top", "bottom"}.issubset(categories):
        occasion_score += profile["separates_bonus"]
    if "dress" in categories:
        occasion_score += profile["dress_bonus"]
    shoes = next((item for item in outfit if item["category"] == "shoes"), None)
    if shoes and shoes["color"].lower() in profile.get("shoe_colors", set()):
        occasion_score += 0.8
    if "outerwear" in categories:
        occasion_score += profile.get("outerwear_bonus", 0)
    if any(item["color"].lower() not in NEUTRALS for item in outfit):
        occasion_score += profile.get("statement_bonus", 0)
    names = [item.get("name", "").lower() for item in outfit]
    occasion_score += sum(
        weight for keyword, weight in profile.get("keywords", {}).items()
        if any(keyword in name for name in names)
    )
    occasion_score -= sum(
        weight for keyword, weight in profile.get("avoid_keywords", {}).items()
        if any(keyword in name for name in names)
    )
    occasion_score -= sum(profile.get("avoid_styles", {}).get(item["style"], 0) for item in outfit)
    trend_matches = [item["color"] for item in outfit if item["color"].lower() in trend_tags]
    reasons = ["Coordinated colours" if color >= 3.9 else "A varied colour combination"]
    if style >= 1.5:
        reasons.append("Consistent style")
    if occasion_score >= 2:
        reasons.append(f"Pinterest-inspired {OCCASION_LABELS[occasion]} profile")
    if trend_matches:
        reasons.append(f"Includes trending {trend_matches[0]}")
    return color + style + occasion_score + min(len(trend_matches), 2) * 0.7, reasons


def rank_outfits(items: list[dict], occasion: str, trend_tags: set[str] | None = None, limit: int | None = 3) -> list[dict]:
    trend_tags = {tag.lower() for tag in (trend_tags or set())}
    available = [item for item in items if item.get("available", 1)]
    groups = {category: [item for item in available if item["category"] == category][:50]
              for category in ("top", "bottom", "shoes", "dress", "outerwear")}
    if not groups["shoes"] or not (groups["dress"] or (groups["top"] and groups["bottom"])):
        return []
    base_outfits = []
    if groups["top"] and groups["bottom"]:
        base_outfits.extend(product(groups["top"], groups["bottom"], groups["shoes"]))
    base_outfits.extend(product(groups["dress"], groups["shoes"]))
    ranked = []
    for outfit in base_outfits:
        score, reasons = _score(outfit, occasion, trend_tags)
        ranked.append({"items": outfit, "score": round(score, 2), "reasons": reasons})
    ranked.sort(key=lambda result: result["score"], reverse=True)
    if groups["outerwear"]:
        # Add a small number of layered choices without multiplying the entire wardrobe.
        layered = []
        for result in ranked[:60]:
            options = []
            for layer in groups["outerwear"]:
                outfit = (*result["items"], layer)
                score, reasons = _score(outfit, occasion, trend_tags)
                options.append({"items": outfit, "score": round(score + 0.15, 2), "reasons": reasons})
            layered.extend(sorted(options, key=lambda choice: choice["score"], reverse=True)[:2])
        ranked.extend(layered)
        ranked.sort(key=lambda result: result["score"], reverse=True)
    # Make the visible recommendations genuinely different when the wardrobe
    # contains enough choices. The first three prefer not to reuse any garment.
    selected, selected_keys, used_item_ids = [], set(), set()
    diverse_target = min(3, len(ranked)) if limit is None else min(limit, 3)
    for result in ranked:
        item_ids = {item["id"] for item in result["items"]}
        if item_ids.isdisjoint(used_item_ids):
            selected.append(result)
            selected_keys.add(tuple(item["id"] for item in result["items"]))
            used_item_ids.update(item_ids)
        if len(selected) == diverse_target:
            break
    for result in ranked:
        key = tuple(item["id"] for item in result["items"])
        if key not in selected_keys:
            selected.append(result)
            selected_keys.add(key)
        if len(selected) == limit:
            break
    return selected


def rank_partial_outfits(items: list[dict], occasion: str, trend_tags: set[str] | None = None, limit: int = 3) -> list[dict]:
    """Pair owned items when exactly one of top, bottom, shoes is missing."""
    trend_tags = {tag.lower() for tag in (trend_tags or set())}
    groups = {category: [item for item in items if item["category"] == category and item.get("available", 1)][:50]
              for category in ("top", "bottom", "shoes")}
    present = [category for category, group in groups.items() if group]
    if len(present) != 2:
        return []
    missing = next(category for category in groups if not groups[category])
    results = []
    for first, second in product(groups[present[0]], groups[present[1]]):
        score = _color_score(first["color"], second["color"])
        score += 1 if first["style"] == second["style"] else 0
        score += sum(1 for item in (first, second) if item["occasion"] == occasion)
        score += sum(0.7 for item in (first, second) if item["color"].lower() in trend_tags)
        results.append({"items": (first, second), "missing": missing, "score": round(score, 2)})
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:limit]


SHOPPING_IDEAS = (
    {"category": "top", "color": "white", "style": "casual", "name": "white everyday top"},
    {"category": "top", "color": "blue", "style": "smart", "name": "blue smart shirt"},
    {"category": "bottom", "color": "black", "style": "casual", "name": "black trousers"},
    {"category": "bottom", "color": "beige", "style": "smart", "name": "beige trousers"},
    {"category": "shoes", "color": "white", "style": "casual", "name": "white sneakers"},
    {"category": "shoes", "color": "black", "style": "formal", "name": "black formal shoes"},
    {"category": "outerwear", "color": "beige", "style": "smart", "name": "beige layering jacket"},
    {"category": "outerwear", "color": "black", "style": "casual", "name": "black everyday jacket"},
)


def suggest_to_buy(items: list[dict], trend_tags: set[str] | None = None, limit: int = 3) -> list[dict]:
    if not items:
        return []
    trend_tags = {tag.lower() for tag in (trend_tags or set())}
    owned = {(item["category"], item["color"].lower(), item["style"]) for item in items}
    categories = {category: [item for item in items if item["category"] == category]
                  for category in ("top", "bottom", "shoes", "dress", "outerwear")}
    has_dress_outfit = bool(categories["dress"] and categories["shoes"])
    ideas = []
    for candidate in SHOPPING_IDEAS:
        if (candidate["category"], candidate["color"], candidate["style"]) in owned:
            continue
        others = [item for category, group in categories.items() if category != candidate["category"] for item in group]
        compatibility = sum(
            1 for item in others
            if _color_score(candidate["color"], item["color"]) >= 1
            and (candidate["style"] == item["style"] or candidate["style"] == "casual")
        )
        missing_required = not categories[candidate["category"]]
        if candidate["category"] in ("top", "bottom") and has_dress_outfit:
            missing_required = False
        gap_bonus = 5 if missing_required and candidate["category"] != "outerwear" else 0
        dress_layer_bonus = 3 if candidate["category"] == "outerwear" and has_dress_outfit and not categories["outerwear"] else 0
        trend_bonus = 2 if candidate["color"] in trend_tags else 0
        idea = dict(candidate)
        idea["score"] = compatibility + gap_bonus + dress_layer_bonus + trend_bonus
        if gap_bonus:
            idea["why"] = f"Fills your missing {candidate['category']} category"
        elif dress_layer_bonus:
            idea["why"] = "Adds an optional layer to your dress outfits"
        else:
            idea["why"] = f"Pairs with {compatibility} items in your wardrobe"
        idea["pinterest_url"] = "https://www.pinterest.com/search/pins/?q=" + quote(candidate["name"] + " outfit")
        ideas.append(idea)
    ideas.sort(key=lambda idea: idea["score"], reverse=True)
    return ideas[:limit]
