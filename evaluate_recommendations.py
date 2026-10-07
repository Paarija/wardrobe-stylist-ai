"""Summarize blinded ratings for rules, colour-only, and random outfits."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from fashion_ai.db import init_db, list_outfit_ratings


def rating_summary(rows: list[dict]) -> dict:
    ratings = [row["rating"] for row in rows]
    mean = sum(ratings) / len(ratings)
    if len(ratings) > 1:
        variance = sum((rating - mean) ** 2 for rating in ratings) / (len(ratings) - 1)
        margin = 1.96 * math.sqrt(variance / len(ratings))
    else:
        margin = 0.0
    return {
        "ratings": len(ratings),
        "mean_rating": round(mean, 3),
        "mean_rating_95pct_normal_interval": [round(max(1, mean - margin), 3), round(min(5, mean + margin), 3)],
        "rating_variation": len(set(ratings)),
    }


def group_metrics(rows: list[dict]) -> dict:
    by_method = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "rules")].append(row)
    rules = by_method.get("rules", [])
    baselines = [row for method, group in by_method.items() if method != "rules" for row in group]
    pair_wins = pair_count = 0
    for rule in rules:
        for baseline in baselines:
            if rule["rating"] != baseline["rating"]:
                pair_count += 1
                pair_wins += rule["rating"] > baseline["rating"]
    return {
        "occasion": rows[0]["occasion"],
        "rated_outfits": len(rows),
        "candidate_pool_size": max((row.get("candidate_pool_size", 0) for row in rows), default=0),
        "methods": {method: rating_summary(group) for method, group in sorted(by_method.items())},
        "informative_rules_vs_baseline_pairs": pair_count,
        "rules_preferred_pairs": pair_wins,
        "pairwise_preference_accuracy": round(pair_wins / pair_count, 4) if pair_count else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blinded outfit-ranking baselines")
    parser.add_argument("--user-id", type=int, help="Restrict the local study to one account")
    parser.add_argument("--output", type=Path, default=Path("reports/recommendation_study.json"))
    args = parser.parse_args()
    init_db()
    rows = list_outfit_ratings(args.user_id)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["user_id"], row["study_id"])].append(row)
    completed = [group_metrics(group) for group in groups.values() if len(group) >= 4]
    all_by_method = defaultdict(list)
    for row in rows:
        all_by_method[row.get("method", "rules")].append(row)
    pairs = sum(group["informative_rules_vs_baseline_pairs"] for group in completed)
    wins = sum(group["rules_preferred_pairs"] for group in completed)
    report = {
        "status": "awaiting_ratings" if not completed else "pilot_results",
        "method": "Blinded 1-to-5 ratings comparing rule-ranked, colour-only, and random valid outfits",
        "ratings_recorded": len(rows),
        "completed_study_sets": len(completed),
        "participants": len({row["user_id"] for row in rows}),
        "method_summaries": {method: rating_summary(group) for method, group in sorted(all_by_method.items())},
        "pairwise_rules_vs_baselines": round(wins / pairs, 4) if pairs else None,
        "informative_pair_count": pairs,
        "groups": completed,
        "notes": [
            "Presentation order is deterministically shuffled and hides the candidate source.",
            "Each study rates a selected subset from candidate_pool_size, not every possible wardrobe outfit.",
            "Normal confidence intervals are descriptive only for this small pilot; participant count and rating variation are reported.",
            "Ratings are not used to train or tune the recommender.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "ratings_recorded", "completed_study_sets", "participants",
        "pairwise_rules_vs_baselines",
    )}))


if __name__ == "__main__":
    main()
