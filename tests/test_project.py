from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

import app
from evaluate_classifier import manifest_examples
from fine_tune_atr import swap_atr_left_right
from fashion_ai import db
from fashion_ai.classifier import category_from_dataset, infer_color, infer_colors
from fashion_ai.color import color_signature
from fashion_ai.checkpoints import config_sha256
from fashion_ai.recommender import rank_outfits, rank_partial_outfits, suggest_to_buy
from fashion_ai.segmentation import extract_items, merge_outerwear_probability, remap_atr_mask, remap_lip_mask
from fashion_ai.study import study_candidates
from evaluate_recommendations import group_metrics
from select_fashionpedia_ensemble import select_threshold
from validate_wardrobe_dataset import validate_manifest


def test_wardrobes_stay_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "wardrobe.db")
    monkeypatch.setattr(db, "UPLOAD_DIR", tmp_path / "uploads")
    db.init_db()
    alex = db.register_user("alex", "password123")
    sam = db.register_user("sam", "password123")
    assert db.authenticate("alex", "password123") == alex
    assert db.authenticate("alex", "wrongpassword") is None

    image = Image.new("RGB", (30, 30), "blue")
    item_id = db.add_item(alex, image, name="Blue shirt", category="top", color="blue", style="casual", occasion="everyday")
    assert len(db.list_items(alex)) == 1
    assert db.list_items(sam) == []
    db.delete_item(sam, item_id)
    assert len(db.list_items(alex)) == 1
    db.update_item(sam, item_id, name="Wrong", category="shoes", color="black", style="formal", occasion="party")
    assert db.list_items(alex)[0]["category"] == "top"
    db.update_item(alex, item_id, name="Blue trousers", category="bottom", color="blue", style="casual", occasion="everyday")
    assert db.list_items(alex)[0]["category"] == "bottom"
    assert db.list_items(alex)[0]["reviewed"] == 1
    db.update_item_color(alex, item_id, "teal")
    assert db.list_items(alex)[0]["color"] == "teal"
    signature = color_signature(image)
    db.save_color_feedback(alex, item_id, "teal", signature)
    assert db.list_color_feedback(alex)[0]["label"] == "teal"
    with pytest.raises(ValueError, match="name"):
        db.update_item(alex, item_id, name="  ", category="bottom", color="blue", style="casual", occasion="everyday")
    db.delete_item(alex, item_id)
    assert db.list_items(alex) == []
    assert db.list_color_feedback(alex) == []


def test_outfits_only_contain_available_uploaded_items():
    items = [
        {"id": "t", "category": "top", "color": "white", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "b", "category": "bottom", "color": "black", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "s", "category": "shoes", "color": "white", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "s2", "category": "shoes", "color": "red", "style": "formal", "occasion": "party", "available": 0},
    ]
    outfits = rank_outfits(items, "college", {"white"})
    assert [item["id"] for item in outfits[0]["items"]] == ["t", "b", "s"]
    assert rank_outfits(items[:-2], "college") == []


def test_purchase_ideas_skip_items_already_owned():
    items = [{"category": "shoes", "color": "white", "style": "casual"}]
    ideas = suggest_to_buy(items, limit=10)
    assert "white sneakers" not in [idea["name"] for idea in ideas]
    assert all(idea["pinterest_url"].startswith("https://www.pinterest.com/search/pins/") for idea in ideas)
    assert suggest_to_buy([]) == []


def test_purchase_ideas_treat_dress_and_shoes_as_complete():
    items = [
        {"category": "dress", "color": "navy", "style": "smart"},
        {"category": "shoes", "color": "black", "style": "smart"},
    ]
    ideas = suggest_to_buy(items, limit=10)
    assert not any("missing top" in idea["why"] or "missing bottom" in idea["why"] for idea in ideas)
    assert any(idea["category"] == "outerwear" and "dress outfits" in idea["why"] for idea in ideas)


def test_partial_outfits_show_missing_category():
    items = [
        {"id": "t", "category": "top", "color": "navy", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "b", "category": "bottom", "color": "brown", "style": "casual", "occasion": "everyday", "available": 1},
    ]
    ideas = rank_partial_outfits(items, "everyday")
    assert len(ideas) == 1
    assert ideas[0]["missing"] == "shoes"
    assert [item["id"] for item in ideas[0]["items"]] == ["t", "b"]


def test_lip_masks_remap_and_extract_clothes():
    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[2:18, 5:35] = 5
    raw[18:32, 5:35] = 9
    raw[32:39, 5:35] = 18
    mapped = remap_lip_mask(Image.fromarray(raw))
    assert {1, 2, 3}.issubset(set(np.unique(mapped)))
    crops = extract_items(Image.new("RGB", (40, 40), "red"), Image.fromarray(mapped), minimum_pixels=20)
    assert set(crops) == {"top", "bottom", "shoes"}


def test_atr_clothing_labels_match_app_categories():
    raw = Image.fromarray(np.array([[4, 5, 6, 9, 10, 7, 11, 0]], dtype=np.uint8))
    assert remap_atr_mask(raw).tolist() == [[1, 2, 2, 3, 3, 4, 0, 0]]


def test_atr_horizontal_flip_swaps_all_directional_classes():
    labels = np.array([[9, 10, 12, 13, 14, 15, 4]])
    assert swap_atr_left_right(labels).tolist() == [[10, 9, 13, 12, 15, 14, 4]]


def test_outerwear_confidence_preserves_other_garments():
    base = np.array([[1, 1, 2, 0]], dtype=np.uint8)
    confidence = np.array([[0.62, 0.61, 0.99, 0.99]], dtype=np.float32)
    assert merge_outerwear_probability(base, confidence, 0.62).tolist() == [[5, 1, 2, 0]]


def test_product_dataset_category_mapping_and_color():
    assert category_from_dataset("Footwear", "Shoes", "Sneakers") == "shoes"
    assert category_from_dataset("Apparel", "Bottomwear", "Skirts") == "bottom"
    assert category_from_dataset("Apparel", "Topwear", "Tops") == "top"
    assert category_from_dataset("Apparel", "Topwear", "Sweaters") == "top"
    assert category_from_dataset("Apparel", "Topwear", "Sweatshirts") == "top"
    assert category_from_dataset("Apparel", "Topwear", "Jackets") == "top"
    assert category_from_dataset("Apparel", "Topwear", "Waistcoat") == "top"
    assert category_from_dataset("Accessories", "Bags", "Handbags") is None
    assert infer_color(Image.new("RGB", (40, 40), "navy")) == "navy"
    assert infer_color(Image.new("RGB", (40, 40), (255, 0, 0))) == "red"
    assert infer_color(Image.new("RGB", (40, 40), (0, 255, 0))) == "green"
    assert infer_color(Image.new("RGB", (40, 40), (0, 0, 255))) == "blue"
    product = Image.new("RGB", (100, 100), "white")
    product.paste(Image.new("RGB", (60, 60), "black"), (20, 20))
    assert infer_colors(product)[0][0] == "black"
    burgundy_product = Image.new("RGB", (100, 100), "#eeeeee")
    burgundy_product.paste(Image.new("RGB", (60, 70), "#722f3a"), (20, 15))
    assert infer_colors(burgundy_product)[0][0] == "burgundy"
    assert infer_color(Image.new("RGB", (40, 40), "#187b80")) == "teal"
    ambiguous = Image.new("RGB", (40, 40), "#777267")
    learned = [{"label": "brown", "lab_l": value[0], "lab_a": value[1], "lab_b": value[2]}
               for value in [color_signature(ambiguous)]]
    assert infer_colors(ambiguous, calibration=learned)[0][0] == "brown"


def test_phone_photo_orientation_is_applied():
    raw = io.BytesIO()
    image = Image.new("RGB", (10, 20), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(raw, format="JPEG", exif=exif)
    assert app.readable_image(raw).size == (20, 10)


def test_blinded_outfit_ratings_persist_and_evaluate(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "wardrobe.db")
    monkeypatch.setattr(db, "UPLOAD_DIR", tmp_path / "uploads")
    db.init_db()
    user_id = db.register_user("rater", "password123")
    items = [
        {"id": f"{category}{number}", "category": category, "color": color,
         "style": "casual", "occasion": "everyday", "available": 1}
        for category, colors in (("top", ("white", "blue")),
                                 ("bottom", ("black", "beige")),
                                 ("shoes", ("white", "brown")))
        for number, color in enumerate(colors)
    ]
    study_id, choices = study_candidates(items, "everyday", {"blue"}, user_id)
    assert len(choices) == 6
    assert sorted(choice["display_position"] for choice in choices) == [1, 2, 3, 4, 5, 6]
    assert {method: sum(choice["method"] == method for choice in choices)
            for method in ("rules", "colour_baseline", "random")} == {
                "rules": 2, "colour_baseline": 2, "random": 2,
            }
    assert all(choice["candidate_pool_size"] == 8 for choice in choices)
    assert study_candidates(items, "everyday", {"blue"}, user_id) == (study_id, choices)
    rated = [dict(choice, rating=5 if choice["method"] == "rules" else 1) for choice in choices]
    db.save_outfit_ratings(user_id, study_id, "everyday", rated)
    saved = db.list_outfit_ratings(user_id)
    assert len(saved) == 6
    metrics = group_metrics(saved)
    assert metrics["pairwise_preference_accuracy"] == 1.0
    assert metrics["methods"]["rules"]["mean_rating"] == 5
    assert metrics["methods"]["random"]["mean_rating"] == 1


def test_personal_manifest_reads_labelled_photos(tmp_path):
    Image.new("RGB", (20, 20), "blue").save(tmp_path / "shirt.jpg")
    manifest = tmp_path / "labels.csv"
    manifest.write_text("image,label\nshirt.jpg,top\n", encoding="utf-8")
    examples, metadata = manifest_examples(manifest)
    assert [label for _, label in examples] == ["top"]
    assert metadata["image_count"] == 1


def test_dress_and_outerwear_can_make_a_complete_outfit():
    items = [
        {"id": key, "category": category, "color": color, "style": "casual",
         "occasion": "everyday", "available": 1}
        for key, category, color in (("d", "dress", "navy"), ("s", "shoes", "navy"),
                                     ("j", "outerwear", "navy"))
    ]
    ranked = rank_outfits(items, "everyday", limit=None)
    assert any([item["category"] for item in result["items"]] == ["dress", "shoes"] for result in ranked)
    assert any([item["category"] for item in result["items"]] == ["dress", "shoes", "outerwear"] for result in ranked)


def test_office_and_party_profiles_choose_different_outfits():
    items = [
        {"id": "office-top", "category": "top", "color": "white", "style": "smart", "occasion": "work", "available": 1},
        {"id": "office-bottom", "category": "bottom", "color": "grey", "style": "smart", "occasion": "work", "available": 1},
        {"id": "party-dress", "category": "dress", "color": "burgundy", "style": "formal", "occasion": "party", "available": 1},
        {"id": "black-shoes", "category": "shoes", "color": "black", "style": "formal", "occasion": "party", "available": 1},
    ]
    office = rank_outfits(items, "work", limit=1)[0]
    party = rank_outfits(items, "party", limit=1)[0]
    assert [item["id"] for item in office["items"]] == ["office-top", "office-bottom", "black-shoes"]
    assert [item["id"] for item in party["items"]] == ["party-dress", "black-shoes"]
    assert "Pinterest-inspired office profile" in office["reasons"]
    assert "Pinterest-inspired party profile" in party["reasons"]


def test_pinterest_cues_separate_office_from_everyday():
    items = [
        {"id": "tee", "name": "Navy tee", "category": "top", "color": "navy", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "denim", "name": "Blue denim skirt", "category": "bottom", "color": "blue", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "plaid", "name": "Beige plaid top", "category": "top", "color": "beige", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "pinstripe", "name": "Black pinstripe skirt", "category": "bottom", "color": "black", "style": "casual", "occasion": "everyday", "available": 1},
        {"id": "shoes", "name": "Black shoes", "category": "shoes", "color": "black", "style": "casual", "occasion": "everyday", "available": 1},
    ]
    everyday_ids = [item["id"] for item in rank_outfits(items, "everyday", limit=1)[0]["items"]]
    office_ids = [item["id"] for item in rank_outfits(items, "work", limit=1)[0]["items"]]
    assert everyday_ids == ["tee", "denim", "shoes"]
    assert office_ids == ["plaid", "pinstripe", "shoes"]


def test_formal_profiles_reject_track_pants_when_an_alternative_exists():
    items = [
        {"id": "top", "name": "White smart shirt", "category": "top", "color": "white", "style": "smart", "occasion": "work", "available": 1},
        {"id": "track", "name": "Brown track pants", "category": "bottom", "color": "brown", "style": "sporty", "occasion": "everyday", "available": 1},
        {"id": "skirt", "name": "Black pinstripe skirt", "category": "bottom", "color": "black", "style": "smart", "occasion": "work", "available": 1},
        {"id": "shoes", "name": "Black formal shoes", "category": "shoes", "color": "black", "style": "formal", "occasion": "work", "available": 1},
    ]
    for occasion in ("work", "party"):
        chosen_ids = {item["id"] for item in rank_outfits(items, occasion, limit=1)[0]["items"]}
        assert "skirt" in chosen_ids
        assert "track" not in chosen_ids


def test_monochrome_beats_clashing_colours():
    items = [
        {"id": str(index), "category": category, "color": color, "style": "casual",
         "occasion": "everyday", "available": 1}
        for index, (category, color) in enumerate((
            ("top", "navy"), ("bottom", "navy"), ("shoes", "navy"),
            ("top", "red"), ("bottom", "white"), ("shoes", "green")))
    ]
    ranked = rank_outfits(items, "everyday", limit=None)
    assert [item["color"] for item in ranked[0]["items"]] == ["navy", "navy", "navy"]
    assert ranked[0]["score"] > next(result["score"] for result in ranked if [item["color"] for item in result["items"]] == ["red", "white", "green"])


def test_visible_recommendations_use_distinct_garments_when_possible():
    items = [
        {"id": f"{category}{index}", "category": category, "color": "navy",
         "style": "casual", "occasion": "everyday", "available": 1}
        for category in ("top", "bottom", "shoes")
        for index in range(3)
    ]
    ranked = rank_outfits(items, "everyday", limit=3)
    ids = [item["id"] for result in ranked for item in result["items"]]
    assert len(ranked) == 3
    assert len(ids) == len(set(ids)) == 9
    assert all([item["category"] for item in result["items"]] == ["top", "bottom", "shoes"] for result in ranked)


def test_rating_set_includes_all_five_when_only_five_exist():
    items = [
        {"id": f"{category}{index}", "category": category, "color": "navy",
         "style": "casual", "occasion": "everyday", "available": 1}
        for category, count in (("top", 1), ("bottom", 1), ("shoes", 5))
        for index in range(count)
    ]
    _, choices = study_candidates(items, "everyday", set(), 1)
    assert len(choices) == 5


def test_verified_checkpoint_and_portable_ensemble_hash(tmp_path, monkeypatch):
    atr = tmp_path / "models" / "atr-segformer"
    atr.mkdir(parents=True)
    weights = atr / "model.safetensors"
    weights.write_bytes(b"checkpoint fixture")
    (atr / "config.json").write_text(json.dumps({"id2label": {str(i): str(i) for i in range(18)}}), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "segmentation_tuning_summary.json").write_text(
        json.dumps({"selected_checkpoint": "models/atr-segformer",
                    "selected_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}), encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)
    assert app.active_segmentation_checkpoint() == atr

    outerwear = tmp_path / "models" / "fashionpedia-segformer"
    outerwear.mkdir()
    outerwear_weights = outerwear / "model.safetensors"
    outerwear_weights.write_bytes(b"outerwear fixture")
    (outerwear / "config.json").write_text(json.dumps({"id2label": {str(i): str(i) for i in range(6)}}), encoding="utf-8")
    ensemble = tmp_path / "models" / "fashionpedia-ensemble"
    ensemble.mkdir()
    config = {"base": "../atr-segformer", "outerwear": "../fashionpedia-segformer", "threshold": 0.62}
    (ensemble / "ensemble.json").write_bytes(json.dumps(config, indent=2).replace("\n", "\r\n").encode())
    (reports / "fashionpedia_accuracy_summary.json").write_text(json.dumps({
        "selected_checkpoint": "models/fashionpedia-ensemble",
        "selected_sha256": config_sha256(config),
        "components": {
            "base": {"checkpoint": "models/atr-segformer", "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()},
            "outerwear": {"checkpoint": "models/fashionpedia-segformer", "sha256": hashlib.sha256(outerwear_weights.read_bytes()).hexdigest()},
        },
    }), encoding="utf-8")
    assert app.active_segmentation_checkpoint() == ensemble
    (ensemble / "ensemble.json").write_text("{invalid", encoding="utf-8")
    assert app.active_segmentation_checkpoint() == atr
    (ensemble / "ensemble.json").write_text(json.dumps(config), encoding="utf-8")
    (outerwear / "config.json").write_text("{}", encoding="utf-8")
    assert app.active_segmentation_checkpoint() == atr
    (outerwear / "config.json").write_text(json.dumps({"id2label": {str(i): str(i) for i in range(6)}}), encoding="utf-8")
    outerwear_weights.write_bytes(b"damaged")
    assert app.active_segmentation_checkpoint() == atr
    weights.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="do not match"):
        app.active_segmentation_checkpoint()


def test_ensemble_selection_rejects_report_for_different_weights():
    base_sha, outerwear_sha = "base-hash", "outerwear-hash"
    report = {
        "image_ids": [1, 2],
        "models": {
            "base": {"checkpoint_sha256": base_sha, "supported_clothing_miou": 0.6,
                     "all_clothing_miou": 0.48, "iou_by_class": {}},
            "fine_tuned": {"checkpoint_sha256": outerwear_sha},
            "threshold_0.62": {
                "base_checkpoint_sha256": base_sha,
                "outerwear_checkpoint_sha256": outerwear_sha,
                "outerwear_probability_threshold": 0.62,
                "supported_clothing_miou": 0.5995,
                "all_clothing_miou": 0.52,
                "iou_by_class": {},
            },
        },
    }
    assert select_threshold(report, base_sha, outerwear_sha) == "threshold_0.62"
    with pytest.raises(ValueError, match="Fashionpedia weights"):
        select_threshold(report, base_sha, "different-hash")


def test_wardrobe_manifest_rejects_garment_split_leakage(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    rows = []
    for index, split in enumerate(("train", "development", "test"), 1):
        image_path = photos / f"item-{index}.jpg"
        Image.new("RGB", (8, 8), (index * 30, 20, 10)).save(image_path)
        rows.append(
            f"photos/{image_path.name},top,garment-{index},session-{index},{split},true,participant,true"
        )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image,label,garment_id,capture_session,split,reviewed,source,consent\n"
        + "\n".join(rows),
        encoding="utf-8",
    )
    report = validate_manifest(manifest)
    assert report["photos"] == 3
    assert report["split_counts"] == {"development": 1, "test": 1, "train": 1}

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("garment-2", "garment-1"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="garment_id crosses dataset splits"):
        validate_manifest(manifest)
