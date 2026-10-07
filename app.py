"""Streamlit UI for the wardrobe and segmentation portfolio project."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from fashion_ai.db import CATEGORIES, add_item, authenticate, delete_item, init_db, list_color_feedback, list_items, list_outfit_ratings, register_user, save_color_feedback, save_outfit_ratings, set_availability, update_item, update_item_color
from fashion_ai.classifier import MODEL_PATH as CLASSIFIER_PATH, classify_item
from fashion_ai.color import PALETTE as COLOR_HEX, color_signature, infer_colors
from fashion_ai.checkpoints import active_segmentation_checkpoint as select_segmentation_checkpoint
from fashion_ai.recommender import COLORS, OCCASIONS, STYLES, occasion_pinterest_url, rank_outfits, rank_partial_outfits, suggest_to_buy
from fashion_ai.segmentation import extract_items, overlay_mask, predict_mask
from fashion_ai.study import study_candidates


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def active_segmentation_checkpoint() -> Path:
    return select_segmentation_checkpoint(ROOT)


def readable_image(uploaded) -> Image.Image:
    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Use an image under 10 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        with Image.open(io.BytesIO(raw)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("This file is not a readable image.") from exc


def trend_data() -> tuple[set[str], dict]:
    info = json.loads((ROOT / "data" / "trend_cues.json").read_text(encoding="utf-8"))
    tags = {tag for trend in info["trends"] for tag in trend["colors"]}
    return tags, info


def show_colour_result(image: Image.Image, user_id: int) -> str:
    palette = infer_colors(image, calibration=list_color_feedback(user_id))
    dominant = palette[0][0]
    swatch = (
        f'<span style="display:inline-block;width:18px;height:18px;border-radius:50%;background:{COLOR_HEX[dominant]};'
        'border:1px solid #555;vertical-align:middle;margin:0 6px"></span>'
    )
    st.markdown(f"**Detected primary colour:** {swatch}{dominant.title()}", unsafe_allow_html=True)
    visible_tones = [f"{name.title()} {share:.0%}" for name, share in palette[1:] if share >= 0.08]
    if visible_tones:
        st.caption(f"Other visible tones: {', '.join(visible_tones)}")
    return dominant


def login_page() -> None:
    st.title("Wardrobe Stylist AI")
    st.write("Upload what you own. Build outfits. Find your next style idea on Pinterest.")
    entry = st.radio("Choose how to enter", ["Create account", "Sign in"], horizontal=True)
    if entry == "Create account":
        st.subheader("New here? Create an account")
        st.caption("No email is needed. Choose a username and password for this local app.")
        with st.form("sign_up"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password (8+ characters)", type="password")
            created = st.form_submit_button("Create account", width="stretch")
        if created:
            try:
                st.session_state.user_id = register_user(new_username, new_password)
                st.session_state.username = new_username.strip().lower()
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.subheader("Welcome back")
        with st.form("sign_in"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", width="stretch")
        if submitted:
            user_id = authenticate(username, password)
            if user_id is None:
                st.error("Username or password is incorrect.")
            else:
                st.session_state.user_id = user_id
                st.session_state.username = username.strip().lower()
                st.rerun()


def upload_page(user_id: int) -> None:
    st.header("Add clothes")
    st.write("Upload one clothing item per image. The trained image model will identify its category automatically.")
    st.caption("Colour is estimated from the image. Style starts as casual and can be edited later in My wardrobe.")
    st.info("For a photo of a person wearing several pieces, use **Outfit segmentation**. It runs the neural segmentation model and extracts each garment automatically.")
    if st.button("Open outfit segmentation"):
        st.session_state.page_nav = "Outfit segmentation"
        st.rerun()
    if not CLASSIFIER_PATH.is_file():
        st.error("The garment classifier checkpoint is missing. Restore models/garment_classifier.keras from the repository.")
        return
    uploads = st.file_uploader("Clothing photos", type=["jpg", "jpeg", "jfif", "png", "webp"], accept_multiple_files=True)
    if not uploads:
        return
    details = []
    for index, upload in enumerate(uploads):
        with st.expander(upload.name, expanded=True):
            try:
                image = readable_image(upload)
                st.image(image, width=220)
            except ValueError as exc:
                st.error(str(exc))
                continue
            if CLASSIFIER_PATH.is_file():
                try:
                    category, confidence = classify_item(image, user_id=user_id)
                    detected_color = show_colour_result(image, user_id)
                    color = st.selectbox(
                        "Colour (correct it to teach future detections)",
                        COLORS,
                        index=COLORS.index(detected_color),
                        key=f"upload_color_{hashlib.sha256(upload.getvalue()).hexdigest()}",
                    )
                    st.markdown(f"**Detected item: {color} {category}** · {confidence:.0%} category confidence")
                    if confidence < 0.55:
                        st.caption("Low confidence. You can change this item's details later in My wardrobe.")
                    details.append((image, f"{color.title()} {category}", category, color, detected_color))
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Could not classify this photo: {exc}")
    if st.button("Save detected clothes", type="primary", disabled=not details):
        saved = 0
        for image, name, category, color, detected_color in details:
            try:
                item_id = add_item(user_id, image, name=name, category=category, color=color, style="casual", occasion="everyday")
                if color != detected_color:
                    save_color_feedback(user_id, item_id, color, color_signature(image))
                saved += 1
            except ValueError as exc:
                st.error(str(exc))
        st.success(f"Saved {saved} item(s) to your wardrobe.")


def segment_page(user_id: int) -> None:
    st.header("Segment an outfit photo")
    try:
        model_path = active_segmentation_checkpoint()
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"The segmentation checkpoint cannot be verified: {exc}")
        return
    st.write("Upload a photo showing one person in an outfit. The neural model automatically marks clothing regions, then detects each garment's colour.")
    is_ensemble = (model_path / "ensemble.json").is_file()
    if is_ensemble:
        st.caption("The outfit model can mark tops, bottoms, shoes, dresses, and outerwear. Review each extracted item before saving it.")
    else:
        st.caption("The ATR fallback detects tops, bottoms, shoes, and dresses, but does not separate outerwear.")
    photo = st.file_uploader("Outfit photo", type=["jpg", "jpeg", "jfif", "png", "webp"], key="outfit_photo")
    if photo is None:
        return
    try:
        image = readable_image(photo)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.image(image, width=320, caption="Original photo")
    image_key = hashlib.sha256(photo.getvalue()).hexdigest()
    result = st.session_state.get("segmentation")
    if not result or result["key"] != image_key:
        try:
            with st.spinner("Running neural segmentation and finding clothing regions..."):
                mask = predict_mask(image, model_path)
                st.session_state.segmentation = {"key": image_key, "mask": mask}
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"Could not load or run the model: {exc}")
            return
    result = st.session_state.get("segmentation")
    if not result or result["key"] != image_key:
        return
    mask = result["mask"]
    st.image(overlay_mask(image, mask), width=420, caption="Neural segmentation result")
    crops = extract_items(image, mask)
    if not crops:
        st.warning("No clothing region was large enough to extract. Try a clearer full-body photo.")
        return
    st.subheader("Review items before saving")
    st.caption("Blue: top · orange: bottom · green: shoes · purple: dress · red: outerwear")
    for category, crop in crops.items():
        with st.expander(category.title(), expanded=True):
            st.image(crop, width=190)
            detected_color = show_colour_result(crop, user_id)
            color = st.selectbox(
                "Colour (correct it to teach future detections)",
                COLORS,
                index=COLORS.index(detected_color),
                key=f"segment_color_{category}",
            )
            st.caption(f"Detected {color} {category}")
            if st.button(f"Save {category}", key=f"save_{category}"):
                item_id = add_item(user_id, crop, name=f"{color.title()} {category}", category=category, color=color, style="casual", occasion="everyday")
                if color != detected_color:
                    save_color_feedback(user_id, item_id, color, color_signature(crop))
                st.success(f"Saved {category}.")


def wardrobe_page(user_id: int) -> None:
    st.header("My wardrobe")
    items = list_items(user_id)
    if not items:
        st.info("Your wardrobe is empty. Add photos to start creating outfits.")
        return
    st.caption(f"{len(items)} saved item(s)")
    if st.button("Re-detect saved colours"):
        with st.spinner("Removing image backgrounds and analysing garment colours..."):
            calibration = list_color_feedback(user_id)
            for item in items:
                with Image.open(item["image_path"]) as source:
                    detected = infer_colors(source.convert("RGB"), limit=1, calibration=calibration)[0][0]
                    update_item_color(user_id, item["id"], detected)
        st.success("Wardrobe colours were re-detected with the LAB colour pipeline.")
        st.rerun()
    columns = st.columns(3)
    for index, item in enumerate(items):
        with columns[index % 3]:
            with st.container(border=True):
                st.image(item["image_path"], width="stretch")
                st.markdown(f"**{item['name']}**")
                st.caption(f"{item['color']} {item['category']} · {item['style']} · {item['occasion']}")
                st.caption("Label reviewed" if item["reviewed"] else "Label needs review")
                with st.expander("Edit details"):
                    st.caption("Saving confirms that you reviewed this item's label.")
                    with st.form(f"edit_{item['id']}"):
                        name = st.text_input("Name", value=item["name"])
                        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(item["category"]))
                        color = st.selectbox("Colour", COLORS, index=COLORS.index(item["color"]))
                        style = st.selectbox("Style", STYLES, index=STYLES.index(item["style"]))
                        occasion = st.selectbox("Occasion", OCCASIONS, index=OCCASIONS.index(item["occasion"]))
                        if st.form_submit_button("Save changes"):
                            try:
                                update_item(user_id, item["id"], name=name, category=category, color=color, style=style, occasion=occasion)
                                if color != item["color"]:
                                    with Image.open(item["image_path"]) as source:
                                        save_color_feedback(
                                            user_id, item["id"], color,
                                            color_signature(source.convert("RGB")),
                                        )
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))
                available = bool(item["available"])
                if st.button("Mark unavailable" if available else "Mark available", key=f"avail_{item['id']}"):
                    set_availability(user_id, item["id"], not available)
                    st.rerun()
                if st.button("Delete", key=f"delete_{item['id']}"):
                    delete_item(user_id, item["id"])
                    st.rerun()


def recommendations_page(user_id: int) -> None:
    st.header("Outfits and Pinterest ideas")
    items = list_items(user_id)
    trend_tags, trends = trend_data()
    st.caption(f"Trend cues: {', '.join(trend['name'] for trend in trends['trends'])} · Reviewed {trends['reviewed_at']}")
    st.link_button("See Pinterest trend source", trends["source_url"])
    occasion = st.selectbox("Where are you going?", OCCASIONS, format_func=lambda value: "office" if value == "work" else value)
    st.link_button(f"See {('office' if occasion == 'work' else occasion)} inspiration on Pinterest", occasion_pinterest_url(occasion))
    ranked = rank_outfits(items, occasion, trend_tags)
    st.subheader("Wear from your wardrobe")
    if not ranked:
        categories = {item["category"] for item in items if item["available"]}
        if "shoes" not in categories:
            message = "Add or mark available a pair of shoes to complete an outfit."
        else:
            missing = [category for category in ("top", "bottom") if category not in categories]
            message = f"Add a dress or complete a top-and-bottom outfit. Missing separates: {', '.join(missing)}."
        st.info(message)
        st.button("Check wardrobe categories", on_click=lambda: st.session_state.update(page_nav="My wardrobe"))
        partial = rank_partial_outfits(items, occasion, trend_tags)
        if partial:
            st.subheader("Outfit ideas to complete")
            st.caption("These are your clothes. The missing piece is a Pinterest idea until you add it to your wardrobe.")
            ideas_by_category = {idea["category"]: idea for idea in suggest_to_buy(items, trend_tags, limit=6)}
            for number, result in enumerate(partial, 1):
                with st.container(border=True):
                    st.markdown(f"**Idea {number}**")
                    columns = st.columns(3)
                    for column, item in zip(columns, result["items"]):
                        with column:
                            st.image(item["image_path"], width="stretch")
                            st.caption(f"Your {item['name']}")
                    with columns[2]:
                        idea = ideas_by_category.get(result["missing"])
                        st.markdown(f"**Add {result['missing']}**")
                        if idea:
                            st.caption(idea["name"].title())
                            st.link_button("Explore on Pinterest", idea["pinterest_url"])
    for number, result in enumerate(ranked, 1):
        with st.container(border=True):
            st.markdown(f"**Outfit {number}** — {', '.join(result['reasons'])}")
            columns = st.columns(len(result["items"]))
            for column, item in zip(columns, result["items"]):
                with column:
                    st.image(item["image_path"], width="stretch")
                    st.caption(f"{item['name']} · {item['category']}")
    st.subheader("What to add next")
    st.write("Suggestions are based on wardrobe gaps and style fit. Pinterest links let you explore the idea; they are not verified product listings.")
    ideas = suggest_to_buy(items, trend_tags)
    if not ideas:
        st.info("Upload some clothes to get a personalised idea for what to add next.")
    for idea in ideas:
        with st.container(border=True):
            st.markdown(f"**{idea['name'].title()}**")
            st.caption(idea["why"])
            st.link_button("Explore on Pinterest", idea["pinterest_url"])

    st.button("Rate outfit ideas", on_click=lambda: st.session_state.update(page_nav="Rate outfits"))


def rate_outfits_page(user_id: int) -> None:
    st.header("Rate outfit ideas")
    st.write("Help check whether these combinations are useful. Choose an occasion and rate each outfit from 1 (would not wear) to 5 (would love to wear).")
    occasion = st.selectbox("Occasion to rate", OCCASIONS, key="rating_occasion")
    trend_tags, _ = trend_data()
    study_id, choices = study_candidates(list_items(user_id), occasion, trend_tags, user_id)
    if len(choices) < 4:
        st.info("At least four distinct outfits are needed for this comparison. Add more clothes or mark items available.")
        return
    st.caption(f"Compare {len(choices)} outfits for {occasion}. Their ranking is hidden while you rate.")
    previous = {row["outfit_key"]: row["rating"] for row in list_outfit_ratings(user_id) if row["study_id"] == study_id}
    st.caption(f"{len(previous)} of {len(choices)} outfits rated for {occasion}. The model's order and scores are hidden while you rate.")
    with st.form("outfit_rating_form"):
        ratings = {}
        for choice in choices:
            with st.container(border=True):
                st.markdown(f"**Outfit {choice['display_position']}**")
                for column, item in zip(st.columns(len(choice["items"])), choice["items"]):
                    with column:
                        st.image(item["image_path"], width="stretch")
                        st.caption(item["category"].title())
                old = previous.get(choice["outfit_key"])
                ratings[choice["outfit_key"]] = st.radio(
                    f"How would you rate outfit {choice['display_position']}?",
                    [1, 2, 3, 4, 5], index=old - 1 if old else None,
                    horizontal=True, key=f"rating_{study_id}_{choice['outfit_key']}",
                )
        submitted = st.form_submit_button("Save my ratings", type="primary")
    if submitted:
        if any(value is None for value in ratings.values()):
            st.warning("Rate every outfit before saving.")
        else:
            save_outfit_ratings(user_id, study_id, occasion, [dict(choice, rating=ratings[choice["outfit_key"]]) for choice in choices])
            st.success("Your ratings were saved. Choose another occasion to rate more outfits.")


def main() -> None:
    st.set_page_config(page_title="Wardrobe Stylist AI", page_icon="👗", layout="wide")
    init_db()
    if "user_id" not in st.session_state:
        login_page()
        return
    user_id = st.session_state.user_id
    if st.session_state.get("page_nav") == "Segment outfit":
        st.session_state.page_nav = "Outfit segmentation"
    st.sidebar.title("Wardrobe Stylist AI")
    st.sidebar.caption(f"Signed in as {st.session_state.username}")
    page = st.sidebar.radio("Go to", ["Outfits & Pinterest", "Add clothes", "Outfit segmentation", "My wardrobe", "Rate outfits"], key="page_nav")
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    if page == "Outfits & Pinterest":
        recommendations_page(user_id)
    elif page == "Add clothes":
        upload_page(user_id)
    elif page == "Outfit segmentation":
        segment_page(user_id)
    elif page == "My wardrobe":
        wardrobe_page(user_id)
    else:
        rate_outfits_page(user_id)


if __name__ == "__main__":
    main()
