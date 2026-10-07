"""Capture the real upload flow using a temporary account and drawn garments.

Optional tool: pip install playwright. Requires a local Chrome installation.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "screenshots"
sys.path.insert(0, str(ROOT))


def sample_image(category: str, color: str) -> Image.Image:
    image = Image.new("RGB", (360, 390), "#f4f0eb")
    draw = ImageDraw.Draw(image)
    ink = {"white": "#ffffff", "navy": "#203959", "blue": "#5c8eaf",
           "black": "#25272b", "beige": "#cbb99d", "brown": "#765744"}[color]
    if category == "top":
        shape = [(110, 65), (150, 40), (210, 40), (250, 65), (285, 145),
                 (245, 165), (230, 125), (230, 320), (130, 320), (130, 125),
                 (115, 165), (75, 145)]
        draw.polygon(shape, fill=ink, outline="#555555", width=3)
    elif category == "bottom":
        shape = [(105, 55), (255, 55), (235, 325), (185, 325), (180, 145),
                 (175, 325), (125, 325)]
        draw.polygon(shape, fill=ink, outline="#555555", width=3)
    else:
        draw.rounded_rectangle((45, 205, 310, 275), radius=28, fill=ink, outline="#555555", width=3)
        draw.polygon([(55, 240), (285, 240), (325, 273), (315, 290), (65, 290)],
                     fill="#fdfdfd", outline="#555555", width=3)
    draw.text((20, 350), f"DEMO {color.upper()} {category.upper()}", fill="#303030")
    return image


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wardrobe-stylist-demo-", ignore_cleanup_errors=True) as directory:
        demo_dir = Path(directory)
        from fashion_ai import db

        db.DATA_DIR = demo_dir
        db.DB_PATH = demo_dir / "wardrobe.db"
        db.UPLOAD_DIR = demo_dir / "uploads"
        db.init_db()
        user_id = db.register_user("demo_user", "demo_password_123")
        samples = []
        for category, colors in (("top", ("white", "navy", "blue")),
                                 ("bottom", ("black", "beige", "navy")),
                                 ("shoes", ("white", "brown"))):
            for color in colors:
                path = demo_dir / f"{color}-{category}.png"
                sample_image(category, color).save(path)
                samples.append(str(path))

        port = free_port()
        env = os.environ.copy()
        env["FASHION_DATA_DIR"] = str(demo_dir)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true",
             f"--server.port={port}", "--server.address=127.0.0.1"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        try:
            health = f"http://127.0.0.1:{port}/_stcore/health"
            for _ in range(100):
                try:
                    if urllib.request.urlopen(health, timeout=1).status == 200:
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Demo app did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                page.set_default_timeout(120000)
                page.goto(f"http://127.0.0.1:{port}")
                page.get_by_text("Create account", exact=True).first.wait_for()
                page.screenshot(path=str(OUTPUT / "01-sign-in.png"), full_page=True)
                page.get_by_text("Sign in", exact=True).first.click()
                page.get_by_label("Username", exact=True).fill("demo_user")
                page.get_by_label("Password", exact=True).fill("demo_password_123")
                page.get_by_role("button", name="Sign in").click()
                page.get_by_text("Add clothes", exact=True).first.click()
                page.locator('input[type="file"]').set_input_files(samples)
                page.get_by_role("button", name="Save detected clothes").wait_for()
                page.wait_for_function("document.body.innerText.split('Detected item:').length >= 9")
                page.screenshot(path=str(OUTPUT / "04-upload-classification.png"), full_page=True)
                page.get_by_role("button", name="Save detected clothes").click()
                page.get_by_text("Saved 8 item(s) to your wardrobe.").wait_for()
                # The demo drawings are outside the classifier's product-photo domain.
                # Correct two known labels as a user would in My wardrobe.
                for item in db.list_items(user_id):
                    if item["name"] in ("White dress", "Brown bottom"):
                        db.update_item(user_id, item["id"], name=f"Demo {item['color']} shoes",
                                       category="shoes", color=item["color"], style=item["style"], occasion=item["occasion"])
                page.get_by_text("Outfits & Pinterest", exact=True).first.click()
                page.get_by_text("What to add next").wait_for()
                page.wait_for_function("document.querySelectorAll('[data-testid=\"stImage\"] img').length >= 9")
                page.get_by_role("heading", name="Outfits and Pinterest ideas").scroll_into_view_if_needed()
                page.locator('[data-testid="stSidebar"]').get_by_text("Wardrobe Stylist AI").scroll_into_view_if_needed()
                page.screenshot(path=str(OUTPUT / "02-outfits.png"), full_page=True)
                page.get_by_text("Rate outfits", exact=True).first.click()
                page.get_by_text("How would you rate outfit 1?").wait_for()
                page.wait_for_function("document.querySelectorAll('[data-testid=\"stImage\"] img').length >= 15")
                page.get_by_role("heading", name="Rate outfit ideas").scroll_into_view_if_needed()
                page.locator('[data-testid="stSidebar"]').get_by_text("Wardrobe Stylist AI").scroll_into_view_if_needed()
                page.screenshot(path=str(OUTPUT / "03-rate-outfits.png"), full_page=True)
                browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(f"Saved screenshots to {OUTPUT}")


if __name__ == "__main__":
    main()
