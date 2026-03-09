"""Generate an Open Graph image for LinkedIn/social sharing."""

from PIL import Image, ImageDraw, ImageFont
import os

# Output settings
WIDTH, HEIGHT = 1200, 627
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "og-preview.png")

# Colors (light theme, consistent with site)
BG_COLOR = "#ffffff"
NAME_COLOR = "#222222"
TITLE_COLOR = "#555555"
TAGLINE_COLOR = "#777777"
URL_COLOR = "#999999"
ACCENT_COLOR = "#4a86c8"  # subtle blue accent line
BORDER_COLOR = "#e0e0e0"


def get_font(size, bold=False):
    """Try to load a clean sans-serif font, fall back to default."""
    font_candidates = [
        # Windows fonts
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",  # bold
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    bold_candidates = [f for f in font_candidates if "b." in f or "bd." in f or "Bold" in f]
    regular_candidates = [f for f in font_candidates if f not in bold_candidates]

    candidates = bold_candidates if bold else regular_candidates
    # Also try the other set as fallback
    candidates = candidates + (regular_candidates if bold else bold_candidates)

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def generate():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Subtle border
    draw.rectangle(
        [0, 0, WIDTH - 1, HEIGHT - 1],
        outline=BORDER_COLOR, width=1
    )

    # Blue accent bar at top
    draw.rectangle([0, 0, WIDTH, 5], fill=ACCENT_COLOR)

    # Fonts
    name_font = get_font(54, bold=True)
    title_font = get_font(30)
    tagline_font = get_font(22)
    url_font = get_font(20)

    # Layout: vertically centered content
    center_x = WIDTH // 2

    # Name
    name_text = "Danny Tabach"
    name_bbox = draw.textbbox((0, 0), name_text, font=name_font)
    name_w = name_bbox[2] - name_bbox[0]
    name_y = 170
    draw.text(
        ((WIDTH - name_w) // 2, name_y),
        name_text, fill=NAME_COLOR, font=name_font
    )

    # Title
    title_text = "Data Scientist at Chase"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_y = name_y + 75
    draw.text(
        ((WIDTH - title_w) // 2, title_y),
        title_text, fill=TITLE_COLOR, font=title_font
    )

    # Divider line
    div_y = title_y + 55
    div_half = 120
    draw.line(
        [(center_x - div_half, div_y), (center_x + div_half, div_y)],
        fill=ACCENT_COLOR, width=2
    )

    # Tagline
    tagline_text = "Experiments  |  Optimization  |  ML"
    tag_bbox = draw.textbbox((0, 0), tagline_text, font=tagline_font)
    tag_w = tag_bbox[2] - tag_bbox[0]
    tag_y = div_y + 25
    draw.text(
        ((WIDTH - tag_w) // 2, tag_y),
        tagline_text, fill=TAGLINE_COLOR, font=tagline_font
    )

    # URL at bottom
    url_text = "danieltabach.github.io"
    url_bbox = draw.textbbox((0, 0), url_text, font=url_font)
    url_w = url_bbox[2] - url_bbox[0]
    url_y = HEIGHT - 70
    draw.text(
        ((WIDTH - url_w) // 2, url_y),
        url_text, fill=URL_COLOR, font=url_font
    )

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH, "PNG", quality=95)
    print(f"Saved OG image: {os.path.abspath(OUTPUT_PATH)}")
    print(f"Dimensions: {WIDTH}x{HEIGHT}")


if __name__ == "__main__":
    generate()
