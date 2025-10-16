# import os, json
# from PIL import Image
# import numpy as np
# from skimage import color
# from sklearn.neighbors import NearestNeighbors

# # ---------- Config ----------
# EMOJI_DIR = "emojis"
# CACHE_DIR = "cache"
# OUT_DIR = "output"

# TILE_PX = 16       # size of emoji tile (pixel)
# GRID_SIZE = 200     # grid resolution (longer axis)
# USE_DITHER = True  # enable or disable dithering

# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)

# # ---------- Helpers ----------
# def load_image_rgba(path):
#     return Image.open(path).convert("RGBA")

# def img_to_unit_rgb(np_img):
#     arr = np.array(np_img).astype(np.float32) / 255.0
#     if arr.shape[2] == 4:
#         return arr[..., :3], arr[..., 3]
#     else:
#         return arr, np.ones(arr.shape[:2], dtype=np.float32)

# def average_color_rgb(img_pil):
#     rgb, alpha = img_to_unit_rgb(img_pil)
#     alpha = alpha[..., None]
#     weighted = rgb * alpha
#     total = alpha.sum()
#     if total == 0:
#         return np.array([1.0, 1.0, 1.0])
#     avg = weighted.sum(axis=(0, 1)) / total
#     return avg

# def rgb_to_lab_vector(rgb):
#     lab = color.rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)
#     return lab

# # ---------- Step 1: Preprocess emoji set ----------
# def build_or_load_emoji_cache(tile_px=TILE_PX):
#     sig_path = os.path.join(CACHE_DIR, "emoji_signatures.npy")
#     names_path = os.path.join(CACHE_DIR, "emoji_names.json")
#     thumbs_dir = os.path.join(CACHE_DIR, "thumbs")
#     os.makedirs(thumbs_dir, exist_ok=True)

#     if os.path.exists(sig_path) and os.path.exists(names_path):
#         emoji_lab = np.load(sig_path)
#         with open(names_path, "r") as f:
#             emoji_files = json.load(f)
#         return emoji_files, emoji_lab

#     emoji_files = []
#     lab_list = []
#     for fname in sorted(os.listdir(EMOJI_DIR)):
#         if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue
#         path = os.path.join(EMOJI_DIR, fname)
#         im = load_image_rgba(path)
#         thumb = im.resize((tile_px, tile_px), Image.LANCZOS)
#         thumb_path = os.path.join(thumbs_dir, fname)
#         thumb.save(thumb_path)
#         avg_rgb = average_color_rgb(thumb)
#         lab = rgb_to_lab_vector(avg_rgb)
#         lab_list.append(lab)
#         emoji_files.append(thumb_path)

#     emoji_lab = np.stack(lab_list, axis=0)
#     np.save(sig_path, emoji_lab)
#     with open(names_path, "w") as f:
#         json.dump(emoji_files, f)
#     return emoji_files, emoji_lab

# # ---------- Step 2: Dithering Helper ----------
# def floyd_steinberg_dither(img_rgb, nn, emoji_lab, emoji_files):
#     """Applies basic Floyd–Steinberg dithering in LAB space."""
#     h, w, _ = img_rgb.shape
#     out_indices = np.zeros((h, w), dtype=int)
#     img_lab = color.rgb2lab(img_rgb)

#     for y in range(h):
#         for x in range(w):
#             lab_pixel = img_lab[y, x]
#             dist, idx = nn.kneighbors(lab_pixel.reshape(1, -1), return_distance=True)
#             nearest_i = int(idx[0, 0])
#             out_indices[y, x] = nearest_i

#             # convert nearest emoji color back to RGB to compute error
#             emoji_lab_color = emoji_lab[nearest_i]
#             emoji_rgb = color.lab2rgb(emoji_lab_color.reshape(1, 1, 3)).reshape(3)
#             err = img_rgb[y, x] - emoji_rgb

#             # distribute error (Floyd–Steinberg kernel)
#             if x + 1 < w:
#                 img_rgb[y, x + 1] += err * (7 / 16)
#             if y + 1 < h and x > 0:
#                 img_rgb[y + 1, x - 1] += err * (3 / 16)
#             if y + 1 < h:
#                 img_rgb[y + 1, x] += err * (5 / 16)
#             if y + 1 < h and x + 1 < w:
#                 img_rgb[y + 1, x + 1] += err * (1 / 16)

#     # clamp back to valid range [0,1]
#     img_rgb = np.clip(img_rgb, 0, 1)
#     return out_indices

# # ---------- Step 3: Build Mosaic ----------
# def build_mosaic(input_path, out_path, grid_size=GRID_SIZE, tile_px=TILE_PX, use_dither=USE_DITHER):
#     emoji_files, emoji_lab = build_or_load_emoji_cache(tile_px=tile_px)
#     nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(emoji_lab)

#     src = Image.open(input_path).convert("RGB")
#     src_w, src_h = src.size
#     aspect = src_w / src_h
#     if src_w >= src_h:
#         grid_w = grid_size
#         grid_h = max(1, int(round(grid_size / aspect)))
#     else:
#         grid_h = grid_size
#         grid_w = max(1, int(round(grid_size * aspect)))

#     small = src.resize((grid_w, grid_h), Image.BILINEAR)
#     small_rgb = np.array(small).astype(np.float32) / 255.0

#     if use_dither:
#         print("Applying Floyd–Steinberg dithering ...")
#         indices = floyd_steinberg_dither(small_rgb.copy(), nn, emoji_lab, emoji_files)
#     else:
#         print("No dithering applied.")
#         h, w, _ = small_rgb.shape
#         indices = np.zeros((h, w), dtype=int)
#         for y in range(h):
#             for x in range(w):
#                 lab = rgb_to_lab_vector(small_rgb[y, x])
#                 _, idx = nn.kneighbors(lab.reshape(1, -1), return_distance=True)
#                 indices[y, x] = int(idx[0, 0])

#     out_w = grid_w * tile_px
#     out_h = grid_h * tile_px
#     output = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))

#     for y in range(grid_h):
#         for x in range(grid_w):
#             emoji_thumb_path = emoji_files[indices[y, x]]
#             thumb = Image.open(emoji_thumb_path).convert("RGBA")
#             output.paste(thumb, (x * tile_px, y * tile_px), thumb)

#     output.save(out_path)
#     print(f"✅ Saved mosaic to {out_path}")

# # ---------- Run ----------
# if __name__ == "__main__":
#     INPUT_PATH = "input/test-1.png"
#     OUTPUT_PATH = "output/t1-dithered-16-200.png"
#     build_mosaic(INPUT_PATH, OUTPUT_PATH)



import os
import json
import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances

# === CONFIG ===
INPUT_IMAGE = "input/test-2.png"        # Input image path
EMOJI_DIR = "emojis"                    # Folder containing emoji PNGs
OUTPUT_IMAGE = "output/t2-dithered-v2-200-200.png"
CACHE_FILE = "cache/emoji_cache.json"

GRID_SIZE = 200   # Number of tiles horizontally
TILE_SIZE = 200   # Emoji size (px)

# === STEP 1: LOAD EMOJI COLOR DATA ===
def compute_average_color(img):
    """Compute average RGB color, ignoring transparent pixels."""
    arr = np.array(img.convert("RGBA"))
    mask = arr[..., 3] > 0
    if np.any(mask):
        rgb = arr[..., :3][mask]
        return rgb.mean(axis=0)
    else:
        return np.array([255, 255, 255])

def load_emojis(emoji_dir):
    """Load emoji thumbnails and compute their average colors."""
    cache = []
    for filename in os.listdir(emoji_dir):
        if filename.lower().endswith((".png", ".jpg")):
            path = os.path.join(emoji_dir, filename)
            img = Image.open(path).convert("RGBA").resize((TILE_SIZE, TILE_SIZE))
            avg_color = compute_average_color(img)
            cache.append({"path": path, "color": avg_color.tolist()})
    return cache

# Cache system to speed up subsequent runs
def load_or_create_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    else:
        cache = load_emojis(EMOJI_DIR)
        os.makedirs("cache", exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    return cache

# === STEP 2: MAIN MOSAIC CREATION (WITH TILE-LEVEL DITHERING) ===
def build_mosaic_with_dithering(input_path, emoji_cache):
    base = Image.open(input_path).convert("RGB")
    width, height = base.size

    tile_w = width // GRID_SIZE
    tile_h = int(tile_w * (TILE_SIZE / TILE_SIZE))  # maintain square-ish tiles
    grid_x = GRID_SIZE
    grid_y = height // tile_h

    # Prepare output canvas
    mosaic = Image.new("RGB", (grid_x * TILE_SIZE, grid_y * TILE_SIZE))

    # Prepare emoji palette colors
    emoji_colors = np.array([e["color"] for e in emoji_cache])
    emoji_paths = [e["path"] for e in emoji_cache]

    # Prepare Floyd–Steinberg diffusion matrix (per tile)
    error_buffer = np.zeros((grid_y + 1, grid_x + 1, 3), dtype=float)

    for y in range(grid_y):
        for x in range(grid_x):
            # Extract block region
            region = base.crop((x * tile_w, y * tile_h, (x + 1) * tile_w, (y + 1) * tile_h))
            avg = np.array(region.resize((1, 1)).getpixel((0, 0)), dtype=float)

            # Add propagated error
            avg = np.clip(avg + error_buffer[y, x], 0, 255)

            # Find closest emoji color
            dist = np.linalg.norm(emoji_colors - avg, axis=1)
            idx = np.argmin(dist)
            chosen_emoji = Image.open(emoji_paths[idx]).resize((TILE_SIZE, TILE_SIZE))

            # Place emoji on mosaic
            mosaic.paste(chosen_emoji, (x * TILE_SIZE, y * TILE_SIZE))

            # Compute quantization error (RGB difference)
            chosen_color = emoji_colors[idx]
            quant_error = avg - chosen_color

            # Distribute error to neighboring tiles (tile-wise Floyd–Steinberg)
            if x + 1 < grid_x:
                error_buffer[y, x + 1] += quant_error * (7 / 16)
            if y + 1 < grid_y:
                if x > 0:
                    error_buffer[y + 1, x - 1] += quant_error * (3 / 16)
                error_buffer[y + 1, x] += quant_error * (5 / 16)
                if x + 1 < grid_x:
                    error_buffer[y + 1, x + 1] += quant_error * (1 / 16)

    return mosaic


# === STEP 3: MAIN ENTRY ===
def main():
    print("🔹 Loading emoji cache...")
    emoji_cache = load_or_create_cache()
    print(f"✅ Loaded {len(emoji_cache)} emojis")

    print("🎨 Building mosaic with tile-level dithering...")
    mosaic = build_mosaic_with_dithering(INPUT_IMAGE, emoji_cache)
    os.makedirs("output", exist_ok=True)
    mosaic.save(OUTPUT_IMAGE)
    print(f"✅ Dithered mosaic saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
