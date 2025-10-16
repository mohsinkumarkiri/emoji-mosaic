## Kyle Chen Style Script working (incl. dithering) - work best for tile=16, GRID=200,300,400, >500 (overkill)


# import os, json
# from PIL import Image
# import numpy as np
# from skimage import color
# from sklearn.neighbors import NearestNeighbors

# # ---------- Config ----------
# EMOJI_DIR = "emojis"
# CACHE_DIR = "cache"
# OUT_DIR = "output"
# INPUT_IMG = "input/test-4.png"     # 👈 specify input image
# OUTPUT_IMG = "output/t4-v3-16-500-kyleChenStyle.png"

# TILE_PX = 16                      # emoji tile size (px)
# GRID_SIZE = 500                    # number of emojis across the longer side

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
#     h, w, _ = rgb.shape
#     alpha = alpha[..., None]
#     weighted = rgb * alpha
#     total = alpha.sum()
#     if total == 0:
#         return np.array([1.0, 1.0, 1.0])
#     avg = weighted.sum(axis=(0, 1)) / total
#     return avg

# def rgb_to_lab_vector(rgb):
#     lab = color.rgb2lab(rgb.reshape(1,1,3)).reshape(3)
#     return lab


# # ---------- Emoji Cache ----------
# def build_or_load_emoji_cache(tile_px=TILE_PX):
#     sig_path = os.path.join(CACHE_DIR, "emoji_signatures.npy")
#     names_path = os.path.join(CACHE_DIR, "emoji_names.json")
#     thumbs_dir = os.path.join(CACHE_DIR, "thumbs")
#     os.makedirs(thumbs_dir, exist_ok=True)

#     if os.path.exists(sig_path) and os.path.exists(names_path):
#         emoji_lab = np.load(sig_path)
#         with open(names_path, "r") as f:
#             emoji_files = json.load(f)
#         print(f"[CACHE] Loaded {len(emoji_files)} emojis.")
#         return emoji_files, emoji_lab

#     emoji_files, lab_list = [], []
#     for fname in sorted(os.listdir(EMOJI_DIR)):
#         if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
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
#     print(f"[CACHE] Built {len(emoji_files)} emoji thumbnails.")
#     return emoji_files, emoji_lab


# # ---------- Pixel-level Dithering (Kyle Chen style) ----------
# def dither_and_match_grid(rgb_small, emoji_lab, nn):
#     h, w, _ = rgb_small.shape
#     lab_small = color.rgb2lab(rgb_small)
#     lab_dithered = lab_small.copy()

#     matched_idx = np.zeros((h, w), dtype=int)

#     for y in range(h):
#         for x in range(w):
#             orig_lab = lab_dithered[y, x]
#             _, idx = nn.kneighbors(orig_lab.reshape(1, -1))
#             i = int(idx[0, 0])
#             matched_idx[y, x] = i
#             new_lab = emoji_lab[i]

#             # Error (in LAB space)
#             error = orig_lab - new_lab

#             # Floyd–Steinberg diffusion
#             if x + 1 < w:
#                 lab_dithered[y, x + 1] += error * (7/16)
#             if y + 1 < h:
#                 if x > 0:
#                     lab_dithered[y + 1, x - 1] += error * (3/16)
#                 lab_dithered[y + 1, x] += error * (5/16)
#                 if x + 1 < w:
#                     lab_dithered[y + 1, x + 1] += error * (1/16)

#     return matched_idx


# # ---------- Build Mosaic ----------
# def build_mosaic(input_path, out_path, grid_size=GRID_SIZE, tile_px=TILE_PX):
#     emoji_files, emoji_lab = build_or_load_emoji_cache(tile_px)
#     nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(emoji_lab)

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
#     rgb_small = np.array(small).astype(np.float32) / 255.0

#     print(f"[INFO] Performing pixel-level dithering on {grid_w}x{grid_h} grid...")
#     matched_idx = dither_and_match_grid(rgb_small, emoji_lab, nn)

#     out_w, out_h = grid_w * tile_px, grid_h * tile_px
#     output = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))

#     for y in range(grid_h):
#         for x in range(grid_w):
#             emoji_path = emoji_files[matched_idx[y, x]]
#             thumb = Image.open(emoji_path).convert("RGBA")
#             output.paste(thumb, (x * tile_px, y * tile_px), thumb)

#     output.save(out_path)
#     print(f"[DONE] Saved dithered mosaic to {out_path}")


# # ---------- Run ----------
# if __name__ == "__main__":
#     build_mosaic(INPUT_IMG, OUTPUT_IMG, GRID_SIZE, TILE_PX)



## Kyle Chen Style Script working (incl. dithering) - work best for tile=16, GRID=200,300,400, >500 (overkill)
## Shadow diffusion fixed in below code



import os, json
from PIL import Image
import numpy as np
from skimage import color
from sklearn.neighbors import NearestNeighbors

# ---------- Config ----------
EMOJI_DIR = "emojis"
CACHE_DIR = "cache"
OUT_DIR = "output"
INPUT_IMG = "input/test-4.png"     # 👈 specify input image
OUTPUT_IMG = "output/t4-v3-sdw-12-400.png"

TILE_PX = 12                       # emoji tile size (px)
GRID_SIZE = 400                    # number of emojis across the longer side

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Helpers ----------
def load_image_rgba(path):
    return Image.open(path).convert("RGBA")

def img_to_unit_rgb(np_img):
    arr = np.array(np_img).astype(np.float32) / 255.0
    if arr.shape[2] == 4:
        return arr[..., :3], arr[..., 3]
    else:
        return arr, np.ones(arr.shape[:2], dtype=np.float32)

def average_color_rgb(img_pil):
    rgb, alpha = img_to_unit_rgb(img_pil)
    h, w, _ = rgb.shape
    alpha = alpha[..., None]
    weighted = rgb * alpha
    total = alpha.sum()
    if total == 0:
        return np.array([1.0, 1.0, 1.0])
    avg = weighted.sum(axis=(0, 1)) / total
    return avg

def rgb_to_lab_vector(rgb):
    lab = color.rgb2lab(rgb.reshape(1,1,3)).reshape(3)
    return lab


# ---------- Emoji Cache ----------
def build_or_load_emoji_cache(tile_px=TILE_PX):
    sig_path = os.path.join(CACHE_DIR, "emoji_signatures.npy")
    names_path = os.path.join(CACHE_DIR, "emoji_names.json")
    thumbs_dir = os.path.join(CACHE_DIR, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    if os.path.exists(sig_path) and os.path.exists(names_path):
        emoji_lab = np.load(sig_path)
        with open(names_path, "r") as f:
            emoji_files = json.load(f)
        print(f"[CACHE] Loaded {len(emoji_files)} emojis.")
        return emoji_files, emoji_lab

    emoji_files, lab_list = [], []
    for fname in sorted(os.listdir(EMOJI_DIR)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(EMOJI_DIR, fname)
        im = load_image_rgba(path)
        thumb = im.resize((tile_px, tile_px), Image.LANCZOS)
        thumb_path = os.path.join(thumbs_dir, fname)
        thumb.save(thumb_path)
        avg_rgb = average_color_rgb(thumb)
        lab = rgb_to_lab_vector(avg_rgb)
        lab_list.append(lab)
        emoji_files.append(thumb_path)

    emoji_lab = np.stack(lab_list, axis=0)
    np.save(sig_path, emoji_lab)
    with open(names_path, "w") as f:
        json.dump(emoji_files, f)
    print(f"[CACHE] Built {len(emoji_files)} emoji thumbnails.")
    return emoji_files, emoji_lab


# ---------- Pixel-level Dithering (Kyle Chen style) ----------
def dither_and_match_grid(rgb_small, emoji_lab, nn):
    h, w, _ = rgb_small.shape
    lab_small = color.rgb2lab(rgb_small)
    lab_dithered = lab_small.copy()

    matched_idx = np.zeros((h, w), dtype=int)

    for y in range(h):
        for x in range(w):
            orig_lab = lab_dithered[y, x]
            _, idx = nn.kneighbors(orig_lab.reshape(1, -1))
            i = int(idx[0, 0])
            matched_idx[y, x] = i
            new_lab = emoji_lab[i]

            # Error (in LAB space)
            error = orig_lab - new_lab

            # Floyd–Steinberg diffusion
            if x + 1 < w:
                lab_dithered[y, x + 1] += error * (7/16)
            if y + 1 < h:
                if x > 0:
                    lab_dithered[y + 1, x - 1] += error * (3/16)
                lab_dithered[y + 1, x] += error * (5/16)
                if x + 1 < w:
                    lab_dithered[y + 1, x + 1] += error * (1/16)

    return matched_idx

# ---------- Pixel-level Dithering (Balanced Shadows) ----------
def perceptual_distance(a, b):
    # Weighted distance: prioritize L channel in darker zones
    L1, a1, b1 = a
    L2, a2, b2 = b
    lightness_weight = 1.5 if L1 < 35 else 1.0  # boost L accuracy in dark zones
    return ((lightness_weight * (L1 - L2)) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def dither_and_match_grid_balanced(rgb_small, emoji_lab, emoji_lab_L, nn):
    h, w, _ = rgb_small.shape
    # Apply mild gamma correction for better shadow handling
    rgb_small = np.clip(rgb_small, 0, 1) ** (1 / 2.2)
    lab_small = color.rgb2lab(rgb_small)
    lab_dithered = lab_small.copy()
    matched_idx = np.zeros((h, w), dtype=int)

    for y in range(h):
        for x in range(w):
            orig_lab = lab_dithered[y, x]
            # Custom nearest search prioritizing dark tone match
            L_target = orig_lab[0]
            if L_target < 40:
                # Focus search among darker emojis
                dark_mask = emoji_lab_L < 60
                subset = emoji_lab[dark_mask]
                if len(subset) > 0:
                    dists = np.linalg.norm(subset - orig_lab, axis=1)
                    best_i_sub = np.argmin(dists)
                    i = np.arange(len(emoji_lab))[dark_mask][best_i_sub]
                else:
                    _, idx = nn.kneighbors(orig_lab.reshape(1, -1))
                    i = int(idx[0, 0])
            else:
                _, idx = nn.kneighbors(orig_lab.reshape(1, -1))
                i = int(idx[0, 0])

            matched_idx[y, x] = i
            new_lab = emoji_lab[i]

            # Error diffusion
            error = orig_lab - new_lab
            if x + 1 < w:
                lab_dithered[y, x + 1] += error * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    lab_dithered[y + 1, x - 1] += error * (3 / 16)
                lab_dithered[y + 1, x] += error * (5 / 16)
                if x + 1 < w:
                    lab_dithered[y + 1, x + 1] += error * (1 / 16)

    return matched_idx


# ---------- Build Mosaic ----------
def build_mosaic(input_path, out_path, grid_size=GRID_SIZE, tile_px=TILE_PX):
    emoji_files, emoji_lab = build_or_load_emoji_cache(tile_px)
    emoji_lab_L = emoji_lab[:, 0]  # extract L-channel
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(emoji_lab)

    src = Image.open(input_path).convert("RGB")
    src_w, src_h = src.size
    aspect = src_w / src_h

    if src_w >= src_h:
        grid_w = grid_size
        grid_h = max(1, int(round(grid_size / aspect)))
    else:
        grid_h = grid_size
        grid_w = max(1, int(round(grid_size * aspect)))

    small = src.resize((grid_w, grid_h), Image.BILINEAR)
    rgb_small = np.array(small).astype(np.float32) / 255.0

    print(f"[INFO] Performing shadow-balanced dithering on {grid_w}x{grid_h} grid...")
    matched_idx = dither_and_match_grid_balanced(rgb_small, emoji_lab, emoji_lab_L, nn)

    out_w, out_h = grid_w * tile_px, grid_h * tile_px
    output = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))

    for y in range(grid_h):
        for x in range(grid_w):
            emoji_path = emoji_files[matched_idx[y, x]]
            thumb = Image.open(emoji_path).convert("RGBA")
            output.paste(thumb, (x * tile_px, y * tile_px), thumb)

    output.save(out_path)
    print(f"[DONE] Saved balanced mosaic to {out_path}")



# ---------- Run ----------
if __name__ == "__main__":
    build_mosaic(INPUT_IMG, OUTPUT_IMG, GRID_SIZE, TILE_PX)
