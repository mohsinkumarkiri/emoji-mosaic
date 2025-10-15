# # prototype.py
# import os, json
# from PIL import Image
# import numpy as np
# from skimage import color
# from sklearn.neighbors import NearestNeighbors

# # ---------- Config ----------
# EMOJI_DIR = "emojis"
# CACHE_DIR = "cache"
# TILE_PX = 32      # size of emoji tile (pixel)
# GRID_SIZE = 80    # target size on the longer axis (approx)
# OUT_DIR = "output"

# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)

# # ---------- Helpers ----------
# def load_image_rgba(path):
#     return Image.open(path).convert("RGBA")

# def img_to_unit_rgb(np_img):
#     # input PIL->numpy uint8 HxWx4 or HxWx3 -> float in [0,1]
#     arr = np.array(np_img).astype(np.float32) / 255.0
#     if arr.shape[2] == 4:
#         return arr[..., :3], arr[..., 3]
#     else:
#         return arr, np.ones(arr.shape[:2], dtype=np.float32)

# def average_color_rgb(img_pil):
#     rgb, alpha = img_to_unit_rgb(img_pil)
#     h,w,_ = rgb.shape
#     alpha = alpha[..., None]
#     weighted = rgb * alpha
#     total = alpha.sum()
#     if total == 0:
#         return np.array([1.0,1.0,1.0])  # fallback white
#     avg = weighted.sum(axis=(0,1)) / total
#     return avg

# def rgb_to_lab_vector(rgb):
#     # rgb: (...,3) in [0..1]
#     lab = color.rgb2lab(rgb.reshape(1,1,3)).reshape(3)
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

#     # else, build
#     emoji_files = []
#     lab_list = []
#     for fname in sorted(os.listdir(EMOJI_DIR)):
#         if not fname.lower().endswith(('.png','.jpg','.jpeg')):
#             continue
#         path = os.path.join(EMOJI_DIR, fname)
#         im = load_image_rgba(path)
#         thumb = im.resize((tile_px, tile_px), Image.LANCZOS)
#         thumb_path = os.path.join(thumbs_dir, fname)
#         thumb.save(thumb_path)
#         avg_rgb = average_color_rgb(thumb)  # [r,g,b] floats
#         lab = rgb_to_lab_vector(avg_rgb)
#         lab_list.append(lab)
#         emoji_files.append(thumb_path)

#     emoji_lab = np.stack(lab_list, axis=0)  # N x 3
#     np.save(sig_path, emoji_lab)
#     with open(names_path, "w") as f:
#         json.dump(emoji_files, f)
#     return emoji_files, emoji_lab

# # ---------- Step 2-6: Build mosaic ----------
# def build_mosaic(input_path, out_path, grid_size=GRID_SIZE, tile_px=TILE_PX, use_dither=False):
#     emoji_files, emoji_lab = build_or_load_emoji_cache(tile_px=tile_px)

#     # Nearest neighbor index on emoji_lab
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

#     # downsample to small image with bilinear
#     small = src.resize((grid_w, grid_h), Image.BILINEAR)
#     small_rgb = np.array(small).astype(np.float32) / 255.0  # H x W x 3

#     # optional dithering would go here (not implemented for brevity)

#     # convert each pixel to lab and find nearest emoji index
#     out_w = grid_w * tile_px
#     out_h = grid_h * tile_px
#     output = Image.new("RGBA", (out_w, out_h), (255,255,255,255))

#     for y in range(grid_h):
#         for x in range(grid_w):
#             rgb = small_rgb[y, x]
#             lab = rgb_to_lab_vector(rgb)
#             dist, idx = nn.kneighbors(lab.reshape(1,-1), return_distance=True)
#             i = int(idx[0,0])
#             emoji_thumb_path = emoji_files[i]
#             thumb = Image.open(emoji_thumb_path).convert("RGBA")
#             output.paste(thumb, (x*tile_px, y*tile_px), thumb)

#     output.save(out_path)
#     print("Saved:", out_path)

# # ---------- CLI ----------
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("input")
#     p.add_argument("--tile", type=int, default=TILE_PX)
#     p.add_argument("--grid", type=int, default=GRID_SIZE)
#     p.add_argument("--out", default=os.path.join(OUT_DIR, "mosaic.png"))
#     args = p.parse_args()
#     build_mosaic(args.input, args.out, grid_size=args.grid, tile_px=args.tile)


import os, json
from PIL import Image
import numpy as np
from skimage import color
from sklearn.neighbors import NearestNeighbors

# ---------- Config ----------
EMOJI_DIR = "emojis"
CACHE_DIR = "cache"
OUT_DIR = "output"
INPUT_IMG = "input/test-4.png"  # 👈 specify input image path here
OUTPUT_IMG = "output/t4-16-400.png"  # 👈 output path

TILE_PX = 16      # 👈 emoji tile size (in pixels)
GRID_SIZE = 400  # 👈 number of tiles along longer image side

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


# ---------- Step 1: Preprocess emoji set ----------
def build_or_load_emoji_cache(tile_px=TILE_PX):
    sig_path = os.path.join(CACHE_DIR, "emoji_signatures.npy")
    names_path = os.path.join(CACHE_DIR, "emoji_names.json")
    thumbs_dir = os.path.join(CACHE_DIR, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    if os.path.exists(sig_path) and os.path.exists(names_path):
        emoji_lab = np.load(sig_path)
        with open(names_path, "r") as f:
            emoji_files = json.load(f)
        print(f"[CACHE] Loaded {len(emoji_files)} emojis from cache.")
        return emoji_files, emoji_lab

    emoji_files, lab_list = [], []
    print("[CACHE] Building emoji cache...")
    for fname in sorted(os.listdir(EMOJI_DIR)):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
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


# ---------- Step 2-6: Build mosaic ----------
def build_mosaic(input_path, out_path, grid_size=GRID_SIZE, tile_px=TILE_PX):
    emoji_files, emoji_lab = build_or_load_emoji_cache(tile_px=tile_px)

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
    small_rgb = np.array(small).astype(np.float32) / 255.0

    out_w = grid_w * tile_px
    out_h = grid_h * tile_px
    output = Image.new("RGBA", (out_w, out_h), (255,255,255,255))

    print(f"[INFO] Building mosaic {grid_w}x{grid_h} tiles ({out_w}x{out_h}px) ...")

    for y in range(grid_h):
        for x in range(grid_w):
            rgb = small_rgb[y, x]
            lab = rgb_to_lab_vector(rgb)
            _, idx = nn.kneighbors(lab.reshape(1,-1))
            i = int(idx[0,0])
            thumb = Image.open(emoji_files[i]).convert("RGBA")
            output.paste(thumb, (x*tile_px, y*tile_px), thumb)

    output.save(out_path)
    print(f"[DONE] Saved mosaic to {out_path}")


# ---------- Run directly ----------
if __name__ == "__main__":
    build_mosaic(INPUT_IMG, OUTPUT_IMG, GRID_SIZE, TILE_PX)
