import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage import color
from PIL import Image
import os
import json

# ---------- Config ----------
EMOJI_DIR = "emojis"
CACHE_DIR = "cache"

INIT_TILE_PX = 12
INIT_GRID_SIZE = 40  # up to 1000

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- Helpers ----------
def load_image_rgba(path):
    return Image.open(path).convert("RGBA")

def average_color_rgb(img_pil):
    arr = np.array(img_pil).astype(np.float32) / 255.0
    if arr.shape[2] == 4:
        rgb, alpha = arr[..., :3], arr[..., 3]
    else:
        rgb, alpha = arr, np.ones(arr.shape[:2], dtype=np.float32)
    weighted = rgb * alpha[..., None]
    total = alpha.sum()
    if total == 0: return np.array([1.0, 1.0, 1.0])
    return weighted.sum(axis=(0,1)) / total

def rgb_to_lab_vector(rgb):
    return color.rgb2lab(rgb.reshape(1,1,3)).reshape(3)

# ---------- Emoji Cache ----------
def build_or_load_emoji_cache(tile_px):
    sig_path = os.path.join(CACHE_DIR, "emoji_signatures.npy")
    names_path = os.path.join(CACHE_DIR, "emoji_names.json")
    thumbs_dir = os.path.join(CACHE_DIR, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    if os.path.exists(sig_path) and os.path.exists(names_path):
        emoji_lab = np.load(sig_path)
        with open(names_path, "r") as f:
            emoji_files = json.load(f)
        print(f"[CACHE] Loaded {len(emoji_files)} emojis.")
    else:
        emoji_files, lab_list = [], []
        for fname in sorted(os.listdir(EMOJI_DIR)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(EMOJI_DIR, fname)
            im = load_image_rgba(path)
            thumb = im.resize((tile_px, tile_px), Image.LANCZOS)
            thumb_path = os.path.join(thumbs_dir, fname)
            thumb.save(thumb_path)
            lab = rgb_to_lab_vector(average_color_rgb(thumb))
            lab_list.append(lab)
            emoji_files.append(thumb_path)
        emoji_lab = np.stack(lab_list, axis=0)
        np.save(sig_path, emoji_lab)
        with open(names_path, "w") as f:
            json.dump(emoji_files, f)
        print(f"[CACHE] Built {len(emoji_files)} emoji thumbnails.")

    # Preload emoji tiles as numpy arrays (original size)
    emoji_imgs_full = [np.array(load_image_rgba(path)) for path in emoji_files]

    emoji_lab_L = emoji_lab[:,0]
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(emoji_lab)
    return emoji_files, emoji_lab, emoji_imgs_full, emoji_lab_L, nn

# ---------- Vectorized Matching ----------
def vectorized_match_grid(rgb_small, emoji_lab, emoji_lab_L, nn):
    h, w, _ = rgb_small.shape
    rgb_small = np.clip(rgb_small, 0, 1) ** (1/2.2)
    lab_small = color.rgb2lab(rgb_small).reshape(-1, 3)

    matched_idx = np.zeros(lab_small.shape[0], dtype=int)

    # Dark pixels
    dark_mask = lab_small[:,0] < 40
    if np.any(dark_mask):
        dark_pixels = lab_small[dark_mask]
        dark_emoji_mask = emoji_lab_L < 60
        dark_subset = emoji_lab[dark_emoji_mask]
        if len(dark_subset) > 0:
            diff = dark_pixels[:,None,:] - dark_subset[None,:,:]
            dist = np.linalg.norm(diff, axis=2)
            best_idx = np.argmin(dist, axis=1)
            emoji_indices = np.arange(len(emoji_lab))[dark_emoji_mask]
            matched_idx[dark_mask] = emoji_indices[best_idx]
        else:
            _, idx = nn.kneighbors(dark_pixels)
            matched_idx[dark_mask] = idx[:,0]

    # Light pixels
    light_mask = ~dark_mask
    if np.any(light_mask):
        light_pixels = lab_small[light_mask]
        _, idx = nn.kneighbors(light_pixels)
        matched_idx[light_mask] = idx[:,0]

    return matched_idx.reshape(h, w)

# ---------- Trackbar ----------
def nothing(x):
    pass

# ---------- Live Camera Loop ----------
def live_emoji_camera(tile_px=INIT_TILE_PX, grid_size=INIT_GRID_SIZE):
    emoji_files, emoji_lab, emoji_imgs_full, emoji_lab_L, nn = build_or_load_emoji_cache(tile_px)
    # Initialize resized emoji array
    emoji_imgs = [cv2.resize(img, (tile_px, tile_px), interpolation=cv2.INTER_AREA) 
                  for img in emoji_imgs_full]

    # Initialize OpenCV window in full-screen mode
    cv2.namedWindow("Emoji Mosaic Live", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Emoji Mosaic Live", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create trackbars for tile size and grid size
    cv2.createTrackbar("Tile Px", "Emoji Mosaic Live", tile_px, 40, nothing)
    cv2.createTrackbar("Grid Size", "Emoji Mosaic Live", grid_size, 1000, nothing)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("[INFO] Press ESC to exit live feed.")
    last_tile = tile_px

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Read trackbar values
        tile_px = max(1, cv2.getTrackbarPos("Tile Px", "Emoji Mosaic Live"))
        grid_size = max(10, cv2.getTrackbarPos("Grid Size", "Emoji Mosaic Live"))

        # Resize emojis if tile_px changed
        if tile_px != last_tile:
            emoji_imgs = [cv2.resize(img, (tile_px, tile_px), interpolation=cv2.INTER_AREA) 
                          for img in emoji_imgs_full]
            last_tile = tile_px

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        aspect = w / h

        if w >= h:
            grid_w = grid_size
            grid_h = max(1, int(round(grid_size / aspect)))
        else:
            grid_h = grid_size
            grid_w = max(1, int(round(grid_size * aspect)))

        # Downsample
        small = cv2.resize(frame_rgb, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
        rgb_small = small.astype(np.float32)/255.0

        matched_idx = vectorized_match_grid(rgb_small, emoji_lab, emoji_lab_L, nn)

        # Build output
        out_h, out_w = grid_h*tile_px, grid_w*tile_px
        output = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        for y in range(grid_h):
            for x in range(grid_w):
                idx = matched_idx[y, x]
                y0, y1 = y*tile_px, (y+1)*tile_px
                x0, x1 = x*tile_px, (x+1)*tile_px
                output[y0:y1, x0:x1] = emoji_imgs[idx]

        cv2.imshow("Emoji Mosaic Live", cv2.cvtColor(output, cv2.COLOR_RGBA2BGR))

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Run ----------
if __name__ == "__main__":
    live_emoji_camera()
