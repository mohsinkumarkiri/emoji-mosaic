# emoji-mosaic
Developed in python that uses LAB color space (perceptual) and nearest-neighbor search for matching.


<img width="1279" height="470" alt="image (1)" src="https://github.com/user-attachments/assets/9f24043f-fb82-48e5-bbf2-0da3adef3af9" />



# 🧩 Emoji Mosaic Generator

Turn ordinary images into **beautiful emoji mosaics** — where every pixel becomes an emoji 🎨.  
This project is written in **Python** as a proof-of-concept and will later be ported to **Unity (C#)** for real-time applications.

---

## 🚀 Overview

The **Emoji Mosaic Generator** replaces each pixel (or group of pixels) in an image with the emoji that best matches its color.  
It uses **color averaging**, **LAB color space comparison**, and **Floyd–Steinberg dithering** for smooth, natural-looking results.

The goal is to demonstrate an efficient image-to-emoji conversion pipeline before integrating it into Unity as part of the final phase.

---

## 🧠 Core Features

✅ Perceptual color matching using **CIE LAB space**  
✅ **Floyd–Steinberg dithering** for smooth gradients  
✅ **Cached emoji data** for faster processing  
✅ Handles **light and shadow balance** effectively  
✅ Ready for **Unity C# porting** (texture-based workflow)

---



---

## 🔬 Project Phases

| **Phase** | **Goal** | **Description** |
|------------|-----------|----------------|
| **Phase 1 – Base Prototype** | ✅ Completed | Basic grid-based emoji mosaic without dithering. |
| **Phase 2 – Dithering** | ✅ Completed | Adds Floyd–Steinberg dithering for smoother tone transitions. |
| **Phase 2.3 – Balanced Shadows** | ✅ Completed | Adjusts brightness weighting to handle darker image areas more naturally. |
| **Phase 3 – Enhancement** | 🔄 Upcoming | Add brightness/contrast normalization and histogram balancing. |
| **Phase 4 – Unity Port** | 🔜 Future | Port the full logic into Unity (C#) using Texture2D and Color32 arrays. |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/emoji-mosaic.git
cd emoji-mosaic

python -m venv venv

On Windows - venv\Scripts\activate
On Linux/macOS - source venv/bin/activate

Install Dependencies - pip install -r requirements.txt

Place you'r emojis inside - emoji-mosaic/emojis/



⭐ Acknowledgements

Inspired by:

Kyle Chen – Emoji Mosaic Research

Open-source emoji sets (Twemoji, Noto Emoji, etc.)

The creative coding community ❤️


