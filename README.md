# DishSnap

A Streamlit app that turns text-only restaurant menu photos into beautiful illustrated menus using OpenAI's GPT-4o and gpt-image-2.

Designed for **private, personal use** — access from your iPhone via Tailscale while running on your home Mac.

## Features

- **Upload** a photo of a restaurant menu (works great with iPhone camera)
- **Automatic perspective correction** — fixes skewed/angled menu photos
- **GPT-4o Vision** extracts menu items (name, price, description) with multiple extraction strategies
- **gpt-image-2** generates appetizing food photos for each dish
- **Smart layout** — places dish images next to their corresponding menu text
- **Multiple layout modes** — choose how images align with text
- **Versioned cache** — caches generated images per menu and extractor version
- **Debug tools** — visualize detected text blocks, compare extractor accuracy
- **Download** the final illustrated menu as a PNG

## Requirements

- **macOS** with Python 3.12+
- **OpenAI API key** with access to GPT-4o and gpt-image-2
- **Tailscale** (for iPhone access from outside your home network)

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. (Optional) Set up Tailscale for iPhone access

1. Install Tailscale on your Mac and iPhone
2. Sign in with the same account on both devices
3. Note your Mac's Tailscale IP (e.g., `100.xxx.xxx.xxx`)

### 4. Run the app

```bash
./run.sh
```

This starts Streamlit on `0.0.0.0:8501`, accessible from any device on your Tailscale network.

### 5. Access from iPhone

1. Open the Tailscale app on your iPhone and ensure VPN is active
2. Open Safari/Chrome and navigate to:
   ```
   http://<your-mac-tailscale-ip>:8501
   ```
3. (Recommended) Add to Home Screen for app-like access

> **Important:** Keep your Mac awake while using remotely. Go to **System Settings → Lock Screen** and enable "Prevent automatic sleeping when the display is off" (while on power adapter).

## Usage

1. **Take a photo** of a restaurant menu with your iPhone
2. **Upload** it via the web interface
3. The app automatically **corrects perspective** and **reads EXIF orientation**
4. Click **"Generate Illustrated Menu"**
5. Wait while the app:
   - Extracts menu items and their positions (Step 1-3)
   - Verifies matches with GPT-4o (Step 4)
   - Generates dish images with gpt-image-2 (Step 5, cached on repeat runs)
   - Composes the final illustrated menu (Step 6)
6. **Preview and download** your illustrated menu as a PNG

### Sidebar Options

- **BBox Extractor Version**: Choose the extraction strategy
  - `v5` (default): GPT-4o Direct — highest accuracy, hybrid with PaddleOCR
  - `v1`: GPT-4o direct bbox extraction
  - `v2-v4`: PaddleOCR-based strategies with different merge algorithms
- **Cache Management**: Clear all or per-menu caches
- **Layout Mode**: Choose how generated images align with menu text

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI, workflow orchestration, session state management |
| `run.sh` | Startup script with `--server.address 0.0.0.0` for Tailscale access |
| `menu_extractor.py` | GPT-4o Vision menu extraction (for v1-v4) |
| `image_generator.py` | gpt-image-2 parallel image generation (max 5 workers) |
| `menu_layout.py` | Pillow-based layout engine with multiple layout modes |
| `image_corrector.py` | Automatic perspective correction using OpenCV |
| `bbox_refiner.py` | GPT-4o-based bbox verification and correction |
| `bbox_extractors/` | Pluggable extractor versions (v1-v5) |
| `bbox_extractors/common_ocr.py` | Shared PaddleOCR singleton (memory-optimized) |
| `bbox_extractors/v5_gpt4o_direct.py` | Hybrid GPT-4o + PaddleOCR extractor (default) |
| `bbox_extractors/v1_gpt4o.py` | Pure GPT-4o bbox extraction |
| `bbox_extractors/v2_paddle_simple.py` | PaddleOCR + simple merge |
| `bbox_extractors/v3_paddle_advanced.py` | PaddleOCR + multi-scale + K-means clustering |
| `bbox_extractors/v4_paddle_fixed.py` | PaddleOCR + fixed threshold merge |

## Technical Notes

- **Image resizing**: High-resolution iPhone photos (e.g., 3024×4032) are automatically resized to a max long edge of 2048px before OCR and API calls. This dramatically speeds up processing without impacting menu-reading accuracy. Bbox coordinates are still computed relative to the original image dimensions.
- **EXIF orientation**: iPhone portrait photos are correctly oriented via `ImageOps.exif_transpose` before processing.
- **Memory optimization**: All PaddleOCR-based extractors (v2-v5) share a single OCR instance via `common_ocr.py`, reducing memory usage by ~75%.
- **Debug comparison**: The "Compare BBox Extractors (v1-v5)" panel requires clicking **"Run Comparison"** to execute — it does not auto-run on expand, preventing accidental memory spikes.
- **Caching**: Generated dish images are cached in `generated_images/<menu_hash>/<version>/`. This allows you to switch layout modes or re-layout without regenerating images.
- **Designed primarily for English menus**, though GPT-4o may handle other languages depending on the model's capabilities.
