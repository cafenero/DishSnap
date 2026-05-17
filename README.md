# DishSnap

A simple Streamlit app that turns text-only restaurant menu photos into beautiful illustrated menus using OpenAI's GPT-4o and DALL-E 3.

## Features

- Upload a photo of a text-only restaurant menu
- GPT-4o Vision extracts menu items (name, price, description)
- DALL-E 3 generates appetizing food photos for each dish
- Combines everything into a single illustrated menu image
- Download the final menu as a PNG

## Setup

1. Clone the repository and navigate to the project folder.

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:

   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open your browser at the URL shown (usually `http://localhost:8501`).

## Usage

1. Upload a clear photo of a restaurant menu.
2. Click **"Generate Illustrated Menu"**.
3. Wait while the app reads the menu and generates dish images.
4. Preview and download your illustrated menu as a PNG.

## Requirements

- Python 3.9+
- OpenAI API key with access to GPT-4o and DALL-E 3

## Project Structure

- `app.py` — Streamlit UI and workflow
- `menu_extractor.py` — GPT-4o Vision menu extraction
- `image_generator.py` — DALL-E 3 image generation
- `menu_layout.py` — Pillow-based menu layout engine

## Notes

- The app is designed primarily for English menus.
- Image generation can take some time depending on the number of menu items.
- If a dish image fails to generate, it is skipped and the rest are included in the final menu.
