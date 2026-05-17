import os
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from menu_extractor import extract_menu_items
from image_generator import generate_all_images
from menu_layout import create_menu_image

load_dotenv()

st.set_page_config(page_title="DishSnap", page_icon="🍽️")

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] {
        color: #31333F !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍽️ DishSnap")
st.markdown(
    "Upload a text-only restaurant menu photo, and we'll generate a beautiful illustrated menu for you!"
)

# APIキーの確認
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OpenAI API key not found. Please set it in your `.env` file or as an environment variable: OPENAI_API_KEY"
    )
    st.stop()

client = OpenAI(api_key=api_key)

# セッション状態の初期化
if "generated" not in st.session_state:
    st.session_state.generated = False
    st.session_state.menu_image_buffer = None
    st.session_state.last_uploaded_name = None
    st.session_state.menu_items = None
    st.session_state.processing_time = None
    st.session_state.progress_log = []

# 画像アップロード
uploaded_file = st.file_uploader("Choose a menu photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # アップロード画像のプレビュー
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Menu", width="stretch")

    # 新しいファイルがアップロードされたら生成状態をリセット
    if st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.generated = False
        st.session_state.menu_image_buffer = None
        st.session_state.menu_items = None
        st.session_state.processing_time = None
        st.session_state.progress_log = []
        st.session_state.last_uploaded_name = uploaded_file.name

    if st.button("✨ Generate Illustrated Menu", type="primary"):
        st.session_state.progress_log = []

        try:
            start_time = time.time()

            # 1. メニュー情報を抽出
            with st.spinner("🔍 Reading menu items from photo..."):
                import io

                image_bytes = io.BytesIO()
                input_image.convert("RGB").save(image_bytes, format="JPEG")
                image_bytes.seek(0)
                menu_items = extract_menu_items(image_bytes, client)

            if not menu_items:
                st.warning("No menu items found in the photo. Please try another image.")
                st.stop()

            items_md = "\n".join(
                [
                    f"- **{item['name']}** ({item['price']}): {item['description']}"
                    for item in menu_items
                ]
            )
            st.session_state.progress_log = [
                f"✅ Found {len(menu_items)} menu item(s)!\n\n{items_md}"
            ]

            # 2. 各料理の画像を並列生成（最大10並列）
            progress_bar = st.progress(0, text="🎨 Generating dish images...")
            images = generate_all_images(menu_items, client)
            progress_bar.progress(100, text="🎨 Generating dish images... Done!")

            successful = sum(1 for img in images if img is not None)
            if successful == 0:
                st.error("Failed to generate all dish images. Please try again.")
                st.stop()

            # 進捗ログを構築（メインスレッドで安全に実行）
            for i, (item, img) in enumerate(zip(menu_items, images)):
                name = item.get("name", "Unknown dish")
                if img is not None:
                    st.session_state.progress_log.append(
                        f"✅ Generated {i + 1}/{len(menu_items)}: {name}"
                    )
                else:
                    st.session_state.progress_log.append(
                        f"❌ Failed {i + 1}/{len(menu_items)}: {name}"
                    )

            # 3. メニュー画像をレイアウト
            st.session_state.progress_log.append("📐 Designing menu layout...")
            menu_image_buffer = create_menu_image(menu_items, images)

            elapsed = time.time() - start_time
            st.session_state.progress_log.append(
                f"⏱️ Total processing time: {elapsed:.1f} seconds"
            )

            # 結果をセッション状態に保存（ダウンロード後のrerun時も保持される）
            st.session_state.generated = True
            st.session_state.menu_image_buffer = menu_image_buffer
            st.session_state.menu_items = menu_items
            st.session_state.processing_time = elapsed

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    # 抽出詳細を表示（生成後もrerun後も保持）
    if st.session_state.menu_items is not None:
        with st.expander("View extracted items", expanded=False):
            st.markdown("\n\n".join(st.session_state.progress_log))

    # 生成済みの結果を表示（ダウンロードボタン押下後のrerun時もここで表示される）
    if st.session_state.generated and st.session_state.menu_image_buffer:
        st.subheader("Your Illustrated Menu")
        st.image(st.session_state.menu_image_buffer, width="stretch")

        st.download_button(
            label="📥 Download Menu (PNG)",
            data=st.session_state.menu_image_buffer,
            file_name="illustrated_menu.png",
            mime="image/png",
        )
