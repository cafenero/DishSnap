import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from bbox_extractors import extract_text_bboxes_versioned
from bbox_refiner import refine_bboxes_with_gpt4o
from menu_extractor import extract_menu_items, match_bboxes
from image_generator import generate_all_images
from menu_layout import create_annotated_menu_image, create_bbox_debug_image
from image_corrector import correct_perspective

load_dotenv()

CACHE_ROOT = Path("generated_images")
CACHE_ROOT.mkdir(exist_ok=True)


def compute_image_hash(image: Image.Image) -> str:
    """PIL Image のバイト列から MD5 ハッシュを計算する"""
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def load_cache(menu_hash: str, bbox_version: str):
    """バージョン別キャッシュディレクトリからメニュー項目と画像を読み込む"""
    cache_dir = CACHE_ROOT / menu_hash / bbox_version
    metadata_path = cache_dir / "metadata.json"

    if not metadata_path.exists():
        return None, None

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    menu_items = metadata.get("menu_items", [])
    images = []
    for i in range(len(menu_items)):
        img_path = cache_dir / f"item_{i}.png"
        if img_path.exists():
            images.append(Image.open(img_path))
        else:
            images.append(None)

    return menu_items, images


def save_metadata(cache_dir: Path, menu_items: list):
    """メニュー項目を metadata.json に保存する"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"menu_items": menu_items}, f, ensure_ascii=False, indent=2)


def _rmdir_recursive(path: Path):
    """ディレクトリとその中身を再帰的に削除する"""
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _rmdir_recursive(child)
        else:
            child.unlink()
    path.rmdir()


def clear_all_cache():
    """全キャッシュを削除する（全メニュー・全バージョン）"""
    for subdir in CACHE_ROOT.iterdir():
        if subdir.is_dir():
            _rmdir_recursive(subdir)


def clear_menu_cache(menu_hash: str, bbox_version: str):
    """特定メニューの特定バージョンキャッシュを削除する"""
    cache_dir = CACHE_ROOT / menu_hash / bbox_version
    if cache_dir.exists():
        _rmdir_recursive(cache_dir)
    # バージョンが空になったら親ディレクトリも削除
    parent_dir = CACHE_ROOT / menu_hash
    if parent_dir.exists() and not any(parent_dir.iterdir()):
        parent_dir.rmdir()


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
    st.session_state.menu_hash = None
    st.session_state.cache_available = False
    st.session_state.processing_steps = []

# layout_mode の初期化（radio key 用）
if "layout_mode" not in st.session_state:
    st.session_state.layout_mode = "layout2"

# bbox_version の初期化
if "bbox_version" not in st.session_state:
    st.session_state.bbox_version = "v5_gpt4o_direct"


def record_step(step_num, name, status, start_time, end_time=None, details=""):
    """処理ステップを記録する"""
    step = {
        "step": step_num,
        "name": name,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration": (end_time - start_time) if end_time else None,
        "details": details,
    }
    st.session_state.processing_steps.append(step)


def update_progress(step_num, total_steps, name, details=""):
    """リアルタイム進捗を更新する"""
    progress_text = f"🔄 Step {step_num}/{total_steps}: {name}"
    if details:
        progress_text += f"\n   {details}"
    return progress_text

# サイドバー
with st.sidebar:
    st.header("🧪 BBox Extractor Version")
    st.radio(
        "Select extraction method",
        ["v1_gpt4o", "v2_paddle_simple", "v3_paddle_advanced", "v4_paddle_fixed", "v5_gpt4o_direct"],
        format_func=lambda x: {
            "v1_gpt4o": "v1: GPT-4o direct",
            "v2_paddle_simple": "v2: PaddleOCR simple merge",
            "v3_paddle_advanced": "v3: PaddleOCR + K-means + GPT-4o",
            "v4_paddle_fixed": "v4: PaddleOCR fixed threshold",
            "v5_gpt4o_direct": "v5: GPT-4o direct menu+bbox (high accuracy)",
        }[x],
        key="bbox_version",
    )

    st.header("Cache Management")
    if st.button("🗑️ Clear All Cache"):
        clear_all_cache()
        st.success("All cache cleared!")
        st.session_state.cache_available = False
        st.rerun()

    st.header("レイアウト設定")
    st.radio(
        "生成画像の配置",
        ["layout1", "layout2"],
        format_func=lambda x: {
            "layout1": "レイアウト1：画像上端をテキスト上端に揃える",
            "layout2": "レイアウト2：画像中央をテキスト中央に揃える",
        }[x],
        key="layout_mode",
    )

# 画像アップロード
uploaded_file = st.file_uploader("Choose a menu photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # アップロード画像の読み込み
    raw_image = Image.open(uploaded_file)

    # 自動台形補正
    input_image, correction_success, correction_info = correct_perspective(raw_image)

    # 補正結果の表示
    if correction_success:
        st.success(f"✅ Perspective correction applied. ({correction_info})")
        with st.expander("🔬 Debug: Perspective Correction", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(raw_image, width="stretch")
            with col2:
                st.markdown("**Corrected**")
                st.image(input_image, width="stretch")
    else:
        st.info(f"ℹ️ Using original image ({correction_info}).")
        input_image = raw_image

    st.image(input_image, caption="Uploaded Menu", width="stretch")

    # 新しいファイルがアップロードされたら生成状態をリセット
    if st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.generated = False
        st.session_state.menu_image_buffer = None
        st.session_state.menu_items = None
        st.session_state.processing_time = None
        st.session_state.progress_log = []
        st.session_state.last_uploaded_name = uploaded_file.name
        st.session_state.menu_hash = None
        st.session_state.cache_available = False

    # ハッシュ計算
    if st.session_state.menu_hash is None:
        st.session_state.menu_hash = compute_image_hash(input_image)

    menu_hash = st.session_state.menu_hash
    bbox_version = st.session_state.bbox_version
    cache_dir = CACHE_ROOT / menu_hash / bbox_version

    # キャッシュ存在チェック（バージョン別）
    if not st.session_state.cache_available:
        cached_items, cached_images = load_cache(menu_hash, bbox_version)
        st.session_state.cache_available = (
            cached_items is not None and cached_images is not None
        )
        if st.session_state.cache_available:
            st.session_state.cached_items = cached_items
            st.session_state.cached_images = cached_images

    # キャッシュ通知（バージョン名付き）
    if st.session_state.cache_available:
        version_label = {
            "v1_gpt4o": "v1",
            "v2_paddle_simple": "v2",
            "v3_paddle_advanced": "v3",
            "v4_paddle_fixed": "v4",
            "v5_gpt4o_direct": "v5",
        }.get(bbox_version, bbox_version)
        st.info(f"📦 Cache found ({version_label}) for this menu image. You can generate using cached images.")
        if st.button("🗑️ Clear This Menu Cache"):
            clear_menu_cache(menu_hash, bbox_version)
            st.session_state.cache_available = False
            st.success(f"This menu's cache ({version_label}) cleared!")
            st.rerun()

    if st.button("✨ Generate Illustrated Menu", type="primary"):
        st.session_state.progress_log = []
        st.session_state.processing_steps = []

        # リアルタイム進捗表示用プレースホルダー
        progress_placeholder = st.empty()
        
        try:
            total_steps = 6
            start_time = time.time()
            bbox_version = st.session_state.bbox_version
            is_v5 = bbox_version == "v5_gpt4o_direct"

            # Step 1: メニュー情報を抽出（キャッシュ有無に関わらず実行）
            step_start = time.time()
            # v5 は bbox 抽出方式が異なるため、キャッシュがあっても必ず再抽出する
            if st.session_state.cache_available and not is_v5:
                progress_placeholder.markdown(update_progress(1, total_steps, "Loading menu items from cache..."))
                menu_items = st.session_state.cached_items
                record_step(1, "Load menu items", "completed", step_start, time.time(), 
                           f"{len(menu_items)} items from cache")
                progress_placeholder.markdown(f"✅ Step 1/6: Loaded {len(menu_items)} items from cache")
            elif is_v5:
                # v5: GPT-4o で name/price/description/bbox を同時に抽出
                progress_placeholder.markdown(update_progress(1, total_steps, "Extracting menu items with GPT-4o Vision..."))
                menu_items = extract_text_bboxes_versioned(input_image, version=bbox_version, client=client)
                
                if not menu_items:
                    record_step(1, "Extract menu items (v5)", "failed", step_start, time.time(), 
                               "No items found")
                    st.warning("No menu items found in the photo. Please try another image.")
                    st.stop()
                
                record_step(1, "Extract menu items (v5)", "completed", step_start, time.time(), 
                           f"Found {len(menu_items)} items with bbox")
                progress_placeholder.markdown(f"✅ Step 1/6: v5 found {len(menu_items)} menu items with bbox")
            else:
                # v1-v4: 従来の extract_menu_items
                progress_placeholder.markdown(update_progress(1, total_steps, "Reading menu items from photo..."))
                import io

                image_bytes = io.BytesIO()
                input_image.convert("RGB").save(image_bytes, format="JPEG")
                image_bytes.seek(0)
                menu_items = extract_menu_items(image_bytes, client)
                
                if not menu_items:
                    record_step(1, "Extract menu items", "failed", step_start, time.time(), 
                               "No items found")
                    st.warning("No menu items found in the photo. Please try another image.")
                    st.stop()
                
                record_step(1, "Extract menu items", "completed", step_start, time.time(), 
                           f"Found {len(menu_items)} items")
                progress_placeholder.markdown(f"✅ Step 1/6: Found {len(menu_items)} menu items")

            # Step 2: bbox を抽出（v1-v4 のみ。v5 は Step 1 で既に bbox 取得済み）
            step_start = time.time()
            if is_v5:
                # v5: bbox は既に menu_items に含まれている
                ocr_results = []  # refine_bboxes_with_gpt4o 用ダミー
                record_step(2, f"OCR text detection ({bbox_version})", "completed", step_start, time.time(), 
                           "Skipped (bbox included in v5 output)")
                progress_placeholder.markdown(f"✅ Step 2/6: v5 bbox included, skipped")
            else:
                progress_placeholder.markdown(update_progress(2, total_steps, f"Extracting text positions with {bbox_version}..."))
                
                # v1 は GPT-4o を使うため client が必要
                if bbox_version == "v1_gpt4o":
                    ocr_results = extract_text_bboxes_versioned(input_image, version=bbox_version, client=client)
                else:
                    ocr_results = extract_text_bboxes_versioned(input_image, version=bbox_version)
                
                record_step(2, f"OCR text detection ({bbox_version})", "completed", step_start, time.time(), 
                           f"Detected {len(ocr_results)} text blocks")
                progress_placeholder.markdown(f"✅ Step 2/6: {bbox_version} detected {len(ocr_results)} blocks")

            # Step 3: マッチング（v1-v4 のみ。v5 は既に bbox 紐付け済み）
            step_start = time.time()
            if is_v5:
                matched_count = sum(1 for item in menu_items if item.get("bbox"))
                record_step(3, "Match menu items with OCR", "completed", step_start, time.time(), 
                           f"Skipped (v5 pre-matched) - {matched_count} items have bbox")
                progress_placeholder.markdown(f"✅ Step 3/6: v5 pre-matched, {matched_count} items have bbox")
            else:
                progress_placeholder.markdown(update_progress(3, total_steps, "Matching menu items with OCR blocks..."))
                menu_items = match_bboxes(menu_items, ocr_results)
                matched_count = sum(1 for item in menu_items if item.get("bbox"))
                record_step(3, "Match menu items with OCR", "completed", step_start, time.time(), 
                           f"Matched {matched_count}/{len(menu_items)} items")
                progress_placeholder.markdown(f"✅ Step 3/6: Matched {matched_count}/{len(menu_items)} items")

            # Step 4: GPT-4o による検証・補正（キャッシュ有無に関わらず毎回実行）
            step_start = time.time()
            progress_placeholder.markdown(update_progress(4, total_steps, "Verifying matches with GPT-4o..."))
            menu_items = refine_bboxes_with_gpt4o(menu_items, ocr_results, client)
            record_step(4, "GPT-4o verification", "completed", step_start, time.time(), 
                       "Bbox verification completed")
            progress_placeholder.markdown("✅ Step 4/6: GPT-4o verification done")

            items_md = "\n".join(
                [
                    f"- **{item['name']}** ({item['price']}): {item['description']}"
                    for item in menu_items
                ]
            )
            st.session_state.progress_log = [
                f"✅ Found {len(menu_items)} menu item(s)!\n\n{items_md}"
            ]

            # Step 5: 各料理の画像を生成（キャッシュがあればスキップ）
            step_start = time.time()
            if st.session_state.cache_available:
                images = st.session_state.cached_images
                record_step(5, "Generate dish images", "skipped", step_start, time.time(), 
                           "Using cached images")
                progress_placeholder.markdown(f"✅ Step 5/6: Using {len(images)} cached images")
            else:
                progress_placeholder.markdown(update_progress(5, total_steps, "Generating dish images..."))
                progress_bar = st.progress(0, text="🎨 Generating dish images...")
                images = generate_all_images(menu_items, client, cache_dir=cache_dir)
                progress_bar.progress(100, text="🎨 Generating dish images... Done!")
                
                successful = sum(1 for img in images if img is not None)
                record_step(5, "Generate dish images", "completed", step_start, time.time(), 
                           f"Generated {successful}/{len(menu_items)} images")
                progress_placeholder.markdown(f"✅ Step 5/6: Generated {successful}/{len(menu_items)} images")

                # メタデータ保存
                save_metadata(cache_dir, menu_items)

                # 進捗ログを構築
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

                successful = sum(1 for img in images if img is not None)
                if successful == 0:
                    st.error("Failed to generate all dish images. Please try again.")
                    st.stop()

            # Step 6: メニュー画像をレイアウト（キャッシュ有無に関わらず実行）
            step_start = time.time()
            progress_placeholder.markdown(update_progress(6, total_steps, "Designing menu layout..."))
            menu_image_buffer = create_annotated_menu_image(
                input_image, menu_items, images, layout_mode=st.session_state.layout_mode
            )
            record_step(6, "Design menu layout", "completed", step_start, time.time(), 
                       "Layout completed")

            elapsed = time.time() - start_time
            progress_placeholder.markdown(f"✅ All steps completed in {elapsed:.1f}s")
            record_step(7, "Total processing", "completed", start_time, time.time(), 
                       f"Total: {elapsed:.1f}s")

            # 結果をセッション状態に保存
            st.session_state.generated = True
            st.session_state.menu_image_buffer = menu_image_buffer
            st.session_state.menu_items = menu_items
            st.session_state.processing_time = elapsed

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            if st.session_state.processing_steps:
                st.session_state.processing_steps[-1]["status"] = "failed"
                st.session_state.processing_steps[-1]["details"] += f" Error: {str(e)}"

    # 処理ステップ詳細を表示（折りたたみ）
    if st.session_state.processing_steps:
        with st.expander("📋 View processing details", expanded=False):
            total_duration = sum(
                step.get("duration", 0) or 0 
                for step in st.session_state.processing_steps
            )
            st.markdown(f"**Total processing time: {total_duration:.1f}s**")
            st.divider()
            
            for step in st.session_state.processing_steps:
                status_icon = {
                    "completed": "✅",
                    "running": "🔄",
                    "failed": "❌",
                }.get(step["status"], "⏳")
                
                st.markdown(
                    f"{status_icon} **Step {step['step']}**: {step['name']}"
                )
                if step.get("duration"):
                    st.markdown(f"   ⏱️ Duration: {step['duration']:.2f}s")
                if step.get("details"):
                    st.markdown(f"   📝 {step['details']}")
                st.markdown("")

    # 抽出詳細を表示
    if st.session_state.menu_items is not None:
        with st.expander("View extracted items", expanded=False):
            st.markdown("\n\n".join(st.session_state.progress_log))

    # 生成済みの結果を表示
    if st.session_state.generated and st.session_state.menu_image_buffer:
        st.subheader("Your Illustrated Menu")
        st.image(st.session_state.menu_image_buffer, width="stretch")

        st.download_button(
            label="📥 Download Menu (PNG)",
            data=st.session_state.menu_image_buffer,
            file_name="illustrated_menu.png",
            mime="image/png",
        )

        # デバッグ: bbox 可視化画像も表示
        if st.session_state.menu_items is not None:
            with st.expander("🔍 Debug: BBox Visualization", expanded=False):
                debug_buffer = create_bbox_debug_image(
                    input_image, st.session_state.menu_items
                )
                st.image(debug_buffer, width="stretch")

        # デバッグ: v5 ピクセル座標 vs 正規化座標の比較表示
        if (
            st.session_state.menu_items is not None
            and st.session_state.bbox_version == "v5_gpt4o_direct"
        ):
            with st.expander("🔬 Debug: v5 BBox Comparison (Pixel vs Normalized)", expanded=False):
                st.markdown("**v5 BBox: Raw Pixel vs Normalized Coordinates**")
                st.caption("Blue = raw pixel bbox / Red = normalized bbox (should be identical)")

                items = st.session_state.menu_items
                rows = []
                for item in items:
                    bp = item.get("bbox_pixels", [0, 0, 0, 0])
                    bn = item.get("bbox", [0, 0, 0, 0])
                    rows.append({
                        "Name": item.get("name", "")[:25],
                        "Pixel Height": bp[3] - bp[1],
                        "Norm Height": round((bn[3] - bn[1]) * input_image.height),
                    })
                st.dataframe(rows, width='stretch')

                # 青枠: bbox_pixels を一時的に bbox として描画
                pixel_items = []
                for item in items:
                    bp = item.get("bbox_pixels", [])
                    if bp:
                        pixel_items.append({
                            "name": item.get("name", "") + " (pixel)",
                            "bbox": [
                                bp[0] / input_image.width,
                                bp[1] / input_image.height,
                                bp[2] / input_image.width,
                                bp[3] / input_image.height,
                            ],
                        })
                blue_buf = create_bbox_debug_image(input_image, pixel_items, color="#0000FF")
                st.image(blue_buf, width="stretch")
                st.caption("Blue: Raw pixel coordinates rendered as normalized")

                # 赤枠: 通常の normalized bbox
                red_buf = create_bbox_debug_image(input_image, items, color="#FF0000")
                st.image(red_buf, width="stretch")
                st.caption("Red: Normalized bbox from v5")

        # デバッグ: 全バージョンの bbox 抽出結果を比較表示
        if input_image is not None:
            with st.expander("🔬 Debug: Compare BBox Extractors (v1-v5)", expanded=False):
                st.markdown("**Compare all extractor versions side-by-side**")
                st.caption("Red boxes show detected text blocks. Green labels show matched menu items.")

                versions = ["v1_gpt4o", "v2_paddle_simple", "v3_paddle_advanced", "v4_paddle_fixed", "v5_gpt4o_direct"]
                version_names = {
                    "v1_gpt4o": "v1: GPT-4o",
                    "v2_paddle_simple": "v2: PaddleOCR Simple",
                    "v3_paddle_advanced": "v3: PaddleOCR + K-means",
                    "v4_paddle_fixed": "v4: PaddleOCR Fixed",
                    "v5_gpt4o_direct": "v5: GPT-4o Direct (name+bbox)",
                }

                # 3列グリッドで表示（v5追加）
                cols = st.columns(3)
                for i, version in enumerate(versions):
                    with cols[i % 3]:
                        st.markdown(f"**{version_names[version]}**")
                        try:
                            if version in ("v1_gpt4o", "v5_gpt4o_direct"):
                                results = extract_text_bboxes_versioned(input_image, version=version, client=client)
                            else:
                                results = extract_text_bboxes_versioned(input_image, version=version)

                            # 擬似的に menu_items 形式に変換して可視化
                            dummy_items = []
                            for r in results:
                                # v5 は name キーを持つ、v1-v4 は text キーを持つ
                                label = r.get("name", r.get("text", ""))[:30]
                                dummy_items.append({
                                    "name": label,
                                    "bbox": r["bbox"]
                                })

                            debug_buf = create_bbox_debug_image(input_image, dummy_items)
                            st.image(debug_buf, width='stretch')
                            st.caption(f"Detected {len(results)} blocks")
                        except Exception as e:
                            st.error(f"Error: {e}")
