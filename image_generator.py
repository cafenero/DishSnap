import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List

import requests
from PIL import Image
from openai import OpenAI, RateLimitError


def generate_dish_image(item: dict, client: OpenAI) -> Image.Image:
    """
    gpt-image-2で1つの料理画像を生成し、PIL Imageとして返す。
    """
    name = item.get("name", "Unknown dish")
    description = item.get("description", "")

    # プロンプト作成（英語）
    prompt = (
        f"A high-quality, appetizing food photography of {name}. "
        f"{description} "
        "Professional lighting, clean background, top-down or 45-degree angle, "
        "restaurant menu style photo."
    )

    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    image_data = response.data[0]

    if image_data.url:
        image_bytes = requests.get(image_data.url, timeout=30).content
    elif image_data.b64_json:
        image_bytes = base64.b64decode(image_data.b64_json)
    else:
        raise ValueError("No image data returned from API")

    return Image.open(BytesIO(image_bytes))


def generate_all_images(menu_items: List[dict], client: OpenAI) -> List[Image.Image]:
    """
    全メニュー項目の画像を並列生成する（最大20並列）。
    429エラー時はリトライする。
    失敗した場合はNoneを返す（呼び出し側でスキップ）。
    """
    results = [None] * len(menu_items)

    def task(idx: int, item: dict):
        name = item.get("name", "Unknown dish")
        for attempt in range(3):
            try:
                img = generate_dish_image(item, client)
                results[idx] = img
                print(f"✅ Generated: {name}")
                return
            except RateLimitError:
                wait_time = 12 * (attempt + 1)
                print(
                    f"Rate limit hit for '{name}', "
                    f"waiting {wait_time}s before retry {attempt + 1}/2..."
                )
                time.sleep(wait_time)
                if attempt == 2:
                    print(f"❌ Failed to generate image for '{name}': rate limit exceeded")
                    results[idx] = None
                    return
            except Exception as e:
                print(f"❌ Failed to generate image for '{name}': {e}")
                results[idx] = None
                return

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(task, i, item): i
            for i, item in enumerate(menu_items)
        }
        for future in as_completed(futures):
            future.result()

    return results
