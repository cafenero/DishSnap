"""
画像の自動台形補正モジュール

OpenCV を使用して、メニュー画像の傾き・台形歪みを自動補正する。
検出失敗時は元画像をそのまま返すフォールバック機構付き。
"""
import cv2
import numpy as np
from PIL import Image


def correct_perspective(image: Image.Image) -> tuple[Image.Image, bool, str]:
    """
    PIL Image を自動台形補正する。

    Args:
        image: PIL Image（RGB）

    Returns:
        (corrected_image, success, info): 補正後画像、成功フラグ、詳細情報（角度や理由）
    """
    # PIL → OpenCV (BGR)
    img_np = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    # グレースケール
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーでノイズ除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny エッジ検出
    edges = cv2.Canny(blurred, 50, 150)

    # 膨張・収縮でエッジを強調
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 輪郭検出
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, False, "No contours detected"

    # 面積順にソート（大きいものから）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 最大輪郭を探し、4頂点を持つ四角形か確認
    target_contour = None
    for contour in contours[:5]:  # 上位5個をチェック
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            target_contour = approx
            break

    if target_contour is None:
        return image, False, "No 4-point quadrilateral found"

    # 4頂点を取得 [top-left, top-right, bottom-right, bottom-left] の順に並べ替え
    pts = target_contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)

    # 幅・高さを計算
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # 最小サイズチェック（画像の30%未満なら失敗とみなす）
    if max_width < orig_w * 0.3 or max_height < orig_h * 0.3:
        return image, False, f"Detected region too small ({int(max_width/orig_w*100)}% width, {int(max_height/orig_h*100)}% height)"

    # 角度閾値チェック（5度未満なら補正をスキップ）
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
    if angle < 5.0:
        return image, False, f"Top edge angle: {angle:.1f}° (below 5° threshold)"

    # 射影変換
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_cv, M, (max_width, max_height))

    # OpenCV → PIL (RGB)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    corrected = Image.fromarray(warped_rgb)

    return corrected, True, f"Top edge angle: {angle:.1f}°"


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    4頂点を [top-left, top-right, bottom-right, bottom-left] の順に並べ替える。
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # x + y の和で top-left（最小）と bottom-right（最大）を特定
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # x - y の差で top-right（最小）と bottom-left（最大）を特定
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect
