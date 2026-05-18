"""
PaddleOCR の共通インスタンス管理モジュール

全 BBox Extractor バージョンで PaddleOCR インスタンスを共有し、
メモリ使用量を抑える。
"""
import contextlib
import os
import sys
import warnings

from paddleocr import PaddleOCR

# Paddle/PaddleOCR の警告とログを抑制
warnings.filterwarnings("ignore", message="No ccache found")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# PaddleOCR インスタンスをグローバルに保持（シングルトン）
_ocr = None


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Paddle/PaddleX の情報ログを一時的に抑制するコンテキストマネージャ"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_ocr():
    """PaddleOCR インスタンスを取得（遅延初期化・シングルトン）"""
    global _ocr
    if _ocr is None:
        with suppress_stdout_stderr():
            _ocr = PaddleOCR(lang="en")
    return _ocr
