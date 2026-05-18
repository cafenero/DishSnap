# DishSnap

English version: [README.md](README.md)

OpenAI の GPT-4o と gpt-image-2 を使って、文字だけのレストランメニュー写真を美しいイラスト付きメニューに変換する Streamlit アプリです。

自宅の Mac で動かし、**iPhone から Tailscale 経由で使える**ように設計されています。

## 主な機能

- **iPhone のカメラで撮影したメニュー写真**をそのままアップロード可能
- **自動台形補正** — 斜めから撮影したメニュー写真をまっすぐに補正
- **複数の抽出戦略**から選べる **GPT-4o Vision** — メニュー項目（料理名、価格、説明）と位置情報を抽出
- **gpt-image-2** — 各料理の魅力的な写真を自動生成
- **スマートレイアウト** — 生成した料理画像をメニュー本文の横に自動配置
- **複数レイアウトモード** — 画像の配置揃えを選択可能
- **バージョン別キャッシュ** — メニューごと・抽出器バージョンごとに画像をキャッシュ
- **デバッグツール** — 検出されたテキストブロックの可視化、抽出器精度の比較
- **PNG ダウンロード** — 完成したイラスト付きメニューを画像として保存

## 必要なもの

- **macOS** + Python 3.12+
- **OpenAI API key**（GPT-4o と gpt-image-2 にアクセス可能なもの）
- **Tailscale**（外出先の iPhone から自宅 Mac にアクセスするため）

## セットアップ

### 1. 依存関係をインストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API キーを設定

```bash
cp .env.example .env
# .env を編集して OPENAI_API_KEY を追加
```

### 3. Tailscale のセットアップ（iPhone からアクセスする場合）

1. Mac と iPhone の両方に Tailscale をインストール
2. 同じアカウントで両方の端末にサインイン
3. Mac の Tailscale IP を確認（例：`100.xxx.xxx.xxx`）

### 4. アプリを起動

```bash
./run.sh
```

これにより、Streamlit が `0.0.0.0:8501` で起動し、Tailscale ネットワーク内のどの端末からでもアクセス可能になります。

### 5. iPhone からアクセス

1. iPhone の Tailscale アプリを開き、VPN が有効になっていることを確認
2. Safari / Chrome で以下の URL を開く：
   ```
   http://<macの-tailscale-ip>:8501
   ```
3. （推奨）Safari の「ホーム画面に追加」でアプリ感覚で起動可能

> **重要：** 外出先から利用する際は、Mac がスリープしないよう設定してください。**システム設定 → ロック画面** で、「ディスプレイがオフになった後にスリープさせない」（電源アダプタ接続時）を有効にしてください。

## 使い方

1. iPhone のカメラで**レストランメニューを撮影**
2. Web ブラウザから**写真をアップロード**
3. アプリが自動で**透視補正**と **EXIF 方向の読み取り**を実行
4. **「Generate Illustrated Menu」**をクリック
5. 以下の処理が自動で実行されます：
   - メニュー項目と位置情報の抽出（Step 1-3）
   - GPT-4o によるマッチング検証（Step 4）
   - gpt-image-2 による料理画像生成（Step 5、2回目以降はキャッシュ利用）
   - イラスト付きメニューの自動レイアウト（Step 6）
6. **プレビューして PNG をダウンロード**

### サイドバー設定

- **BBox Extractor Version**: 抽出戦略を選択
  - `v5`（デフォルト）: GPT-4o Direct — 最高精度、PaddleOCR とのハイブリッド方式
  - `v1`: GPT-4o による直接 bbox 抽出
  - `v2-v4`: PaddleOCR ベースの戦略（マージアルゴリズムの違い）
- **Cache Management**: キャッシュの全削除またはメニュー単位の削除
- **Layout Mode**: 生成画像とメニュー本文の配置揃えを選択

## プロジェクト構成

| ファイル | 説明 |
|------|-------------|
| `app.py` | Streamlit UI、ワークフロー統合、セッション状態管理 |
| `run.sh` | `--server.address 0.0.0.0` 付きの起動スクリプト（Tailscale アクセス用） |
| `menu_extractor.py` | GPT-4o Vision によるメニュー抽出（v1-v4 用） |
| `image_generator.py` | gpt-image-2 による並列画像生成（最大 5 ワーカー） |
| `menu_layout.py` | Pillow ベースのレイアウトエンジン（複数レイアウトモード対応） |
| `image_corrector.py` | OpenCV による自動台形補正 |
| `bbox_refiner.py` | GPT-4o による bbox 検証・補正 |
| `bbox_extractors/` | プラガブルな抽出器バージョン（v1-v5） |
| `bbox_extractors/common_ocr.py` | 共有 PaddleOCR シングルトン（メモリ最適化） |
| `bbox_extractors/v5_gpt4o_direct.py` | ハイブリッド GPT-4o + PaddleOCR 抽出器（デフォルト） |
| `bbox_extractors/v1_gpt4o.py` | 純粋 GPT-4o bbox 抽出 |
| `bbox_extractors/v2_paddle_simple.py` | PaddleOCR + シンプルマージ |
| `bbox_extractors/v3_paddle_advanced.py` | PaddleOCR + マルチスケール + K-means クラスタリング |
| `bbox_extractors/v4_paddle_fixed.py` | PaddleOCR + 固定閾値マージ |

## 技術的な補足

- **画像のリサイズ**: iPhone の高解像度写真（例：3024×4032）は、OCR と API 呼び出し前に長辺 2048px に自動縮小されます。これにより処理が劇的に高速化され、メニュー読み取り精度には影響しません。bbox 座標は元画像サイズ基準で計算されます。
- **EXIF Orientation**: iPhone で縦向きに撮影した写真は、`ImageOps.exif_transpose` により正しい向きで処理されます。
- **メモリ最適化**: PaddleOCR ベースの抽出器（v2-v5）はすべて `common_ocr.py` を通じて単一の OCR インスタンスを共有し、メモリ使用量を約 75% 削減しています。
- **デバッグ比較**: 「Compare BBox Extractors (v1-v5)」パネルは、**「Run Comparison」ボタンを押したときのみ**実行されます。exander を開いただけでは自動実行されないため、意図しないメモリ消費を防ぎます。
- **キャッシュ**: 生成された料理画像は `generated_images/<menu_hash>/<version>/` にキャッシュされます。レイアウトモードの切り替えや再レイアウト時に画像を再生成する必要がなくなります。
- **主に英語のメニュー向けに設計**されていますが、GPT-4o の能力次第で他の言語も処理できる場合があります。
