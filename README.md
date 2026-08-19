# Local Lens

一個以 Tauri 2、React 與 Rust 製作的 Windows 本機圖片搜尋 MVP。照片會在使用者選擇的資料夾中讀取，索引快取與搜尋索引保存在 app-data 的 SQLite；人物姓名與代表向量也只保存在本機，所有資料都不會上傳。

## 已完成的 MVP

- 選擇資料夾後遞迴掃描 JPG、JPEG、PNG、WebP、GIF、BMP
- 在 Rust 背景端產生縮圖，避免結果頁直接讀取全尺寸照片
- 依檔名搜尋，支援多個關鍵字的結果排序
- 若系統可找到 Tesseract，建立索引時會抽取 OCR 文字並納入搜尋
- 若 FastEmbed 模型可用，建立索引時會產生 CLIP 圖片向量，支援自然語言語意搜尋
- 使用 SCRFD 偵測人臉、ArcFace 產生 512 維身份向量，自動分群並讓使用者標記姓名
- 在設定中分別控制 CLIP、Face 的 DirectML GPU 加速及 Tesseract OCR 的實驗性 OpenCL 請求
- 格狀預覽、圖片尺寸與雙擊於檔案總管開啟原圖
- 預留 `ocr_text`、`people`、`score` 欄位，讓 OCR、人臉辨識及語意搜尋共用同一個索引介面
- 使用 SQLite FTS5 建立 OCR／檔名全文索引，使用 sqlite-vec `vec0` 建立 512 維 CLIP 與人臉向量索引
- Tauri capability 僅開啟對話框與開啟檔案功能；掃描在 Rust 命令端執行

## 在 Windows 開發環境執行

需先安裝 Node.js LTS、Rust stable，以及 Visual Studio 的 Desktop development with C++ 工作負載。此專案也暫時鎖定 `time` crate，以相容 Rust/Cargo 1.82；建議仍升級至較新的 stable Rust：

```powershell
rustup update stable
rustup default stable
```

```powershell
npm install
npm run icon
npm run tauri dev
```

`npm run icon` 會從專案內的 SVG 產生 Windows 所需的 `src-tauri/icons/icon.ico`。第一次執行（或替換 SVG 圖示後）需要執行一次。

## OCR 設定

OCR 使用本機 Tesseract，不會上傳圖片。請安裝 Tesseract 並確保 `tesseract` 在 PATH；繁體中文需要同時安裝 `chi_tra` 語言資料。若執行檔不在 PATH，可設定完整路徑：

```powershell
$env:LOCAL_LENS_TESSERACT = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:LOCAL_LENS_TESSERACT_LANG = "eng+chi_tra"
npm run tauri dev
```

若未安裝 Tesseract，應用程式仍可建立圖片索引，但會顯示「OCR 未啟用」，並只用檔名搜尋。

若照片主要是繁體中文，可將語言設為 `chi_tra`；中英混合才使用 `eng+chi_tra`。語言資料不完整時，Tesseract 很容易產生亂碼或低信心結果。

建置 Windows 安裝檔：

```powershell
npm run tauri build
```

## SQLite 索引快取與增量掃描

索引會寫入 app-data 的 `index.sqlite3`，保存圖片的路徑、檔案大小、修改時間、EXIF 拍攝時間、OCR 結果與人臉群組資料；CLIP 向量只存放在 `image_vectors`／`vector_rows`，不會放進記憶體中的 `ImageRecord` 或 metadata `record_json`。縮圖 JPEG 另外存放在 `image_thumbnails` BLOB 表，不會放進每張圖片的 metadata `record_json`。掃描會以兩次 streaming WalkDir 通過資料夾：第一次只統計數量與變更狀態，第二次逐張處理；快取資料逐路徑查詢，不會建立完整圖片路徑或快取 Vec。每批寫入後會釋放該批的縮圖、向量與指紋暫存。重新選擇同一資料夾時，程式會以檔案大小與修改時間判斷是否有變更：未變更的圖片直接重用 SQLite 快取，不會重新產生縮圖、執行 OCR 或模型推論；新增或變更的檔案才會重建索引。SQLite 會依設定的批次大小分批提交，介面進度與完成訊息會顯示重用的快取筆數；進度 ETA 只依需要新增或重建的檔案估算，避免快速重用快取拉低完成時間預估。EXIF 會優先讀取 `DateTimeOriginal`，再依序回退到 `DateTimeDigitized` 與 `DateTime`；沒有 EXIF 的圖片會保留空值，之後日期查詢可回退到檔案修改時間。

SQLite 會同時保存快取資料與搜尋索引：`image_thumbnails` 以 BLOB 保存縮圖，搜尋結果會以每頁最多 60 張的方式動態載入，只有目前頁面才懶載入縮圖；`image_ocr_fts` 是 FTS5 虛擬表，查詢 OCR／檔名文字時直接使用 SQLite MATCH；`image_vectors` 是 sqlite-vec `vec0` 虛擬表，CLIP 語意搜尋與人臉分群候選會使用 cosine KNN。`vector_rows` 保存向量與圖片路徑／人臉群組的對應，避免把所有向量一次載入記憶體。索引更新仍依檔案指紋增量處理，並依設定的批次大小提交。舊版本若將縮圖放在 `record_json`，第一次重用該快取時會重新產生並移至 `image_thumbnails`；舊版本若將 CLIP 向量放在 `record_json`，第一次載入時會以小批次搬移至 `image_vectors`，不會一次載入全部向量。

目前使用的 sqlite-vec crates.io 套件未包含選用的 DiskANN／rescore C 原始檔；專案根目錄的 `.cargo/config.toml` 會關閉這兩個非必要模組，保留本專案使用的核心 `vec0` cosine KNN 功能。

## 規則式 Query Parser

搜尋輸入會先經過本機規則式 Parser，再交給 FTS5 文字查詢與 sqlite-vec CLIP 搜尋流程。Parser 可辨識日期（例如「去年夏天」、「2024 年 7 月」）、已標記人物（例如「小明的照片」、「不要小明」）、人臉條件（「有人臉」、「沒有人的照片」）、圖片副檔名（例如 `jpg`）與結果數量（例如「最多 20 張」）；剩餘描述會交給語意搜尋。這一層不需要 LLM，完全離線，之後仍可在相同 QueryPlan 介面上加入可選的本機 LLM Parser。

## Semantic Search

Semantic Search 使用 FastEmbed 的配對 CLIP vision/text ONNX 模型。第一次建立索引（圖片模型）或第一次搜尋（文字模型）時會下載模型並快取；若下載失敗，索引仍會完成，但會退回檔名與 OCR 文字搜尋。此配對模型主要以英文文字訓練，英文描述通常比繁體中文查詢準確；中文 OCR／檔名搜尋仍可正常使用。

搜尋結果使用原始 cosine similarity，並套用最低信心與「距離最佳結果」的自適應門檻；語意搜尋會對完整向量集合套用閥值，所有符合條件的結果都能透過分頁載入。評分階段只保留圖片索引、路徑與分數，完成排序後才複製目前頁面的完整 metadata；檔名或 OCR 明確命中的圖片會優先保留。

應用程式會把模型快取放到目前平台的 app-data `models` 資料夾；若要自行指定位置，可在啟動前設定：

```powershell
$env:FASTEMBED_CACHE_DIR = "$env:LOCALAPPDATA\LocalLens\models"
npm run tauri dev
```

建立索引與第一次搜尋都在 Rust 背景工作執行，模型下載或載入時視窗仍可回應；首次使用需要網路，之後可離線使用已快取的模型。

## GPU 加速設定

按右上角「設定」可分別啟用 CLIP、Face 與 OCR 的 GPU 選項。CLIP 與 Face 在 Windows 使用 ONNX Runtime DirectML；若 DirectML 無法載入或模型工作階段無法建立，應用程式會自動退回 CPU，並在設定、建立索引進度與完成訊息顯示實際後端與原因。縮圖目前由 Rust `image` 函式庫處理，因此解碼、縮放與 JPEG 編碼固定使用 CPU；設定中的縮圖 GPU 選項會清楚標示為不可用，直到導入真正的 GPU 影像後端。設定保存在 app-data 的 `settings.json`，變更後需要重新選擇照片資料夾以重建索引。

專案內的 `vendor/fastembed-patched` 是 FastEmbed 4.9.1 的可重現副本，額外為 DirectML 工作階段關閉 ONNX Runtime 不支援的 memory pattern 與平行執行設定；這能避免勾選 GPU 後模型初始化失敗卻靜默退回 CPU。

OCR 仍使用外部 Tesseract。勾選「OCR OpenCL（實驗性）」會為 Tesseract 子程序設定 `TESSERACT_OPENCL_DEVICE=1`，但只有自行編譯且啟用 OpenCL 的 Tesseract 才會使用 GPU；官方不建議一般使用者依賴這項實驗性功能，而且它不一定比 CPU 快。標準 Windows 安裝版通常仍會使用 CPU。

- [ONNX Runtime DirectML 說明](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [Tesseract OpenCL 說明](https://github.com/tesseract-ocr/tessdoc/blob/main/TesseractOpenCL.md)

## Face Recognition

Face Recognition 使用本機 ONNX Runtime 執行 SCRFD 偵測器與 ArcFace／MobileFaceNet 辨識器；Rust 前後處理流程依 InsightFace 模型格式實作。首次建立索引時會從 `WePrompt/buffalo_sc` 下載約 16 MB 的兩個 ONNX 模型，之後從 app-data 快取離線載入。每張臉會經五點定位與對齊後產生 512 維向量，cosine similarity 達到 `0.45` 才會歸入同一人物群組。

在「標記人物」輸入姓名並儲存後，姓名與該群組的代表向量會寫入 app-data 的 `people.json`；下次重新掃描時會自動辨識已標記人物。將姓名清空再儲存即可移除該人物標記。若要使用自行取得授權的模型，可設定：

```powershell
$env:LOCAL_LENS_FACE_DETECTOR = "D:\models\det_500m.onnx"
$env:LOCAL_LENS_FACE_RECOGNIZER = "D:\models\w600k_mbf.onnx"
npm run tauri dev
```

注意：人臉向量屬敏感生物特徵資料。此版本只保存在本機；若要商業散布，仍應確認所選預訓練模型的資料集與權利條款，並加入一次清除全部人物資料及關閉人臉功能的設定。
