# Local Lens

一個以 Tauri 2、React 與 Rust 製作的 Windows 本機圖片搜尋 MVP。照片會在使用者選擇的資料夾中讀取，縮圖與暫存索引只保留在應用程式記憶體；不會上傳圖片。

## 已完成的 MVP

- 選擇資料夾後遞迴掃描 JPG、JPEG、PNG、WebP、GIF、BMP
- 在 Rust 背景端產生縮圖，避免結果頁直接讀取全尺寸照片
- 依檔名搜尋，支援多個關鍵字的結果排序
- 若系統可找到 Tesseract，建立索引時會抽取 OCR 文字並納入搜尋
- 格狀預覽、圖片尺寸與雙擊於檔案總管開啟原圖
- 預留 `ocr_text`、`people`、`score` 欄位，讓 OCR、人臉辨識及語意搜尋共用同一個索引介面
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

## 接下來的 AI 實作順序

1. 增加 SQLite：保存路徑、檔案修改時間、縮圖快取與索引版本，讓重新開啟應用後不用重掃。
2. 將目前記憶體內的 OCR 結果移入 SQLite，並加上 SQLite FTS5 文字索引。
3. ONNX Runtime worker：產生多語言圖片／文字 embedding，改由向量相似度處理自然語言搜尋。
4. 人臉偵測、特徵向量與「待確認的人臉」畫面；僅在使用者確認後才把臉標記為姓名。

注意：人臉向量屬敏感資料。正式版應提供清除人物資料、重新建立索引、關閉人臉功能，並在使用前明確告知使用者。
