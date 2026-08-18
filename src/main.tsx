import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import "./styles.css";

type ImageRecord = {
  id: string;
  path: string;
  filename: string;
  modified_at: string;
  width?: number;
  height?: number;
  thumbnail: string;
  ocr_text: string;
  people: string[];
  score: number;
};

type ScanResult = { root: string; indexed: number; skipped: number; ocr_available: boolean; semantic_available: boolean };
type ScanProgress = { processed: number; total: number; indexed: number; skipped: number; ocr_available: boolean; semantic_available: boolean };

function App() {
  const [query, setQuery] = useState("");
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [folder, setFolder] = useState("");
  const [status, setStatus] = useState("選擇一個照片資料夾來建立本機索引。");
  const [busy, setBusy] = useState(false);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [peopleOnly, setPeopleOnly] = useState(false);

  const visibleImages = useMemo(
    () => (peopleOnly ? images.filter((image) => image.people.length > 0) : images),
    [images, peopleOnly]
  );

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void listen<ScanProgress>("scan-progress", (event) => setScanProgress(event.payload))
      .then((stopListening) => { unlisten = stopListening; });
    return () => unlisten?.();
  }, []);

  async function chooseAndScan() {
    const selected = await open({ directory: true, multiple: false, title: "選擇照片資料夾" });
    if (!selected || Array.isArray(selected)) return;
    setBusy(true);
    setScanProgress({ processed: 0, total: 0, indexed: 0, skipped: 0, ocr_available: false, semantic_available: false });
    setStatus("正在產生縮圖、OCR 與語意向量（首次使用可能下載模型）…");
    try {
      const result = await invoke<ScanResult>("scan_folder", { folder: selected });
      setFolder(result.root);
      setQuery("");
      const indexed = await invoke<ImageRecord[]>("search_images", { query: "" });
      setImages(indexed);
      setStatus(`已索引 ${result.indexed} 張圖片，先載入最多 200 張預覽；${result.ocr_available ? "OCR 已啟用" : "找不到 Tesseract，僅使用檔名搜尋"}；${result.semantic_available ? "Semantic Search 已啟用" : "語意模型未就緒，僅使用文字搜尋"}${result.skipped ? `；略過 ${result.skipped} 個無法讀取的檔案` : ""}。`);
    } catch (error) {
      setStatus(`建立索引失敗：${String(error)}`);
    } finally {
      setBusy(false);
      setScanProgress(null);
    }
  }

  async function search(nextQuery = query) {
    setBusy(true);
    setStatus(nextQuery.trim() ? "正在載入語意模型並搜尋…" : "正在載入已索引圖片…");
    try {
      const result = await invoke<ImageRecord[]>("search_images", { query: nextQuery });
      setImages(result);
      setStatus(nextQuery.trim()
        ? result.length > 0
          ? `顯示 ${result.length} 張高相關圖片（最多 60 張）。`
          : "找不到高信心結果，請改用更具體的描述；目前語意模型以英文查詢較準確。"
        : `顯示最多 200 張已索引圖片（目前 ${result.length} 張）。`);
    } catch (error) {
      setStatus(`搜尋失敗：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">PRIVATE · LOCAL · WINDOWS</p>
          <h1>Local Lens</h1>
          <p className="subtitle">用文字、OCR 與人名，找回你的照片。</p>
        </div>
        <button className="folder-button" onClick={chooseAndScan} disabled={busy}>
          {busy ? "處理中…" : "選擇照片資料夾"}
        </button>
      </header>

      <section className="search-panel">
        <form onSubmit={(event) => { event.preventDefault(); void search(); }}>
          <span className="search-icon">⌕</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="例如：海邊的狗、發票、和小明的合照"
            disabled={!folder || busy}
          />
          <button type="submit" disabled={!folder || busy}>搜尋</button>
        </form>
        <div className="filters">
          <button className={!peopleOnly ? "chip active" : "chip"} onClick={() => setPeopleOnly(false)}>全部</button>
          <button className={peopleOnly ? "chip active" : "chip"} onClick={() => setPeopleOnly(true)}>已標記人物</button>
          {folder && <span className="folder-name" title={folder}>索引位置：{folder}</span>}
        </div>
      </section>

      {scanProgress && (
        <section className="progress-panel" aria-live="polite">
          <div className="progress-label">
            <span>正在建立圖片索引</span>
            <span>{scanProgress.total ? `${scanProgress.processed} / ${scanProgress.total}` : "正在統計圖片…"}</span>
          </div>
          <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={scanProgress.total || undefined} aria-valuenow={scanProgress.processed}>
            <div className="progress-fill" style={{ width: scanProgress.total ? `${Math.round((scanProgress.processed / scanProgress.total) * 100)}%` : "3%" }} />
          </div>
          <p>已建立 {scanProgress.indexed} 張縮圖；{scanProgress.ocr_available ? "同步進行 OCR" : "OCR 未啟用"}；{scanProgress.semantic_available ? "同步建立語意向量" : "語意模型未就緒"}{scanProgress.skipped ? `；略過 ${scanProgress.skipped} 個無法讀取的檔案` : ""}。</p>
        </section>
      )}

      <section className="results" aria-live="polite">
        <div className="result-heading"><span>{status}</span><span>{visibleImages.length} results</span></div>
        {visibleImages.length === 0 ? (
          <div className="empty"><div>◫</div><h2>還沒有可顯示的圖片</h2><p>選擇一個包含 JPG、PNG、WebP、GIF 或 BMP 的資料夾開始。</p></div>
        ) : (
          <div className="gallery">
            {visibleImages.map((image) => (
              <article
                className="card"
                key={image.id}
                onDoubleClick={() => void revealItemInDir(image.path).catch((error) => setStatus(`無法在系統檔案管理員顯示圖片：${String(error)}`))}
                title="雙擊以在系統檔案管理員顯示"
              >
                <img src={image.thumbnail} alt={image.filename} loading="lazy" />
                <div className="card-info">
                  <strong>{image.filename}</strong>
                  <span>{image.width && image.height ? `${image.width} × ${image.height}` : "圖片"}</span>
                  {image.people.length > 0 && <small>{image.people.join("、")}</small>}
                  {image.ocr_text && <small className="ocr-text" title={image.ocr_text}>OCR：{image.ocr_text.slice(0, 100)}</small>}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
      <footer>圖片不會上傳。此 MVP 已具備檔名、OCR 與 Semantic Search；人臉標記會沿用同一份本機索引。</footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
