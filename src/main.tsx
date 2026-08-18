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

type FaceGroup = { id: string; name?: string; face_count: number; image_count: number; preview: string };
type ScanResult = { root: string; indexed: number; reused: number; skipped: number; ocr_available: boolean; semantic_available: boolean; face_available: boolean; faces_detected: number; face_groups: number; clip_gpu_active: boolean; face_gpu_active: boolean; thumbnail_gpu_requested: boolean; thumbnail_gpu_active: boolean; ocr_gpu_requested: boolean; ocr_gpu_active: boolean; gpu_warning?: string | null };
type ScanProgress = { processed: number; total: number; eta_seconds: number | null; indexed: number; reused: number; skipped: number; ocr_available: boolean; semantic_available: boolean; face_available: boolean; faces_detected: number; clip_gpu_active: boolean; face_gpu_active: boolean; thumbnail_gpu_requested: boolean; thumbnail_gpu_active: boolean; ocr_gpu_requested: boolean; ocr_gpu_active: boolean; gpu_warning?: string | null };
type ModelSettings = { max_indexed_images: number | null; index_batch_size: number; thumbnail_gpu: boolean; ocr_gpu: boolean; clip_gpu: boolean; face_gpu: boolean };
type SettingsInfo = { settings: ModelSettings; directml_available: boolean; directml_error?: string | null; thumbnail_gpu_available: boolean; ocr_gpu_experimental: boolean };

const DEFAULT_MODEL_SETTINGS: ModelSettings = { max_indexed_images: 3000, index_batch_size: 50, thumbnail_gpu: false, ocr_gpu: false, clip_gpu: false, face_gpu: false };

function formatEta(seconds: number | null): string {
  if (seconds === null) return "正在估算完成時間…";
  if (seconds <= 0) return "即將完成";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  const duration = hours > 0
    ? `${hours} 小時 ${minutes} 分鐘`
    : minutes > 0
      ? `${minutes} 分鐘 ${remainingSeconds} 秒`
      : `${remainingSeconds} 秒`;
  const finishAt = new Date(Date.now() + seconds * 1000).toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" });
  return `預估剩餘 ${duration}（約 ${finishAt} 完成）`;
}

function App() {
  const [query, setQuery] = useState("");
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [folder, setFolder] = useState("");
  const [status, setStatus] = useState("選擇一個照片資料夾來建立本機索引。");
  const [busy, setBusy] = useState(false);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [scanPaused, setScanPaused] = useState(false);
  const [peopleOnly, setPeopleOnly] = useState(false);
  const [faceGroups, setFaceGroups] = useState<FaceGroup[]>([]);
  const [faceNames, setFaceNames] = useState<Record<string, string>>({});
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsInfo, setSettingsInfo] = useState<SettingsInfo | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<ModelSettings>(DEFAULT_MODEL_SETTINGS);

  const visibleImages = useMemo(
    () => (peopleOnly ? images.filter((image) => image.people.length > 0) : images),
    [images, peopleOnly]
  );

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void listen<ScanProgress>("scan-progress", (event) => setScanProgress(event.payload))
      .then((stopListening) => { unlisten = stopListening; });
    void invoke<SettingsInfo>("get_model_settings")
      .then((info) => {
        setSettingsInfo(info);
        setSettingsDraft(info.settings);
      })
      .catch((error) => setStatus(`無法讀取模型設定：${String(error)}`));
    return () => unlisten?.();
  }, []);

  async function saveModelSettings() {
    setBusy(true);
    setScanPaused(false);
    try {
      const info = await invoke<SettingsInfo>("update_model_settings", { settings: settingsDraft });
      setSettingsInfo(info);
      setSettingsDraft(info.settings);
      const wantsDirectMl = info.settings.clip_gpu || info.settings.face_gpu;
      setStatus(wantsDirectMl && !info.directml_available
        ? `設定已儲存，但 CLIP 與 Face 將自動使用 CPU：${info.directml_error ?? "找不到可用的 DirectML"}。`
        : "模型加速設定已儲存，請重新選擇照片資料夾以重建索引並套用。"
      );
    } catch (error) {
      setStatus(`儲存模型設定失敗：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function chooseAndScan() {
    const selected = await open({ directory: true, multiple: false, title: "選擇照片資料夾" });
    if (!selected || Array.isArray(selected)) return;
    setBusy(true);
    setScanProgress({ processed: 0, total: 0, eta_seconds: null, indexed: 0, reused: 0, skipped: 0, ocr_available: false, semantic_available: false, face_available: false, faces_detected: 0, clip_gpu_active: false, face_gpu_active: false, thumbnail_gpu_requested: settingsInfo?.settings.thumbnail_gpu ?? false, thumbnail_gpu_active: false, ocr_gpu_requested: settingsInfo?.settings.ocr_gpu ?? false, ocr_gpu_active: false, gpu_warning: null });
    setStatus("正在產生縮圖、OCR、語意與人臉向量（首次使用可能下載模型）…");
    try {
      const result = await invoke<ScanResult>("scan_folder", { folder: selected });
      setFolder(result.root);
      setQuery("");
      const [indexed, groups] = await Promise.all([
        invoke<ImageRecord[]>("search_images", { query: "" }),
        invoke<FaceGroup[]>("list_face_groups")
      ]);
      setImages(indexed);
      setFaceGroups(groups);
      setFaceNames(Object.fromEntries(groups.map((group) => [group.id, group.name ?? ""])));
      const gpuWarning = result.gpu_warning ? `；⚠ ${result.gpu_warning}` : "";
      setStatus(`已索引 ${result.indexed} 張圖片${result.reused ? `（重用 ${result.reused} 筆 SQLite 快取）` : ""}，先載入最多 200 張預覽；縮圖${result.thumbnail_gpu_requested ? `（${result.thumbnail_gpu_active ? "GPU" : "CPU，GPU 不可用"}）` : "使用一般處理流程"}；${result.ocr_available ? `OCR 已啟用${result.ocr_gpu_requested ? `（${result.ocr_gpu_active ? "OpenCL GPU" : "CPU，OpenCL 不可用"}）` : ""}` : "找不到 Tesseract，僅使用檔名搜尋"}；${result.semantic_available ? `Semantic Search 已啟用${result.clip_gpu_active ? "（GPU）" : "（CPU）"}` : "語意模型未就緒，僅使用文字搜尋"}；${result.face_available ? `偵測到 ${result.faces_detected} 張臉、${result.face_groups} 個人物群組${result.face_gpu_active ? "（GPU）" : "（CPU）"}` : "人臉模型未就緒"}${result.skipped ? `；略過 ${result.skipped} 個無法讀取的檔案` : ""}${gpuWarning}。`);
    } catch (error) {
      setStatus(`建立索引失敗：${String(error)}`);
    } finally {
      setBusy(false);
      setScanProgress(null);
      setScanPaused(false);
    }
  }

  async function toggleScanPause() {
    try {
      if (scanPaused) {
        await invoke("resume_scan");
        setScanPaused(false);
        setStatus("已繼續建立圖片索引…");
      } else {
        await invoke("pause_scan");
        setScanPaused(true);
        setStatus("建立圖片索引已暫停；可隨時繼續。");
      }
    } catch (error) {
      setStatus(`無法${scanPaused ? "繼續" : "暫停"}索引：${String(error)}`);
    }
  }

  async function saveFaceName(group: FaceGroup) {
    const name = (faceNames[group.id] ?? "").trim();
    setBusy(true);
    try {
      const groups = await invoke<FaceGroup[]>("label_face_group", { groupId: group.id, name });
      setFaceGroups(groups);
      setFaceNames(Object.fromEntries(groups.map((item) => [item.id, item.name ?? ""])));
      const refreshed = await invoke<ImageRecord[]>("search_images", { query });
      setImages(refreshed);
      setStatus(name ? `已將這組人臉標記為「${name}」，現在可以用姓名搜尋。` : "已清除此人物群組的姓名標記。");
    } catch (error) {
      setStatus(`儲存人物姓名失敗：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  function searchPerson(name: string) {
    setQuery(name);
    void search(name);
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
        <div className="header-actions">
          <button className="settings-button" onClick={() => setSettingsOpen((open) => !open)} disabled={busy} aria-expanded={settingsOpen}>
            ⚙ 設定
          </button>
          <button className="folder-button" onClick={chooseAndScan} disabled={busy}>
            {busy ? "處理中…" : "選擇照片資料夾"}
          </button>
        </div>
      </header>

      {settingsOpen && (
        <section className="settings-panel" aria-label="模型加速設定">
          <div className="settings-heading">
            <div><span className="eyebrow">MODEL ACCELERATION</span><h2>GPU 加速</h2></div>
            <span className={settingsInfo?.directml_available ? "backend-badge available" : "backend-badge"}>
              DirectML {settingsInfo?.directml_available ? "可用" : "未偵測到"}
            </span>
          </div>
          {settingsInfo?.directml_error && <p className="settings-error">DirectML：{settingsInfo.directml_error}</p>}
          <div className="settings-grid">
            <label className="setting-option">
              <input type="checkbox" checked={settingsDraft.max_indexed_images === null} onChange={(event) => setSettingsDraft((current) => ({ ...current, max_indexed_images: event.target.checked ? null : (current.max_indexed_images ?? 3000) }))} disabled={busy} />
              <span><strong>掃描圖片數量無上限</strong><small>啟用後會掃描資料夾內所有支援的圖片；大量照片首次建立索引可能需要較長時間與較多儲存空間。</small></span>
            </label>
            {settingsDraft.max_indexed_images !== null && (
              <label className="setting-option limit-option">
                <span><strong>每個資料夾的圖片上限</strong><small>只會處理掃描順序中的前幾張圖片。</small></span>
                <input className="limit-input" type="number" min="1" step="1" value={settingsDraft.max_indexed_images} onChange={(event) => setSettingsDraft((current) => ({ ...current, max_indexed_images: Math.max(1, Number.parseInt(event.target.value, 10) || 1) }))} disabled={busy} />
              </label>
            )}
            <label className="setting-option limit-option">
              <span><strong>SQLite 每批寫入張數</strong><small>每完成一批就提交一次快取；數字越小越能保留中斷前的進度，但寫入次數會增加。</small></span>
              <input className="limit-input" type="number" min="1" step="1" value={settingsDraft.index_batch_size} onChange={(event) => setSettingsDraft((current) => ({ ...current, index_batch_size: Math.max(1, Number.parseInt(event.target.value, 10) || 1) }))} disabled={busy} />
            </label>
            <label className="setting-option">
              <input type="checkbox" checked={settingsDraft.thumbnail_gpu} onChange={(event) => setSettingsDraft((current) => ({ ...current, thumbnail_gpu: event.target.checked }))} disabled={busy || !settingsInfo?.thumbnail_gpu_available} />
              <span><strong>建立縮圖時使用 GPU 加速</strong><small>{settingsInfo?.thumbnail_gpu_available ? "在下一次建立索引時套用。" : "目前版本沒有 GPU 縮圖後端，因此固定使用 CPU。"}</small></span>
            </label>
            <label className="setting-option">
              <input type="checkbox" checked={settingsDraft.clip_gpu} onChange={(event) => setSettingsDraft((current) => ({ ...current, clip_gpu: event.target.checked }))} disabled={busy} />
              <span><strong>CLIP Semantic Search</strong><small>使用 DirectML 建立圖片與查詢向量；無法啟用時自動退回 CPU。</small></span>
            </label>
            <label className="setting-option">
              <input type="checkbox" checked={settingsDraft.face_gpu} onChange={(event) => setSettingsDraft((current) => ({ ...current, face_gpu: event.target.checked }))} disabled={busy} />
              <span><strong>Face Recognition</strong><small>使用 DirectML 執行 SCRFD 與 ArcFace；無法啟用時自動退回 CPU。</small></span>
            </label>
            <label className="setting-option experimental">
              <input type="checkbox" checked={settingsDraft.ocr_gpu} onChange={(event) => setSettingsDraft((current) => ({ ...current, ocr_gpu: event.target.checked }))} disabled={busy} />
              <span><strong>OCR OpenCL（實驗性）</strong><small>僅自行編譯且啟用 OpenCL 的 Tesseract 有效；一般安裝版可能仍使用 CPU。</small></span>
            </label>
          </div>
          <div className="settings-footer">
            <p>GPU 選項會在下一次建立索引時套用；原有索引不會自動重算。完成後請以進度與結果訊息中的「GPU／CPU」為準；若模型初始化失敗，會顯示退回 CPU 的原因。</p>
            <button onClick={() => void saveModelSettings()} disabled={busy}>儲存設定</button>
          </div>
        </section>
      )}

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
            <span>{scanPaused ? "已暫停" : (scanProgress.total ? `${scanProgress.processed} / ${scanProgress.total}` : "正在統計圖片…")}</span>
          </div>
          <div className="progress-eta">{scanPaused ? `暫停中 · ${formatEta(scanProgress.eta_seconds)}` : formatEta(scanProgress.eta_seconds)}</div>
          <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={scanProgress.total || undefined} aria-valuenow={scanProgress.processed}>
            <div className="progress-fill" style={{ width: scanProgress.total ? `${Math.round((scanProgress.processed / scanProgress.total) * 100)}%` : "3%" }} />
          </div>
          <p>已建立 {scanProgress.indexed} 張縮圖{scanProgress.reused ? `（重用 ${scanProgress.reused} 筆 SQLite 快取）` : ""}{scanProgress.thumbnail_gpu_requested ? `（${scanProgress.thumbnail_gpu_active ? "GPU" : "CPU，GPU 不可用"}）` : ""}；{scanProgress.ocr_available ? `同步進行 OCR${scanProgress.ocr_gpu_requested ? `（${scanProgress.ocr_gpu_active ? "OpenCL GPU" : "CPU，OpenCL 不可用"}）` : ""}` : "OCR 未啟用"}；{scanProgress.semantic_available ? `同步建立語意向量${scanProgress.clip_gpu_active ? "（GPU）" : "（CPU）"}` : "語意模型未就緒"}；{scanProgress.face_available ? `已偵測 ${scanProgress.faces_detected} 張臉${scanProgress.face_gpu_active ? "（GPU）" : "（CPU）"}` : "人臉模型未就緒"}{scanProgress.skipped ? `；略過 ${scanProgress.skipped} 個無法讀取的檔案` : ""}{scanProgress.gpu_warning ? `；⚠ ${scanProgress.gpu_warning}` : ""}。</p>
          <button
            className={`scan-control-button ${scanPaused ? "resume" : "pause"}`}
            type="button"
            onClick={() => void toggleScanPause()}
            aria-label={scanPaused ? "繼續索引" : "暫停索引"}
          >
            <svg className="scan-control-icon" viewBox="0 0 24 24" aria-hidden="true">
              {scanPaused ? <path d="M8 5.5v13l10-6.5z" /> : <><rect x="6" y="5" width="4.5" height="14" rx="1" /><rect x="13.5" y="5" width="4.5" height="14" rx="1" /></>}
            </svg>
            <span>{scanPaused ? "繼續索引" : "暫停索引"}</span>
          </button>
        </section>
      )}

      {faceGroups.length > 0 && (
        <section className="people-panel" aria-label="人物群組">
          <div className="people-heading">
            <div><span className="eyebrow">PEOPLE</span><h2>標記人物</h2></div>
            <p>系統會把相似臉分組；輸入姓名後即可用人名搜尋照片。</p>
          </div>
          <div className="people-list">
            {faceGroups.map((group) => (
              <article className="face-group" key={group.id}>
                <img src={group.preview} alt={group.name ? `${group.name}的人臉` : "未命名人臉"} />
                <div className="face-group-body">
                  <input
                    className="face-name-input"
                    value={faceNames[group.id] ?? ""}
                    onChange={(event) => setFaceNames((current) => ({ ...current, [group.id]: event.target.value }))}
                    onKeyDown={(event) => { if (event.key === "Enter") void saveFaceName(group); }}
                    placeholder="輸入姓名"
                    aria-label="人物姓名"
                    disabled={busy}
                  />
                  <small>{group.face_count} 張臉 · {group.image_count} 張照片</small>
                  <div className="face-actions">
                    <button onClick={() => void saveFaceName(group)} disabled={busy}>儲存</button>
                    {group.name && <button className="secondary" onClick={() => searchPerson(group.name!)} disabled={busy}>搜尋</button>}
                  </div>
                </div>
              </article>
            ))}
          </div>
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
      <footer>圖片、人臉向量與姓名標記都保留在本機，不會上傳。</footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
