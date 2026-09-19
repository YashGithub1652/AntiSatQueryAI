/**
 * SatQuery AI — Real API Client Controller (ISRO SIH26167)
 * Connects to the real FastAPI backend with session-based uploads,
 * LangGraph agent, and WebSocket live trace streaming.
 */
document.addEventListener("DOMContentLoaded", () => {

  // ---------- State ----------
  let currentSessionId = null;        // Real session ID from backend
  let activeWebSocket = null;         // Live trace WebSocket
  let currentMode = "bi_temporal";
  let currentLanguage = "en";
  let currentImages = null;           // Loaded from real backend upload
  let currentScenarioMeta = null;     // Compat shim for legacy UI refs
  let currentScenarioId = null;       // Selected Indian benchmark scenario ID
  let allScenarios = [];              // Catalog loaded from backend
  let lastAnalysisResult = null;
  let lastAnalysisMode = null;
  let isDraggingSlider = null;
  let activeView = "workspace";
  let canvasZoom = 1;
  let modelsRegistryCache = null;
  let uploadedFiles = [];             // Files staged for upload
  const runHistory = [];
  const BASE_URL = window.location.origin;

  const viewModeMap = {
    "workspace": "bi_temporal",
    "map-view": null,
    "change-detection": "bi_temporal",
    "sar-analysis": "sar_fusion",
    "visual-grounding": "grounding",
    "history": null,
    "models-data": null,
    "export-report": null
  };

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);

  const scenarioDropdownList = $("demo-scenes-list");
  const mapScenarioGrid = $("map-scenarios-grid");
  const queryInput = $("query-input");
  const queryChipsContainer = $("query-suggestions-container");
  const analyzeBtn = $("analyze-btn");
  const voiceBtn = $("voice-btn");
  const langToggleBtn = $("lang-toggle-btn");
  const currentLangText = $("current-lang-text");
  const chatLog = $("chat-log");
  const modeChipRow = $("mode-chip-row");
  const querySceneLabel = $("query-scenario-label");

  const canvasSceneSubtitle = $("canvas-scene-subtitle");
  const canvasSceneLabel = $("canvas-scene-label");
  const canvasSensorLabel = $("canvas-sensor-label");
  const canvasCoords = $("canvas-coords");
  const canvasArea = $("canvas-area");
  const canvasBands = $("canvas-bands");

  const imgT1 = $("img-t1"), imgT2 = $("img-t2"), imgSar = $("img-sar"), imgFused = $("img-fused"), imgChange = $("img-change");
  const sliderWrapper = $("slider-wrapper"), sliderDivider = $("slider-divider");
  const canvasContainer = $("canvas-container"), bboxContainer = $("bbox-container");
  const badgeBoxes = $("badge-boxes"), badgeSegmentation = $("badge-segmentation");

  const canvasToggles = document.querySelectorAll(".canvas-toggle");
  const zoomInBtn = $("zoom-in-btn"), zoomOutBtn = $("zoom-out-btn"), layersCycleBtn = $("layers-cycle-btn");

  const aiAnswerText = $("ai-answer-text");
  const confidenceIndicator = $("confidence-indicator");
  const detectedClasses = $("detected-classes");
  const evidenceCitations = $("evidence-citations");
  const traceContainer = $("trace-container");
  const traceStatusBadge = $("trace-status-badge");
  const replayTraceBtn = $("replay-trace-btn");

  const loadingOverlay = $("loading-overlay");
  const loadingStatusText = $("loading-status-text");

  const activeSceneName = $("active-scene-name");
  const activeSceneSub = $("active-scene-sub");
  const sceneThumb1 = $("scene-thumb-1"), sceneThumb2 = $("scene-thumb-2");

  const uploadDropzone = $("upload-dropzone");
  const uploadFileInput = $("upload-file-input");
  const uploadBrowseBtn = $("upload-browse-btn");
  const uploadStatusBadge = $("upload-status-badge");

  // Change detection view
  const cdImgT1 = $("cd-img-t1"), cdImgT2 = $("cd-img-t2");
  const cdSliderWrapper = $("cd-slider-wrapper"), cdSliderDivider = $("cd-slider-divider");
  const cdCanvasContainer = $("cd-canvas-container");
  const cdLabelBefore = $("cd-label-before"), cdLabelAfter = $("cd-label-after");
  const cdMetrics = $("cd-metrics");

  // SAR view
  const sarVvBars = $("sar-vv-bars"), sarVhBars = $("sar-vh-bars");
  const sarVvDb = $("sar-vv-db"), sarVhDb = $("sar-vh-db");
  const sarInterpretationText = $("sar-interpretation-text");

  // Grounding view
  const groundingList = $("grounding-list");

  // History view
  const historyList = $("history-list");

  // Models view
  const modelsList = $("models-list");

  // Export view
  const reportPreviewThumb = $("report-preview-thumb");
  const reportPreviewQuery = $("report-preview-query");
  const reportPreviewAnswer = $("report-preview-answer");
  const reportPreviewMeta = $("report-preview-meta");
  const exportPdfBtn = $("export-pdf-btn");
  const exportGeojsonBtn = $("export-geojson-btn");
  const exportTraceBtn = $("export-trace-btn");

  // Modal
  const infoBtn = $("info-btn");
  const registryModal = $("registry-modal");
  const closeModalBtn = $("close-modal-btn");

  // ---------- Init ----------
  async function init() {
    setupSliderDrag(canvasContainer, sliderWrapper, sliderDivider, () => "workspace");
    setupSliderDrag(cdCanvasContainer, cdSliderWrapper, cdSliderDivider, () => "cd");
    setupEventListeners();
    await loadModelsRegistry();
    await loadScenarios();
    appendChatMessage("agent",
      "Welcome to SatQuery AI (SIH26167). Select an Indian benchmark scenario from the list below or upload custom GeoTIFF files to begin.", null);
  }

  // ── Scenarios Preset Integration ───────────────────────────
  async function loadScenarios() {
    try {
      const res = await fetch(`${BASE_URL}/api/v1/scenarios`);
      if (!res.ok) return;
      const data = await res.json();
      allScenarios = data.scenarios || [];
      renderScenariosDropdown(allScenarios);
      renderMapScenariosGrid(allScenarios);
      if (allScenarios.length > 0 && !currentSessionId) {
        await selectScenario(allScenarios[0].id);
      }
    } catch (e) {
      console.warn("Could not load scenarios catalog:", e);
    }
  }

  function renderScenariosDropdown(scenarios) {
    if (!scenarioDropdownList) return;
    scenarioDropdownList.innerHTML = "";
    scenarios.forEach(sc => {
      const item = document.createElement("div");
      item.className = "demo-scene-item";
      item.dataset.scenarioId = sc.id;
      item.innerHTML = `
        <div class="demo-scene-title">${sc.title}</div>
        <div class="demo-scene-location">${sc.location} · ${sc.default_mode || 'multi-modal'}</div>
      `;
      item.addEventListener("click", () => selectScenario(sc.id));
      scenarioDropdownList.appendChild(item);
    });
  }

  function renderMapScenariosGrid(scenarios) {
    if (!mapScenarioGrid) return;
    mapScenarioGrid.innerHTML = "";
    scenarios.forEach(sc => {
      const card = document.createElement("div");
      card.className = "map-scenario-card";
      card.dataset.scenarioId = sc.id;
      card.innerHTML = `
        <div class="map-card-header">
          <span class="badge badge-blue">${sc.default_mode || 'EO'}</span>
          <span class="map-card-coords">${sc.coordinates || ''}</span>
        </div>
        <div class="map-card-title">${sc.title}</div>
        <div class="map-card-desc">${sc.location} (${sc.crs})</div>
        <button class="btn btn-secondary btn-sm mt-2" style="margin-top:8px; cursor:pointer;">Load Scene</button>
      `;
      card.addEventListener("click", () => {
        selectScenario(sc.id);
        switchView("workspace");
      });
      mapScenarioGrid.appendChild(card);
    });
  }

  async function selectScenario(scenarioId) {
    currentScenarioId = scenarioId;
    currentSessionId = null; // Clear session to use scenario mode
    const sc = allScenarios.find(s => s.id === scenarioId);
    if (!sc) return;
    currentScenarioMeta = sc;

    // Highlight active in dropdown
    if (scenarioDropdownList) {
      scenarioDropdownList.querySelectorAll(".demo-scene-item").forEach(el => {
        el.classList.toggle("active", el.dataset.scenarioId === scenarioId);
      });
    }

    if (querySceneLabel) querySceneLabel.textContent = sc.title;
    if (activeSceneName) activeSceneName.textContent = sc.title;
    if (activeSceneSub) activeSceneSub.textContent = `${sc.location} · ${sc.crs}`;

    // Update query chips from scenario suggestions
    if (sc.query_suggestions && sc.query_suggestions.length) {
      queryChipsContainer.innerHTML = "";
      sc.query_suggestions.forEach(q => {
        const chip = document.createElement("button");
        chip.className = "query-chip";
        chip.textContent = q;
        chip.addEventListener("click", () => runAnalysis(q));
        queryChipsContainer.appendChild(chip);
      });
      queryInput.value = sc.query_suggestions[0];
    }

    if (sc.default_mode) {
      currentMode = sc.default_mode;
      syncModeChips();
    }

    // Fetch imagery
    try {
      showLoading(`Loading calibrated imagery for ${sc.title}...`);
      const imgRes = await fetch(`${BASE_URL}/api/v1/scenarios/${scenarioId}/images`);
      if (imgRes.ok) {
        const imgData = await imgRes.json();
        const imgs = imgData.images || {};
        currentImages = imgs;
        if (imgs.image_t1) {
          if (imgT1) imgT1.src = imgs.image_t1;
          if (cdImgT1) cdImgT1.src = imgs.image_t1;
          if (sceneThumb1) sceneThumb1.src = imgs.image_t1;
        }
        if (imgs.image_t2) {
          if (imgT2) imgT2.src = imgs.image_t2;
          if (cdImgT2) cdImgT2.src = imgs.image_t2;
          if (sceneThumb2) sceneThumb2.src = imgs.image_t2;
        }
        if (imgs.image_sar && imgSar) imgSar.src = imgs.image_sar;
        if (imgs.image_fused && imgFused) imgFused.src = imgs.image_fused;
        if (imgs.image_change && imgChange) imgChange.src = imgs.image_change;

        if (canvasSceneLabel) canvasSceneLabel.innerHTML = `SCENE <b>${sc.title}</b>`;
        if (canvasSensorLabel) canvasSensorLabel.textContent = `${sc.optical_bands || 'MSI'} · ${sc.resolution || '10m'}`;
        if (canvasCoords) canvasCoords.textContent = sc.coordinates || '--';
        if (canvasArea) canvasArea.textContent = `CRS ${sc.crs || '--'}`;
        if (canvasBands) canvasBands.textContent = sc.sar_bands || 'Bands: Optical + SAR';

        setCanvasLayer("split");
      }
    } catch (e) {
      console.warn("Could not load scenario images:", e);
    } finally {
      hideLoading();
    }
  }

  // ── Upload: single or pair ────────────────────────────────
  async function uploadImages(files) {
    if (!files || files.length === 0) return;
    showLoading("Uploading & parsing geospatial metadata...");
    uploadStatusBadge.textContent = `● Uploading ${files.length} file(s)...`;

    try {
      let sessionData;
      if (files.length === 1) {
        const form = new FormData();
        form.append("file", files[0]);
        const res = await fetch(`${BASE_URL}/api/v1/upload/image`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await res.text());
        sessionData = await res.json();
        currentSessionId = sessionData.session_id;
        currentImages = { image_t1: sessionData.rgb_preview_b64, image_t2: null };
        _applySessionMetaToUI(sessionData, null);
      } else {
        const form = new FormData();
        form.append("file1", files[0]);
        form.append("file2", files[1]);
        const res = await fetch(`${BASE_URL}/api/v1/upload/pair`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await res.text());
        sessionData = await res.json();
        currentSessionId = sessionData.session_id;
        currentImages = {
          image_t1: sessionData.image_1?.rgb_preview_b64,
          image_t2: sessionData.image_2?.rgb_preview_b64,
        };
        _applySessionMetaToUI(sessionData.image_1, sessionData.image_2);

        // Show coregistration status
        const coreg = sessionData.coregistration || {};
        if (!coreg.passed && !coreg.skipped) {
          appendChatMessage("agent",
            `⚠️ Co-registration check: ${coreg.message || 'Alignment warning'}. ` +
            `CRS mismatch: ${coreg.crs_match === false ? 'Yes' : 'No'}. ` +
            "Analysis will proceed with best-effort alignment.", null);
        }
      }

      uploadStatusBadge.textContent = `● Ready · Session ${currentSessionId.slice(0, 8)}`;
      appendChatMessage("agent",
        `✓ ${files.length} image(s) loaded. Session: ${currentSessionId.slice(0, 8)}. ` +
        `Ask your question below.`, null);
      _connectWebSocket();
      hideLoading();

    } catch (e) {
      console.error("Upload error:", e);
      uploadStatusBadge.textContent = "● Upload Failed";
      appendChatMessage("agent", `Upload failed: ${e.message}`, null);
      hideLoading();
    }
  }

  function _applySessionMetaToUI(img1Data, img2Data) {
    const meta1 = img1Data?.metadata || {};
    const meta2 = img2Data?.metadata || {};

    // Sidebar
    if (activeSceneName) activeSceneName.textContent = img1Data?.sensor || "Uploaded Image";
    if (activeSceneSub) activeSceneSub.textContent = `${meta1.acquisition_date || ''} ${meta2.acquisition_date ? '→ ' + meta2.acquisition_date : ''}`;
    if (sceneThumb1 && img1Data?.rgb_preview_b64) sceneThumb1.src = img1Data.rgb_preview_b64;
    if (sceneThumb2 && img2Data?.rgb_preview_b64) sceneThumb2.src = img2Data.rgb_preview_b64;

    // Canvas
    const sensor = img1Data?.sensor || "Satellite";
    if (canvasSceneLabel) canvasSceneLabel.innerHTML = `SCENE <b>${sensor}</b>`;
    if (canvasSensorLabel) canvasSensorLabel.textContent = `${sensor} · ${meta1.resolution_m || '?'}m`;
    if (canvasCoords) canvasCoords.textContent = meta1.crs || '--';
    if (canvasArea) canvasArea.textContent = `CRS ${meta1.crs || '--'}`;
    if (canvasBands) canvasBands.textContent = `Bands ${meta1.band_count || '--'}`;

    // Canvas images
    if (img1Data?.rgb_preview_b64) {
      if (imgT1) imgT1.src = img1Data.rgb_preview_b64;
      if (cdImgT1) cdImgT1.src = img1Data.rgb_preview_b64;
    }
    if (img2Data?.rgb_preview_b64) {
      if (imgT2) imgT2.src = img2Data.rgb_preview_b64;
      if (cdImgT2) cdImgT2.src = img2Data.rgb_preview_b64;
    }

    // Query chips from sensor type
    const queries = _defaultQueriesForSensor(img1Data?.sensor, img2Data ? 2 : 1);
    queryChipsContainer.innerHTML = "";
    queries.forEach(q => {
      const chip = document.createElement("button");
      chip.className = "query-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => runAnalysis(q));
      queryChipsContainer.appendChild(chip);
    });
    if (queries.length) queryInput.value = queries[0];
    setCanvasLayer("split");
  }

  function _defaultQueriesForSensor(sensor, nImages) {
    const s = (sensor || "").toLowerCase();
    if (nImages === 2) {
      if (s.includes("sentinel-1") || s.includes("sar"))
        return ["What does the SAR and optical fusion reveal about land cover?",
                "Detect flooded areas using SAR backscatter"];
      return ["What changed between the two images?",
              "Detect deforestation or urban expansion",
              "Highlight all changed regions and estimate area in km²"];
    }
    return ["Describe this satellite image",
            "What land cover types are visible?",
            "Locate water bodies in this image",
            "Identify built-up areas"];
  }

  function _connectWebSocket() {
    if (!currentSessionId) return;
    if (activeWebSocket) activeWebSocket.close();
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/${currentSessionId}`;
    activeWebSocket = new WebSocket(wsUrl);
    activeWebSocket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "trace") {
          _appendLiveTraceStep(msg);
        } else if (msg.type === "complete") {
          _handleQueryComplete(msg.result);
        } else if (msg.type === "error") {
          appendChatMessage("agent", `Error: ${msg.error}`, null);
          traceStatusBadge.textContent = "● Error";
          hideLoading();
        }
      } catch(e) { console.warn("WS parse error:", e); }
    };
    activeWebSocket.onerror = (e) => console.warn("WebSocket error:", e);
    // Send heartbeat
    setInterval(() => {
      if (activeWebSocket && activeWebSocket.readyState === WebSocket.OPEN)
        activeWebSocket.send("ping");
    }, 20000);
  }

  // ---------- View Navigation ----------
  function switchView(viewName) {
    activeView = viewName;
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    $(`view-${viewName}`).classList.add("active");

    document.querySelectorAll(".nav-tab, .side-nav-item").forEach(el => {
      el.classList.toggle("active", el.dataset.view === viewName);
    });

    if (viewModeMap[viewName]) currentMode = viewModeMap[viewName];
    syncModeChips();
    refreshView(viewName);
  }

  function refreshView(viewName) {
    switch (viewName) {
      case "change-detection":
        if (lastAnalysisResult && lastAnalysisResult.change_analysis) renderCDMetrics(lastAnalysisResult);
        else runAnalysis(queryInput.value || "Detect changes between the two acquisitions", "bi_temporal");
        break;
      case "sar-analysis":
        if (lastAnalysisResult && lastAnalysisMode === "sar_fusion") renderSARView(lastAnalysisResult);
        else runAnalysis(queryInput.value || "Interpret SAR backscatter for flood extent", "sar_fusion");
        break;
      case "visual-grounding":
        if (lastAnalysisResult && lastAnalysisResult.grounding_analysis) renderGroundingList(lastAnalysisResult.grounding_analysis.grounded_regions || []);
        else runAnalysis(queryInput.value || "Ground and localize key features in this scene", "grounding");
        break;
      case "models-data":
        loadModelsRegistry();
        break;
      case "history":
        renderHistoryList();
        break;
      case "export-report":
        renderExportPreview();
        break;
    }
  }

  function syncModeChips() {
    document.querySelectorAll(".mode-chip").forEach(chip => {
      chip.classList.toggle("active", chip.dataset.mode === currentMode);
    });
  }

  // ---------- Slider drag (generic) ----------
  function setupSliderDrag(container, wrapper, divider, ownerFn) {
    function setPos(x) {
      const rect = container.getBoundingClientRect();
      let pos = (x - rect.left) / rect.width;
      pos = Math.min(0.98, Math.max(0.02, pos));
      const pct = pos * 100;
      divider.style.left = `${pct}%`;
      wrapper.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    }
    divider.addEventListener("mousedown", (e) => { isDraggingSlider = ownerFn(); e.preventDefault(); });
    divider.addEventListener("touchstart", () => { isDraggingSlider = ownerFn(); });
    window.addEventListener("mousemove", (e) => { if (isDraggingSlider === ownerFn()) setPos(e.clientX); });
    window.addEventListener("touchmove", (e) => { if (isDraggingSlider === ownerFn()) setPos(e.touches[0].clientX); });
  }
  window.addEventListener("mouseup", () => { isDraggingSlider = null; });
  window.addEventListener("touchend", () => { isDraggingSlider = null; });

  // ---------- Layer Management ----------
  function setCanvasLayer(mode) {
    canvasToggles.forEach(b => b.classList.remove("active"));
    imgSar.style.display = "none";
    imgFused.style.display = "none";
    imgChange.style.display = "none";
    sliderWrapper.style.display = "none";
    sliderDivider.style.display = "none";

    const map = { split: "canvas-toggle-split", sar: "canvas-toggle-water", fused: "canvas-toggle-veg", change: "canvas-toggle-change" };
    const btn = $(map[mode]);
    if (btn) btn.classList.add("active");

    if (mode === "split") {
      sliderWrapper.style.display = "block";
      sliderDivider.style.display = "block";
    } else if (mode === "sar") {
      imgSar.style.display = "block";
    } else if (mode === "fused") {
      imgFused.style.display = "block";
    } else if (mode === "change") {
      imgT1.style.display = "block";
      imgChange.style.display = "block";
    }
  }

  // ---------- Events ----------
  function setupEventListeners() {
    document.querySelectorAll(".nav-tab, .side-nav-item").forEach(el => {
      el.addEventListener("click", () => switchView(el.dataset.view));
    });

    modeChipRow.querySelectorAll(".mode-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        currentMode = chip.dataset.mode;
        syncModeChips();
      });
    });

    canvasToggles.forEach(btn => {
      btn.addEventListener("click", () => setCanvasLayer(btn.dataset.mode));
    });

    zoomInBtn.addEventListener("click", () => { canvasZoom = Math.min(2, canvasZoom + 0.15); applyZoom(); });
    zoomOutBtn.addEventListener("click", () => { canvasZoom = Math.max(1, canvasZoom - 0.15); applyZoom(); });
    layersCycleBtn.addEventListener("click", () => {
      const order = ["split", "sar", "fused", "change"];
      const activeBtn = document.querySelector(".canvas-toggle.active");
      const currentIdx = activeBtn ? order.indexOf(activeBtn.dataset.mode) : 0;
      setCanvasLayer(order[(currentIdx + 1) % order.length]);
    });

    function applyZoom() {
      [imgT1, imgT2, imgSar, imgFused, imgChange].forEach(img => img.style.transform = `scale(${canvasZoom})`);
    }

    badgeBoxes.addEventListener("click", () => {
      bboxContainer.style.display = bboxContainer.style.display === "none" ? "block" : "none";
      badgeBoxes.classList.toggle("dim");
    });
    badgeSegmentation.addEventListener("click", () => badgeSegmentation.classList.toggle("dim"));

    analyzeBtn.addEventListener("click", () => runAnalysis());
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runAnalysis(); }
    });

    voiceBtn.addEventListener("click", () => {
      voiceBtn.classList.add("recording");
      setTimeout(() => {
        voiceBtn.classList.remove("recording");
        queryInput.value = "Detect structural expansion and identify flood risk regions.";
        runAnalysis();
      }, 1400);
    });

    langToggleBtn.addEventListener("click", () => {
      currentLanguage = currentLanguage === "en" ? "hi" : "en";
      currentLangText.textContent = currentLanguage === "en" ? "EN" : "HI";
      runAnalysis(queryInput.value);
    });

    uploadBrowseBtn.addEventListener("click", () => uploadFileInput.click());
    uploadDropzone.addEventListener("click", (e) => { if (e.target === uploadDropzone) uploadFileInput.click(); });
    uploadFileInput.addEventListener("change", () => {
      if (uploadFileInput.files.length) {
        uploadedFiles = Array.from(uploadFileInput.files);
        uploadStatusBadge.textContent = `● ${uploadedFiles.map(f => f.name).join(" + ")}`;
        uploadImages(uploadedFiles);
      }
    });
    ["dragover", "dragenter"].forEach(evt => uploadDropzone.addEventListener(evt, (e) => {
      e.preventDefault(); uploadDropzone.style.borderColor = "var(--green-fg)";
    }));
    uploadDropzone.addEventListener("drop", (e) => {
      e.preventDefault(); uploadDropzone.style.borderColor = "";
      const files = Array.from(e.dataTransfer.files);
      if (files.length) { uploadedFiles = files; uploadImages(files); }
    });
    uploadDropzone.addEventListener("dragleave", (e) => { e.preventDefault(); uploadDropzone.style.borderColor = ""; });

    replayTraceBtn.addEventListener("click", () => {
      if (lastAnalysisResult) renderTrace(lastAnalysisResult.trace_log || lastAnalysisResult.trace || [], true);
    });

    infoBtn.addEventListener("click", () => registryModal.classList.add("visible"));
    closeModalBtn.addEventListener("click", () => registryModal.classList.remove("visible"));
    registryModal.addEventListener("click", (e) => { if (e.target === registryModal) registryModal.classList.remove("visible"); });

    exportPdfBtn.addEventListener("click", downloadPdfReport);
    exportGeojsonBtn.addEventListener("click", downloadGeoJSON);
    exportTraceBtn.addEventListener("click", downloadTraceLog);
  }

  // ---------- Analysis (Real API) ----------
  async function runAnalysis(queryOverride, modeOverride) {
    const query = (queryOverride !== undefined ? queryOverride : queryInput.value).trim();
    if (!query) return;

    if (!currentSessionId && !currentScenarioId) {
      appendChatMessage("agent",
        "Please select a scenario preset or upload a satellite image first using the Upload area.", null);
      return;
    }

    appendChatMessage("user", query);
    queryInput.value = "";
    traceContainer.innerHTML = "";
    showLoading("LangGraph Controller: Dispatching Inference Pipeline...");
    traceStatusBadge.textContent = "● Running";
    bboxContainer.innerHTML = "";

    try {
      const payload = {
        query,
        mode: modeOverride || currentMode,
        language: currentLanguage
      };
      if (currentSessionId) {
        payload.session_id = currentSessionId;
      } else if (currentScenarioId) {
        payload.scenario_id = currentScenarioId;
      }

      const res = await fetch(`${BASE_URL}/api/v1/query/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      const taskData = await res.json();

      // If immediate complete (scenario preset flow)
      if (taskData.status === "complete" || taskData.answer) {
        _handleQueryComplete(taskData.result || taskData);
        return;
      }

      const taskId = taskData.task_id;
      // Poll for completion (WebSocket handles live trace for uploaded sessions)
      await _pollTaskCompletion(taskId);

    } catch (e) {
      console.error("Analysis error:", e);
      appendChatMessage("agent", `Analysis failed: ${e.message}`, null);
      traceStatusBadge.textContent = "● Error";
      hideLoading();
    }
  }

  async function _pollTaskCompletion(taskId, maxWaitSec = 120) {
    const start = Date.now();
    while ((Date.now() - start) / 1000 < maxWaitSec) {
      await new Promise(r => setTimeout(r, 2000));
      const res = await fetch(`${BASE_URL}/api/v1/query/${taskId}/status`);
      const data = await res.json();
      if (data.status === "complete") {
        _handleQueryComplete(data.result);
        return;
      }
      if (data.status === "error") {
        appendChatMessage("agent", `Agent error: ${data.error}`, null);
        traceStatusBadge.textContent = "● Error";
        hideLoading();
        return;
      }
    }
    appendChatMessage("agent", "Analysis timed out. The model may still be loading.", null);
    hideLoading();
  }

  function _handleQueryComplete(data) {
    lastAnalysisResult = data;
    lastAnalysisMode = data.task_type;
    hideLoading();
    traceStatusBadge.textContent = "● Idle";

    // Primary answer text
    const answerText = data.answer || data.fused_findings ||
                       data.description || data.optical_findings ||
                       data.message || "Analysis complete.";
    appendChatMessage("agent", answerText, data);
    if (aiAnswerText) aiAnswerText.textContent = answerText;

    // Confidence
    const confPct = Math.round((data.confidence || 0) * 100);
    if (confidenceIndicator) {
      confidenceIndicator.textContent = `${confPct}% CONFIDENCE`;
      confidenceIndicator.style.background = confPct > 85 ? "var(--green-bg)" : "var(--amber-bg)";
      confidenceIndicator.style.color = confPct > 85 ? "var(--green-fg)" : "var(--amber-fg)";
    }

    // Update images from real analysis output
    _updateCanvasFromResult(data);
    renderDetectedClasses(data);
    renderEvidence(data);
    renderTrace(data.trace_log || []);

    // Task-specific view rendering
    const tt = data.task_type;
    if (tt === "BI_TEMPORAL_CHANGE") renderCDMetrics(data);
    if (tt === "CROSS_MODAL_SAR_OPTICAL") renderSARView(data);
    if (tt === "REGION_GROUNDING") {
      renderGroundingList(data.boxes || []);
      if (data.boxes) renderBoundingBoxes(data.boxes);
    }
    if (data.clarification_needed) {
      const opts = (data.options || []).map(o => `• ${o}`).join("\n");
      appendChatMessage("agent", `${data.message}\n${opts}`, null);
    }

    runHistory.unshift({
      query: data.query || "", confidence: confPct, timestamp: new Date(),
      thumb: data.t2_preview_b64 || data.optical_preview_b64 ||
             (currentImages && currentImages.image_t2) || "",
      scenarioTitle: `Session ${currentSessionId ? currentSessionId.slice(0,8) : ''}`,
    });
    if (activeView === "history") renderHistoryList();
    if (activeView === "export-report") renderExportPreview();
  }

  function _updateCanvasFromResult(data) {
    const tt = data.task_type;
    if (tt === "BI_TEMPORAL_CHANGE") {
      if (data.t1_preview_b64 && imgT1) imgT1.src = data.t1_preview_b64;
      if (data.t2_preview_b64 && imgT2) imgT2.src = data.t2_preview_b64;
      if (data.overlay_b64 && imgChange) { imgChange.src = data.overlay_b64; setCanvasLayer("change"); }
      if (data.t1_preview_b64 && cdImgT1) cdImgT1.src = data.t1_preview_b64;
      if (data.t2_preview_b64 && cdImgT2) cdImgT2.src = data.t2_preview_b64;
    } else if (tt === "CROSS_MODAL_SAR_OPTICAL") {
      if (data.optical_preview_b64 && imgT1) imgT1.src = data.optical_preview_b64;
      if (data.sar_preview_b64 && imgSar) { imgSar.src = data.sar_preview_b64; }
      if (data.fusion_overlay_b64 && imgFused) { imgFused.src = data.fusion_overlay_b64; setCanvasLayer("fused"); }
    } else if (tt === "REGION_GROUNDING") {
      if (data.annotated_image_b64 && imgT2) imgT2.src = data.annotated_image_b64;
    } else if (data.preview_b64 && imgT1) {
      imgT1.src = data.preview_b64;
    }
  }

  function _appendLiveTraceStep(msg) {
    const icon = msg.status === "running" ? "⟳" : (msg.status === "error" ? "✗" : "✓");
    const colorClass = msg.status === "error" ? "c-red" : (msg.status === "running" ? "c-amber" : "c-green");
    const div = document.createElement("div");
    div.className = "trace-step";
    div.innerHTML = `
      <div class="trace-icon ${colorClass}">${icon}</div>
      <div class="trace-content">
        <div class="trace-header">
          <span class="trace-node">${msg.node.replace(/NODE_\d+_/, "").replace(/_/g, " ")}</span>
          <span class="trace-latency">${msg.status}</span>
        </div>
        <div class="trace-desc">${msg.detail || ""}</div>
      </div>
    `;
    traceContainer.appendChild(div);
    traceContainer.scrollTop = traceContainer.scrollHeight;
  }

  function appendChatMessage(role, text, data) {
    const row = document.createElement("div");
    row.className = "chat-msg";
    const avatarLetter = role === "user" ? "U" : "◆";
    row.innerHTML = `
      <div class="chat-avatar ${role}">${avatarLetter}</div>
      <div>
        <div class="chat-meta">${role === "user" ? "You" : "SatQuery Agent"}</div>
        <div class="chat-bubble">${text}</div>
      </div>
    `;
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function extractStatItems(data) {
    const items = [];
    if (data.change_analysis && data.change_analysis.change_metrics) {
      for (const [k, v] of Object.entries(data.change_analysis.change_metrics)) {
        items.push({ label: k.replace(/_/g, " "), val: v });
      }
    } else if (data.sar_fusion_analysis && data.sar_fusion_analysis.metrics) {
      data.sar_fusion_analysis.metrics.forEach(m => items.push({ label: m.label, val: m.value }));
    } else if (data.findings) {
      data.findings.forEach(f => items.push({ label: f.category, val: f.detail }));
    }
    return items;
  }

  function renderDetectedClasses(data) {
    detectedClasses.innerHTML = "";
    const items = extractStatItems(data).slice(0, 3);
    const fills = ["fill-blue", "fill-green", "fill-red", "fill-amber"];
    items.forEach((item, idx) => {
      let pct = 50;
      const m = String(item.val).match(/(\d+(\.\d+)?)\s*%/);
      if (m) pct = parseFloat(m[1]);
      const div = document.createElement("div");
      div.innerHTML = `
        <div class="class-stat-label"><span>${item.label.toUpperCase()}</span><span class="class-stat-value">${item.val}</span></div>
        <div class="class-stat-bar"><div class="class-stat-fill ${fills[idx % fills.length]}" style="width:${Math.min(100, pct)}%"></div></div>
      `;
      detectedClasses.appendChild(div);
    });
    if (!items.length) detectedClasses.innerHTML = `<div class="card-subtitle">No classification metrics returned.</div>`;
  }

  function renderEvidence(data) {
    evidenceCitations.innerHTML = "";
    let lines = [];
    if (data.findings && data.findings.length) {
      lines = data.findings.map(f => `${f.category}: ${f.detail}`);
    } else if (data.trace && data.trace.length) {
      lines = data.trace.slice(0, 4).map(t => t.action);
    }
    if (!lines.length) lines = ["No auxiliary evidence returned for this query."];
    lines.forEach(l => {
      const div = document.createElement("div");
      div.className = "evidence-item";
      div.textContent = l;
      evidenceCitations.appendChild(div);
    });
  }

  function renderTrace(traceList) {
    traceContainer.innerHTML = "";
    (traceList || []).forEach((step) => {
      const status = step.status || "done";
      const colorClass = status === "error" ? "c-red" : (status === "running" ? "c-amber" : "c-green");
      const icon = status === "error" ? "✗" : (status === "running" ? "⟳" : "✓");

      const div = document.createElement("div");
      div.className = "trace-step";
      div.innerHTML = `
        <div class="trace-icon ${colorClass}">${icon}</div>
        <div class="trace-content">
          <div class="trace-header">
            <span class="trace-node">${(step.node || "").replace(/NODE_\d+_/, "").replace(/_/g, " ")}</span>
            <span class="trace-latency">${step.elapsed_sec !== undefined ? step.elapsed_sec + "s" : ""}</span>
          </div>
          <div class="trace-desc">${step.detail || step.summary || ""}</div>
        </div>
      `;
      traceContainer.appendChild(div);
    });
    if (!traceList || !traceList.length)
      traceContainer.innerHTML = `<div class="card-subtitle">No trace steps yet. Run a query to see the agent trace.</div>`;
  }

  function renderBoundingBoxes(boxes) {
    bboxContainer.innerHTML = "";
    (boxes || []).forEach(box => {
      // New API: {x1, y1, x2, y2, label, confidence}
      const x1 = box.x1 ?? (box.bbox && box.bbox[1]) ?? 0;
      const y1 = box.y1 ?? (box.bbox && box.bbox[0]) ?? 0;
      const x2 = box.x2 ?? (box.bbox && box.bbox[3]) ?? 50;
      const y2 = box.y2 ?? (box.bbox && box.bbox[2]) ?? 50;
      const overlay = document.createElement("div");
      overlay.className = "bbox-overlay";
      overlay.style.top = `${y1}px`;
      overlay.style.left = `${x1}px`;
      overlay.style.width = `${x2 - x1}px`;
      overlay.style.height = `${y2 - y1}px`;
      const label = document.createElement("div");
      label.className = "bbox-label";
      label.textContent = `${box.label || "Region"} (${Math.round((box.confidence || 0) * 100)}%)`;
      overlay.appendChild(label);
      bboxContainer.appendChild(overlay);
    });
  }

  // ---------- Change Detection View ----------
  function renderCDMetrics(data) {
    if (!cdMetrics) return;
    cdMetrics.innerHTML = "";
    const palette = [
      { bg: "var(--red-bg)", fg: "var(--red-fg)" },
      { bg: "var(--blue-bg)", fg: "var(--blue-fg)" },
      { bg: "var(--green-bg)", fg: "var(--green-fg)" }
    ];

    // New API response shape
    const metrics = {
      "Changed Area": `${data.change_pct ?? 0}%`,
      "Area km²": `${data.changed_area_km2 ?? 0} km²`,
      "Regions": data.change_stats?.n_change_regions ?? "--",
    };
    if (data.evaluation && data.evaluation.iou !== undefined) {
      metrics["IoU"] = data.evaluation.iou;
      metrics["F1"] = data.evaluation.f1;
    }

    let i = 0;
    for (const [k, v] of Object.entries(metrics)) {
      const chip = document.createElement("div");
      chip.className = "cd-chip";
      const c = palette[i % palette.length];
      chip.style.background = c.bg; chip.style.color = c.fg;
      chip.textContent = `${k.toUpperCase()}  ${v}`;
      cdMetrics.appendChild(chip);
      i++;
    }
    const modelChip = document.createElement("div");
    modelChip.className = "cd-chip";
    modelChip.style.background = "#26382a"; modelChip.style.color = "#fff";
    modelChip.textContent = `MODEL ${data.model_used || "ChangeFormer"}`.substring(0, 40);
    cdMetrics.appendChild(modelChip);
  }

  // ---------- SAR View ----------
  function renderBars(container, seed) {
    if (!container) return;
    container.innerHTML = "";
    for (let i = 0; i < 22; i++) {
      const h = 20 + Math.abs(Math.sin(i * 0.7 + seed)) * 70;
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.height = `${h}%`;
      container.appendChild(bar);
    }
  }

  function renderSARView(data) {
    // New API: data.sar_findings, data.optical_findings, data.fused_findings, data.fusion_stats
    const stats = data.fusion_stats || {};
    if (sarVvDb) sarVvDb.textContent = stats.sar_feature_norm !== undefined ? `${stats.sar_feature_norm.toFixed(1)} (norm)` : "--";
    if (sarVhDb) sarVhDb.textContent = stats.optical_feature_norm !== undefined ? `${stats.optical_feature_norm.toFixed(1)} (norm)` : "--";
    if (sarVvBars) renderBars(sarVvBars, 1);
    if (sarVhBars) renderBars(sarVhBars, 2.4);
    if (sarInterpretationText) {
      const sar = data.sar_findings || "";
      const fused = data.fused_findings || data.optical_findings || "Interpretation pending.";
      sarInterpretationText.innerHTML = `<b>SAR Findings:</b> ${sar}<br><br><b>Fused Analysis:</b> ${fused}`;
    }
  }

  // ---------- Grounding View ----------
  function renderGroundingList(regions) {
    if (!groundingList) return;
    groundingList.innerHTML = "";
    if (!regions || !regions.length) {
      groundingList.innerHTML = `<div class="card-subtitle">No grounded regions returned for this query.</div>`;
      return;
    }
    regions.forEach(r => {
      const item = document.createElement("div");
      item.className = "grounding-item";
      // New API boxes: {x1,y1,x2,y2,label,confidence}
      const bboxStr = r.bbox
        ? r.bbox.join(", ")
        : `${r.x1},${r.y1} → ${r.x2},${r.y2}`;
      item.innerHTML = `
        <div class="grounding-icon"><svg class="icon"><use href="#i-crosshair"/></svg></div>
        <div class="grounding-body">
          <div class="grounding-phrase">"${r.label || 'Region ' + (r.region_idx||'')}"</div>
          <div class="grounding-meta">bbox [${bboxStr}]  conf ${Math.round((r.confidence||0) * 100)}%</div>
        </div>
        <button class="grounding-show">Show</button>
      `;
      item.querySelector(".grounding-show").addEventListener("click", () => {
        switchView("workspace");
        setCanvasLayer("split");
        renderBoundingBoxes(regions);
      });
      groundingList.appendChild(item);
    });
  }

  // ---------- History View ----------
  function renderHistoryList() {
    historyList.innerHTML = "";
    if (!runHistory.length) {
      historyList.innerHTML = `<div class="card-subtitle">No queries run yet this session.</div>`;
      return;
    }
    runHistory.forEach(h => {
      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <img class="history-thumb" src="${h.thumb}" alt="">
        <div class="history-body">
          <div class="history-title">${h.query}</div>
          <div class="history-meta">${timeAgo(h.timestamp)} · conf ${h.confidence}% · ${h.scenarioTitle}</div>
        </div>
        <svg class="icon history-chevron"><use href="#i-chevron"/></svg>
      `;
      historyList.appendChild(item);
    });
  }

  function timeAgo(date) {
    const diffMs = Date.now() - date.getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  }

  // ---------- Models & Data View ----------
  async function loadModelsRegistry() {
    if (!modelsList) return;
    modelsList.innerHTML = `<div class="card-subtitle">Loading registry...</div>`;
    try {
      if (!modelsRegistryCache) {
        const res = await fetch(`${BASE_URL}/api/v1/models`);
        const data = await res.json();
        // New registry: top-level keys are model types
        modelsRegistryCache = data.models || data;
      }
      renderModelsList(modelsRegistryCache);
    } catch (e) {
      console.error("Failed fetching model registry:", e);
      if (modelsList) modelsList.innerHTML = `<div class="card-subtitle">Registry unavailable (server starting up?).</div>`;
    }
  }

  function renderModelsList(registry) {
    if (!modelsList) return;
    modelsList.innerHTML = "";
    const badgeClasses = ["badge-green", "badge-blue", "badge-red", "badge-amber"];
    let i = 0;
    const items = Array.isArray(registry) ? registry
      : Object.entries(registry).map(([k, v]) => (typeof v === "object" ? { ...v, _key: k } : { id: k, _key: k }));
    items.forEach((mod) => {
      const item = document.createElement("div");
      item.className = "model-item";
      const name = mod.id || mod.name || mod._key || "Unknown";
      const type = mod.specialization || mod.type || mod.task || "";
      const dataset = mod.training_dataset || mod.fine_tuned_on || "";
      const status = mod.status || "available";
      item.innerHTML = `
        <div>
          <div class="model-name">${name}</div>
          <div class="model-type">${type}</div>
          ${dataset ? `<div class="model-adapted">TRAINED ON <b>${dataset}</b></div>` : ""}
        </div>
        <span class="badge ${badgeClasses[i % badgeClasses.length]}">● ${status}</span>
      `;
      modelsList.appendChild(item);
      i++;
    });
    if (!items.length) modelsList.innerHTML = `<div class="card-subtitle">No models in registry.</div>`;
  }

  // ---------- Export Report View ----------
  function renderExportPreview() {
    if (!lastAnalysisResult) {
      reportPreviewQuery.textContent = "--";
      reportPreviewAnswer.textContent = "Run an analysis to populate the report preview.";
      reportPreviewMeta.textContent = "--";
      reportPreviewThumb.src = (currentImages && currentImages.image_t2) || "";
      return;
    }
    reportPreviewThumb.src = (currentImages && currentImages.image_t2) || "";
    reportPreviewQuery.textContent = queryInput.value || (chatLog.lastElementChild ? "" : "--");
    reportPreviewAnswer.textContent = lastAnalysisResult.answer || "--";
    const traceLen = (lastAnalysisResult.trace || []).length;
    reportPreviewMeta.textContent = `Trace: ${traceLen} steps · ${lastAnalysisResult.total_latency_sec || "--"}s · Confidence ${Math.round((lastAnalysisResult.confidence || 0) * 100)}%`;
  }

  async function downloadPdfReport() {
    if (!lastAnalysisResult) { alert("Please run an analysis query first!"); return; }
    try {
      const subEl = exportPdfBtn.querySelector(".export-option-sub");
      if (subEl) subEl.textContent = "Generating...";

      const payload = {};
      if (currentSessionId) {
        payload.session_id = currentSessionId;
        payload.query = lastAnalysisResult.query || "";
      } else {
        payload.analysis_result = lastAnalysisResult;
        payload.scenario_id = currentScenarioId || "kerala_floods_2018";
      }

      const res = await fetch(`${BASE_URL}/api/v1/export/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const contentType = res.headers.get("content-type") || "";
      if (contentType.includes("application/pdf")) {
        const blob = await res.blob();
        triggerDownload(blob, `SatQuery_Report_${(currentScenarioId || currentSessionId || 'mission').slice(0, 16)}.pdf`);
      } else {
        const data = await res.json();
        if (data.pdf_b64) {
          const link = document.createElement("a");
          link.href = data.pdf_b64;
          link.download = data.filename || "satquery_report.pdf";
          link.click();
        } else {
          const blob = await res.blob();
          triggerDownload(blob, `SatQuery_Report_${(currentSessionId || 'mission').slice(0,8)}.pdf`);
        }
      }
      if (subEl) subEl.textContent = "Full report + trace";
    } catch (e) {
      console.error("PDF Export error:", e);
      const subEl = exportPdfBtn.querySelector(".export-option-sub");
      if (subEl) subEl.textContent = "Export failed — try again";
    }
  }

  function downloadGeoJSON() {
    if (!lastAnalysisResult) { alert("Please run an analysis query first!"); return; }
    // New API: data.boxes[]
    const boxes = lastAnalysisResult.boxes || [];
    const geojson = {
      type: "FeatureCollection",
      features: boxes.map(b => ({
        type: "Feature",
        properties: { label: b.label, confidence: b.confidence, region_idx: b.region_idx },
        geometry: { type: "Polygon", coordinates: [[
          [b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2], [b.x1, b.y1]
        ]] }
      }))
    };
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: "application/geo+json" });
    triggerDownload(blob, `SatQuery_Grounding_${(currentSessionId||'session').slice(0,8)}.geojson`);
  }

  function bboxToPolygon(bbox) {
    const [ymin, xmin, ymax, xmax] = bbox;
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]];
  }

  function downloadTraceLog() {
    if (!lastAnalysisResult) { alert("Please run an analysis query first!"); return; }
    const trace = lastAnalysisResult.trace_log || lastAnalysisResult.trace || [];
    const blob = new Blob([JSON.stringify(trace, null, 2)], { type: "application/json" });
    triggerDownload(blob, `SatQuery_Trace_${(currentSessionId||'session').slice(0,8)}.json`);
  }

  function triggerDownload(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }

  // ---------- Loading ----------
  function showLoading(text) {
    loadingStatusText.textContent = text;
    loadingOverlay.classList.add("visible");
  }
  function hideLoading() {
    loadingOverlay.classList.remove("visible");
  }

  init();
});