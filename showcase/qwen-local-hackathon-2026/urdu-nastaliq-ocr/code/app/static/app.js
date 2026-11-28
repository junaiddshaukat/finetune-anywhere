// Urdu Reader front end. Every ticked model reads every image. Each model works through the images in
// the order they were added, one at a time, and the models run in parallel with each other.
"use strict";

const $ = (id) => document.getElementById(id);

let models = [];           // { key, name, tag, state, detail, test_cer, on }
const items = [];          // { key, file, url, name, id, up, error, w, h, mode, ref, saved, res: { [model]: result } }
                           // result: { status: reading|done|error, text, seconds, lines, mode, error, score }
const busy = {};           // model key -> a read is in flight
let selected = null, nextKey = 1;

const MODE_NOTES = {
  whole: "Best for single lines and news-style pages.",
  lines: "Cuts the image into lines first; every model reads the same crops.",
};
const OFF_KEY = "uocr.models.off";

// ---------------------------------------------------------------- helpers

const pct = (x) => (x * 100).toFixed(x < 0.1 ? 1 : 0) + "%";
// colour bands: CER 5% / 15%, WER 10% / 30%
const grade = (x, good = 0.05, fair = 0.15) => (x <= good ? "good" : x <= fair ? "fair" : "poor");
const mode = () => document.querySelector("input[name=mode]:checked").value;
const active = () => models.filter((m) => m.on);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 2600);
}

async function api(path, body) {
  const res = await fetch(path, body instanceof FormData ? { method: "POST", body }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

// Phone photos can be 12+ MP. The models see at most 1024 px a side, so 2400 keeps plenty of detail.
async function shrink(file, maxSide = 2400) {
  const img = await createImageBitmap(file, { imageOrientation: "from-image" });
  const s = Math.min(1, maxSide / Math.max(img.width, img.height));
  if (s === 1 && file.size < 4e6) return file;
  const c = document.createElement("canvas");
  c.width = Math.round(img.width * s);
  c.height = Math.round(img.height * s);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.93));
}

// ---------------------------------------------------------------- models

async function loadModels() {
  let data;
  try {
    data = await (await fetch("/api/models")).json();
  } catch {
    $("model-label").textContent = "Server not reachable";
    $("model").className = "model bad";
    return setTimeout(loadModels, 4000);
  }
  const off = new Set(JSON.parse(localStorage.getItem(OFF_KEY) || "[]"));
  const first = !models.length;
  models = data.models.map((m) => ({ ...m, on: first ? !off.has(m.key) : (models.find((o) => o.key === m.key) || {}).on ?? true }));
  renderModels();
  renderBadge(data.mock);
  if (selected) renderStage();
  models.forEach((m) => pump(m));
  if (models.some((m) => m.state === "loading")) setTimeout(loadModels, 2500);
}

function renderBadge(mock) {
  const ready = models.filter((m) => m.state === "ready").length;
  const bad = models.filter((m) => m.state === "error");
  const label = $("model-label");
  $("model").className = "model " + (bad.length ? "bad" : ready === models.length ? "ok" : "wait");
  label.textContent = bad.length ? `${bad.length} model${bad.length > 1 ? "s" : ""} failed to load`
    : ready === models.length ? `${ready} model${ready > 1 ? "s" : ""} ready · all local${mock ? " · mock" : ""}`
    : `Loading models · ${ready} of ${models.length} ready`;
  $("model").title = models.map((m) => `${m.name}: ${m.detail}`).join("\n");
}

function renderModels() {
  $("model-list").replaceChildren(...models.map((m) => {
    const li = el("li", `mrow ${m.state}`);
    li.title = m.detail;
    const lab = el("label");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = m.on;
    cb.onchange = () => toggleModel(m, cb.checked);
    const txt = el("span", "mtext");
    txt.append(el("span", "mname", m.name), el("span", "mtag", m.tag));
    const right = el("span", "mright");
    if (m.test_cer != null) right.append(el("span", "chip " + grade(m.test_cer), pct(m.test_cer)));
    right.append(el("span", "dot " + m.state, ""));
    lab.append(cb, txt, right);
    li.append(lab);
    if (m.state === "error") li.append(el("p", "merr", m.detail));
    return li;
  }));
}

function toggleModel(m, on) {
  m.on = on;
  localStorage.setItem(OFF_KEY, JSON.stringify(models.filter((x) => !x.on).map((x) => x.key)));
  renderQueue();
  renderStage();
  pump(m);   // switching a model on catches it up on the images it has not read
}

// ---------------------------------------------------------------- queue

function addFiles(files) {
  const imgs = [...files].filter((f) => f.type.startsWith("image/"));
  if (!imgs.length) return toast("Those aren't images");
  const added = imgs.map((file) => ({ key: nextKey++, file, url: URL.createObjectURL(file),
    name: file.name || "pasted image", up: "uploading", ref: "", res: {}, mode: mode() }));
  items.push(...added);
  if (!selected) select(added[0]); else renderQueue();
  added.forEach(upload);
}

async function upload(it) {
  try {
    const body = new FormData();
    body.append("image", await shrink(it.file), it.name);
    const d = await api("/api/upload", body);
    Object.assign(it, { id: d.id, w: d.width, h: d.height, up: "ready" });
  } catch (e) {
    Object.assign(it, { up: "error", error: e.message });
  }
  renderQueue();
  if (it === selected) renderStage();
  models.forEach((m) => pump(m));
}

// One read per model at a time, oldest image first.
async function pump(m) {
  if (busy[m.key] || !m.on || m.state === "error") return;
  const it = items.find((i) => i.up === "ready" && !i.res[m.key]);
  if (!it) return;
  busy[m.key] = true;
  const r = (it.res[m.key] = { status: "reading", mode: it.mode });
  refresh(it);
  try {
    const body = new FormData();
    body.append("id", it.id);
    body.append("model", m.key);
    body.append("mode", it.mode);
    Object.assign(r, await api("/api/read", body), { status: "done" });
    if (it.ref.trim()) r.score = await api("/api/score", { pred: r.text, ref: it.ref });
  } catch (e) {
    Object.assign(r, { status: "error", error: e.message });
    // The server was restarted under this tab: refresh the model list / upload the image again.
    if (/Unknown model/.test(e.message)) {
      delete it.res[m.key];
      busy[m.key] = false;
      return loadModels();
    }
    if (/no longer in memory/.test(e.message)) {
      delete it.res[m.key];
      busy[m.key] = false;
      it.up = "uploading";
      return upload(it);
    }
  }
  if (it.res[m.key] === r) refresh(it);   // ignore results of a read that "Read again" replaced
  busy[m.key] = false;
  pump(m);
}

function refresh(it) {
  renderQueue();
  if (it === selected) renderStage();
}

function renderQueue() {
  $("queue").replaceChildren(...items.map((it) => {
    const li = el("li", `item${it === selected ? " sel" : ""}`);
    li.tabIndex = 0;
    const img = new Image();
    img.src = it.url;
    img.alt = "";
    const body = el("div", "ibody");
    const marks = el("div", "marks");
    if (it.up !== "ready") {
      marks.append(el("span", "st " + it.up, it.up === "error" ? "failed" : "uploading"));
    } else {
      for (const m of active()) {
        const r = it.res[m.key];
        const mk = r && r.score ? el("span", "chip " + grade(r.score.cer), pct(r.score.cer))
          : el("span", "pip " + (r ? r.status : "queued"));
        mk.title = `${m.name}: ${r ? (r.score ? `CER ${pct(r.score.cer)}` : r.status) : "waiting"}`;
        marks.append(mk);
      }
    }
    body.append(el("span", "nm", it.name), marks);
    li.append(img, body);
    li.onclick = () => select(it);
    li.onkeydown = (e) => e.key === "Enter" && select(it);
    return li;
  }));
  renderSummary();
}

function renderSummary() {
  const scored = items.filter((i) => Object.values(i.res).some((r) => r.score));
  $("summary").hidden = !items.some((i) => Object.values(i.res).some((r) => r.status === "done"));
  $("sum-n").textContent = scored.length ? `· ${scored.length} scored` : "";
  // corpus rates, as in the eval: total edits / total reference length
  const rows = active().map((m) => {
    const sc = items.map((i) => i.res[m.key] && i.res[m.key].score).filter(Boolean);
    const sum = (k) => sc.reduce((a, s) => a + s[k], 0);
    const tr = el("tr");
    tr.append(el("td", "", m.name + (models.filter((x) => x.name === m.name).length > 1 ? ` (${m.tag.split("·").pop().trim()})` : "")));
    if (!sc.length) {
      tr.append(el("td", "num muted", "–"), el("td", "num muted", "–"));
    } else {
      const cer = sum("char_edits") / Math.max(1, sum("ref_chars"));
      const wer = sum("word_edits") / Math.max(1, sum("ref_words"));
      tr.append(el("td", "num mono " + grade(cer), pct(cer)), el("td", "num mono " + grade(wer, 0.1, 0.3), pct(wer)));
    }
    return tr;
  });
  $("sum-table").tBodies[0].replaceChildren(...rows);
}

// ---------------------------------------------------------------- stage

function select(it) {
  selected = it;
  renderQueue();
  renderStage();
}

function renderStage() {
  const it = selected;
  $("empty").hidden = !!it;
  $("workspace").hidden = !it;
  if (!it) return;

  if ($("view-img").getAttribute("src") !== it.url) $("view-img").src = it.url;
  $("view-name").textContent = it.name;
  $("view-size").textContent = it.w ? `${it.w} × ${it.h}` : "";
  if ($("ref").value !== it.ref) $("ref").value = it.ref;
  $("save-note").textContent = it.saved ? "Saved to your test set" : "";

  const results = Object.values(it.res);
  const anyDone = results.some((r) => r.status === "done");
  $("score").disabled = !anyDone;
  $("save").disabled = it.up !== "ready";
  $("reread").disabled = it.up !== "ready" || results.some((r) => r.status === "reading");

  // the lowest CER among the visible models gets a "Best here" badge (ties: all of them)
  const visible = active();
  const cers = visible.map((m) => it.res[m.key] && it.res[m.key].score ? it.res[m.key].score.cer : null);
  const best = cers.filter((c) => c != null).length > 1 ? Math.min(...cers.filter((c) => c != null)) : null;
  $("legend").hidden = !cers.some((c) => c != null);

  $("no-models").hidden = visible.length > 0;
  $("columns").replaceChildren(...visible.map((m, i) => renderColumn(m, it, cers[i] != null && cers[i] === best)));
}

function renderColumn(m, it, isBest) {
  const node = $("col-tpl").content.firstElementChild.cloneNode(true);
  const q = (s) => node.querySelector(s);
  const r = it.res[m.key];
  q(".col-name").textContent = m.name;
  q(".col-tag").textContent = m.tag;
  q(".badge").hidden = !isBest;
  node.classList.toggle("best", isBest);

  const out = q(".out");
  if (it.up === "error") {
    out.textContent = `Could not upload this image: ${it.error}`;
    out.classList.add("msg");
  } else if (!r || r.status === "reading") {
    out.textContent = m.state === "loading" ? "Loading model" : r ? "Reading" : "Waiting for the model";
    out.classList.add("msg", "pending");
  } else if (r.status === "error") {
    out.textContent = r.error;
    out.classList.add("msg", "err");
  } else {
    out.textContent = r.text || "(no text found)";
    q(".col-meta").textContent = `${r.seconds}s · ${r.mode === "lines" ? `split into ${r.lines} line${r.lines === 1 ? "" : "s"}` : "whole image"}`
      + (r.score ? ` · ${r.score.char_edits}/${r.score.ref_chars} chars, ${r.score.word_edits}/${r.score.ref_words} words wrong` : "");
  }
  const copy = q(".copy");
  copy.disabled = !r || r.status !== "done";
  copy.onclick = async () => { await navigator.clipboard.writeText(r.text); toast(`Copied ${m.name}`); };

  if (r && r.score) renderScore(node, r.score);
  return node;
}

// Once scored, the diff replaces the plain output, so the text is not shown twice. Copy still copies the plain text.
function renderScore(node, s) {
  const q = (sel) => node.querySelector(sel);
  node.querySelectorAll(".stat").forEach((e) => (e.hidden = false));
  q(".cer").textContent = pct(s.cer);
  q(".cer").classList.add(grade(s.cer));
  q(".wer").textContent = pct(s.wer);
  q(".wer").classList.add(grade(s.wer, 0.1, 0.3));
  q(".out").hidden = true;
  const diff = q(".diff");
  diff.hidden = false;
  s.diff.forEach((d, i) => {
    if (i) diff.append(" ");
    if (d.op === "equal") return diff.append(d.pred);
    const span = el("span", "w " + { replace: "wrong", delete: "missed", insert: "extra" }[d.op],
                    d.op === "delete" ? d.ref : d.pred);   // delete: in the true text only; insert: in the output only
    if (d.op === "replace") span.title = "True text: " + d.ref;
    diff.append(span);
  });
}

// ---------------------------------------------------------------- events

$("files").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });

const drop = $("drop");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
document.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
document.addEventListener("paste", (e) => {
  if (e.target.matches("textarea, input")) return;   // pasting the true text must not add an image
  const files = [...e.clipboardData.items].filter((i) => i.kind === "file").map((i) => i.getAsFile());
  if (files.length) { e.preventDefault(); addFiles(files); }
});

document.querySelectorAll("input[name=mode]").forEach((r) =>
  r.addEventListener("change", () => ($("mode-note").textContent = MODE_NOTES[mode()])));

$("ref").addEventListener("input", (e) => {
  if (!selected) return;
  selected.ref = e.target.value;
  selected.saved = false;
});

$("reread").onclick = () => {
  Object.assign(selected, { res: {}, mode: mode() });
  refresh(selected);
  models.forEach((m) => pump(m));
};

$("score").onclick = async () => {
  const it = selected;
  if (!it.ref.trim()) return toast("Enter the true text first");
  try {
    await Promise.all(Object.values(it.res).filter((r) => r.status === "done").map(async (r) => {
      r.score = await api("/api/score", { pred: r.text, ref: it.ref });
    }));
  } catch (e) { toast(e.message); }
  refresh(it);
};

$("save").onclick = async () => {
  const it = selected;
  const body = new FormData();
  body.append("id", it.id);
  body.append("ref", it.ref);
  body.append("name", it.name.replace(/\.[^.]+$/, ""));
  try {
    const d = await api("/api/save", body);
    it.saved = true;
    if (it === selected) $("save-note").textContent = `Saved · ${d.count} in your test set`;
  } catch (e) { toast(e.message); }
};

$("export").onclick = () => {
  const txt = items.filter((i) => Object.keys(i.res).length).map((i) => {
    const parts = models.filter((m) => i.res[m.key] && i.res[m.key].status === "done").map((m) => {
      const r = i.res[m.key];
      return `## ${m.name} (${m.tag})${r.score ? `  CER ${pct(r.score.cer)}` : ""}\n${r.text}`;
    });
    return `# ${i.name}\n${i.ref.trim() ? `## True text\n${i.ref.trim()}\n` : ""}${parts.join("\n")}\n`;
  }).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain;charset=utf-8" }));
  a.download = "urdu-reader-comparison.txt";
  a.click();
};

document.addEventListener("keydown", (e) => {
  if (e.target.matches("textarea, input") || !selected) return;
  const i = items.indexOf(selected);
  if (e.key === "ArrowDown" || e.key === "j") select(items[Math.min(items.length - 1, i + 1)]);
  if (e.key === "ArrowUp" || e.key === "k") select(items[Math.max(0, i - 1)]);
});

loadModels();
// pick up a restarted server (models added or removed) when you come back to the tab
document.addEventListener("visibilitychange", () => document.visibilityState === "visible" && loadModels());
