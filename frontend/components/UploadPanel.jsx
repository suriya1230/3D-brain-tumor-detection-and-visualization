"use client";

import { useCallback, useRef, useState } from "react";
import { INK, MODALITIES } from "./theme";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Filename patterns, most specific first. t1c must be tested before t1n or
// "BraTS-GLI-00046-101-t1c.nii.gz" would match the looser t1 rule.
const RULES = [
  ["t2f", ["t2f", "flair", "_fl"]],
  ["t1c", ["t1c", "t1ce", "t1gd", "t1_ce", "t1-ce"]],
  ["t2w", ["t2w", "_t2", "t2-", "t2."]],
  ["t1n", ["t1n", "t1w", "_t1", "t1-", "t1."]],
];

function classify(files) {
  const picked = {};
  const taken = new Set();
  for (const [key, tokens] of RULES) {
    for (const f of files) {
      if (taken.has(f)) continue;
      if (tokens.some((t) => f.name.toLowerCase().includes(t))) {
        picked[key] = f;
        taken.add(f);
        break;
      }
    }
  }
  return picked;
}

export default function UploadPanel({ onResult }) {
  const [files, setFiles] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  const accept = useCallback((list) => {
    const arr = Array.from(list).filter((f) =>
      /\.nii(\.gz)?$/i.test(f.name)
    );
    if (!arr.length) {
      setError("Those files are not NIfTI. Expected .nii or .nii.gz.");
      return;
    }
    setError(null);
    setFiles((prev) => ({ ...prev, ...classify(arr) }));
  }, []);

  const ready = MODALITIES.every((m) => files[m.key]);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      MODALITIES.forEach((m) => body.append(m.key, files[m.key]));

      const res = await fetch(`${API}/api/predict`, {
        method: "POST",
        body,
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.detail || `server returned ${res.status}`);

      onResult({
        meta: json,
        brainUrl: `${API}${json.assets.brain}`,
        tumorUrl: `${API}${json.assets.tumor}`,
      });
    } catch (e) {
      setError(
        e.message.includes("fetch")
          ? `Cannot reach the API at ${API}. Is the backend running?`
          : e.message
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={S.wrap}>
      <div style={S.card}>
        <h1 style={S.h1}>Upload a scan</h1>
        <p style={S.lede}>
          Four co-registered volumes from one session, 1&nbsp;mm isotropic and
          skull-stripped. NIfTI only.
        </p>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            accept(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          style={{
            ...S.drop,
            borderColor: drag ? INK.brain : "rgba(79,216,255,0.25)",
            background: drag ? "rgba(79,216,255,0.07)" : "transparent",
          }}
        >
          <p style={S.dropText}>
            Drop all four files here, or click to choose them.
          </p>
          <p style={S.dropSub}>
            Named BraTS-style, they sort themselves into the right slots.
          </p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".nii,.nii.gz"
            onChange={(e) => accept(e.target.files)}
            style={{ display: "none" }}
          />
        </div>

        <ul style={S.slots}>
          {MODALITIES.map((m) => (
            <li key={m.key} style={S.slot}>
              <span
                style={{
                  ...S.tick,
                  background: files[m.key] ? INK.brain : "transparent",
                  borderColor: files[m.key] ? INK.brain : INK.dim,
                }}
              />
              <span style={S.slotName}>
                {m.label}
                {m.hint && <span style={S.slotHint}> {m.hint}</span>}
              </span>
              <span style={S.slotFile}>
                {files[m.key]?.name ?? "not set"}
              </span>
            </li>
          ))}
        </ul>

        {error && <p style={S.error}>{error}</p>}

        <button
          onClick={submit}
          disabled={!ready || busy}
          style={{
            ...S.cta,
            opacity: !ready || busy ? 0.42 : 1,
            cursor: !ready || busy ? "not-allowed" : "pointer",
          }}
        >
          {busy ? "Segmenting…" : "Segment and build 3D view"}
        </button>

        {busy && (
          <p style={S.wait}>
            Sliding-window inference over the whole volume. A few seconds on a
            GPU, two to five minutes on CPU.
          </p>
        )}

        <p style={S.foot}>
          SegResNet trained on BraTS 2024 post-treatment glioma. Selection Dice
          0.8018 on 150 held-out patients. Research use only, not for
          diagnosis.
        </p>
      </div>
    </main>
  );
}

const FONT = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";

const S = {
  wrap: {
    minHeight: "100dvh",
    display: "grid",
    placeItems: "center",
    padding: 24,
    background: `radial-gradient(120% 90% at 50% 30%, ${INK.deep} 0%, ${INK.void} 70%)`,
    fontFamily: FONT,
    color: INK.text,
  },
  card: {
    width: "min(520px, 100%)",
    padding: "34px 32px 26px",
    borderRadius: 5,
    border: "1px solid rgba(79,216,255,0.16)",
    background: "rgba(4,10,18,0.6)",
  },
  h1: { margin: 0, fontSize: 22, fontWeight: 600, letterSpacing: "-0.015em" },
  lede: {
    margin: "8px 0 22px",
    fontSize: 13,
    lineHeight: 1.6,
    color: INK.dim,
    maxWidth: "44ch",
  },
  drop: {
    padding: "26px 20px",
    borderRadius: 4,
    borderWidth: 1,
    borderStyle: "dashed",
    textAlign: "center",
    cursor: "pointer",
    transition: "background 140ms ease, border-color 140ms ease",
  },
  dropText: { margin: 0, fontSize: 13.5 },
  dropSub: { margin: "6px 0 0", fontSize: 11.5, color: INK.dim },
  slots: {
    listStyle: "none",
    margin: "22px 0 0",
    padding: 0,
    display: "grid",
    gap: 11,
  },
  slot: { display: "flex", alignItems: "center", gap: 11, fontSize: 12.5 },
  tick: {
    flex: "0 0 auto",
    width: 11,
    height: 11,
    borderRadius: 2,
    borderWidth: 1,
    borderStyle: "solid",
  },
  slotName: { flex: "0 0 118px", fontWeight: 500 },
  slotHint: { fontWeight: 400, color: INK.dim },
  slotFile: {
    flex: 1,
    minWidth: 0,
    color: INK.dim,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  error: {
    margin: "18px 0 0",
    fontSize: 12.5,
    lineHeight: 1.55,
    color: "#ff8a9c",
  },
  cta: {
    width: "100%",
    marginTop: 24,
    padding: "12px 16px",
    fontFamily: FONT,
    fontSize: 13.5,
    fontWeight: 500,
    color: INK.void,
    background: INK.brain,
    border: "none",
    borderRadius: 3,
  },
  wait: {
    margin: "12px 0 0",
    fontSize: 11.5,
    lineHeight: 1.55,
    color: INK.dim,
  },
  foot: {
    margin: "24px 0 0",
    paddingTop: 18,
    borderTop: "1px solid rgba(79,216,255,0.12)",
    fontSize: 10.5,
    lineHeight: 1.6,
    color: "#43606f",
  },
};
