"use client";

import { useEffect, useRef, useState } from "react";
import { INK } from "./theme";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function citeLabel(s) {
  if (s.source_type === "nci_pdq") return "NCI PDQ";
  if (s.source_type === "web") {
    try {
      return `🌐 ${new URL(s.url).hostname.replace(/^www\./, "")}`;
    } catch {
      return "🌐 web";
    }
  }
  return s.title.slice(0, 28);
}

function Citations({ sources, chunkIds }) {
  const cited = sources.filter((s) => chunkIds.includes(s.chunk_id));
  if (!cited.length) return null;
  return (
    <div style={S.citeRow}>
      {cited.map((s) => (
        <a
          key={s.chunk_id}
          href={s.url}
          target="_blank"
          rel="noreferrer"
          title={s.section || s.title}
          style={{
            ...S.citeTag,
            ...(s.source_type === "web" ? S.citeTagWeb : null),
          }}
        >
          {citeLabel(s)}
        </a>
      ))}
    </div>
  );
}

/* --------------------------------------------- Phase 4 default analysis */
// The five automatic sections. No citations here on purpose — this is
// general clinical-education explanation grounded in this case's own
// segmentation, not a literature lookup. See clinical_agent.py.

function Section({ number, title, children }) {
  return (
    <div style={S.section}>
      <h3 style={S.sectionTitle}>
        <span style={S.sectionNum}>{number}</span>
        {title}
      </h3>
      {children}
    </div>
  );
}

function Fact({ label, value }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div style={S.factRow}>
      <span style={S.factLabel}>{label}</span>
      <span style={S.factValue}>{value}</span>
    </div>
  );
}

function DefaultAnalysis({ data }) {
  if (!data) return null;
  const t = data.tumor_analysis || {};
  const b = data.brain_effects || {};
  const g = data.growth_assessment || {};
  const p = data.prognosis_assessment || {};
  const tr = data.treatment_information || {};

  return (
    <>
      <Section number="1" title="What is the tumor?">
        <Fact label="Tumor detected" value={t.tumor_detected ? "Yes" : "No"} />
        <Fact label="Predicted category" value={t.predicted_category} />
        <Fact label="Tumor volume" value={t.tumor_volume_cc != null ? `${t.tumor_volume_cc} cm³` : null} />
        <Fact
          label="Model holdout Dice"
          value={t.model_holdout_dice != null ? `${t.model_holdout_dice} (model performance, not this scan's confidence)` : null}
        />
        {t.explanation && <p style={S.sectionText}>{t.explanation}</p>}
      </Section>

      <Section number="2" title="How can it affect the brain/body?">
        {b.locations?.length > 0 && (
          <Fact label="Location" value={b.locations.join(", ")} />
        )}
        {b.explanation && <p style={S.sectionText}>{b.explanation}</p>}
        {b.possible_effects?.length > 0 && (
          <>
            <p style={S.sectionMeta}>Possible effects (not confirmed symptoms):</p>
            <ul style={S.bulletList}>
              {b.possible_effects.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </>
        )}
      </Section>

      <Section number="3" title="How long has it been growing?">
        <p style={S.sectionTextStrong}>{g.estimate}</p>
        {g.reason && <p style={S.sectionTextDim}>{g.reason}</p>}
        {g.future_capability && <p style={S.sectionTextDim}>{g.future_capability}</p>}
      </Section>

      <Section number="4" title="What is the survival outlook?">
        <p style={S.sectionTextStrong}>{p.individual_survival_probability}</p>
        {p.required_factors?.length > 0 && (
          <>
            <p style={S.sectionMeta}>Established factors in prognosis:</p>
            <ul style={S.bulletList}>
              {p.required_factors.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </>
        )}
        {p.note && <p style={S.sectionTextDim}>{p.note}</p>}
      </Section>

      <Section number="5" title="What treatments are used?">
        {tr.explanation && <p style={S.sectionText}>{tr.explanation}</p>}
        {tr.typical_categories?.length > 0 && (
          <ul style={S.bulletList}>
            {tr.typical_categories.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        )}
      </Section>
    </>
  );
}

function Warnings({ warnings }) {
  if (!warnings?.length) return null;
  return (
    <div style={S.warningsBox}>
      <span style={S.warningsLabel}>Warnings</span>
      {warnings.map((w, i) => (
        <p key={i} style={S.warningLine}>
          ⚠ {w}
        </p>
      ))}
    </div>
  );
}

/* ----------------------------------------------- freeform "ask" answers */
// Unlike the default analysis, these ARE citation-gated — /ask still runs
// through the Phase 3 retrieval + contract-enforced answer pipeline. A
// different safety model for a different kind of question (evidence
// lookup vs. explaining this case's own segmentation).

function AnswerBlock({ answer }) {
  if (!answer) return null;
  const isImaging = answer.routed_to === "imaging_agent";
  const hasContent = answer.claims?.length > 0;

  return (
    <div style={S.answerBlock}>
      {isImaging && (
        <p style={S.imagingNote}>Answered directly from this scan's measurements — no literature search needed.</p>
      )}

      {isImaging && answer.measurements?.length > 0 && (
        <div style={S.measureBoxInline}>
          {answer.measurements.map((m, i) => (
            <p key={i} style={S.measureLine}>{m}</p>
          ))}
        </div>
      )}

      {!isImaging && hasContent && (
        <ul style={S.claimList}>
          {answer.claims.map((c, i) => (
            <li key={i} style={S.claimItem}>
              <p style={S.claimText}>{c.text}</p>
              {c.flagged && (
                <p style={S.flagNote} title={c.flag_reason || ""}>
                  ⚠ flagged for review{c.flag_reason ? `: ${c.flag_reason}` : ""}
                </p>
              )}
              <Citations sources={answer.sources || []} chunkIds={c.chunk_ids} />
            </li>
          ))}
        </ul>
      )}
      {!isImaging && !hasContent && (
        <p style={S.noEvidence}>The evidence corpus doesn't cover this.</p>
      )}
      {answer.unsupported?.length > 0 && (
        <div style={S.unsupportedBox}>
          <span style={S.unsupportedLabel}>Not addressed by current evidence</span>
          {answer.unsupported.map((u, i) => (
            <p key={i} style={S.unsupportedText}>{u}</p>
          ))}
        </div>
      )}
    </div>
  );
}

export default function EvidencePanel({ jobId, open = true }) {
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [history, setHistory] = useState([]); // [{question, answer}]
  const lastQuestionRef = useRef(null); // resolves "this"/"it" in the next follow-up

  const askedAnalysis = useRef(false);

  useEffect(() => {
    if (!open || askedAnalysis.current) return;
    askedAnalysis.current = true;
    setAnalyzing(true);
    fetch(`${API}/api/cases/${jobId}/explain`, { method: "POST" })
      .then((r) => r.json())
      .then((d) => setAnalysis(d))
      .catch(() => setError("Couldn't reach the evidence service."))
      .finally(() => setAnalyzing(false));
  }, [open, jobId]);

  async function ask() {
    const q = question.trim();
    if (!q || asking) return;
    setAsking(true);
    setQuestion("");
    try {
      const res = await fetch(`${API}/api/cases/${jobId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, previous_question: lastQuestionRef.current }),
      });
      const answer = await res.json();
      setHistory((h) => [...h, { question: q, answer }]);
      lastQuestionRef.current = q;
    } catch {
      setHistory((h) => [
        ...h,
        { question: q, answer: { claims: [], unsupported: ["Request failed."], sources: [] } },
      ]);
    } finally {
      setAsking(false);
    }
  }

  if (!open) return null;

  return (
    <aside style={S.drawer}>
      <div style={S.header}>
        <h2 style={S.title}>NeuroEvidence AI</h2>
      </div>

      <p style={S.disclaimer}>
        AI-assisted analysis, for clinician review — not autonomous
        diagnosis. Sections 1, 2 and 5 explain this case's own segmentation
        in general clinical terms; anything you ask below is answered from
        the cited evidence corpus instead.
      </p>

      <div style={S.scroll}>
        {analyzing && <p style={S.loading}>Analyzing this case…</p>}
        {error && <p style={S.errorText}>{error}</p>}

        <DefaultAnalysis data={analysis} />
        <Warnings warnings={analysis?.warnings} />

        {history.map((turn, i) => (
          <div key={i} style={S.turn}>
            <p style={S.question}>{turn.question}</p>
            <AnswerBlock answer={turn.answer} />
          </div>
        ))}
        {asking && <p style={S.loading}>Thinking…</p>}
      </div>

      <div style={S.inputRow}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Ask your own question…"
          style={S.input}
        />
        <button onClick={ask} disabled={asking || !question.trim()} style={S.askBtn}>
          Ask
        </button>
      </div>
    </aside>
  );
}

const FONT = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";

const S = {
  drawer: {
    position: "absolute",
    left: 24,
    top: 60,
    bottom: 24,
    width: "min(400px, calc(100vw - 48px))",
    display: "flex",
    flexDirection: "column",
    borderRadius: 4,
    border: "1px solid rgba(79,216,255,0.16)",
    background: "rgba(4,10,18,0.86)",
    backdropFilter: "blur(14px)",
    fontFamily: FONT,
    color: INK.text,
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "16px 16px 0",
  },
  title: { margin: 0, fontSize: 15, fontWeight: 600 },
  disclaimer: {
    margin: "8px 16px 0",
    fontSize: 10.5,
    lineHeight: 1.5,
    color: INK.dim,
    paddingBottom: 12,
    borderBottom: "1px solid rgba(79,216,255,0.12)",
  },
  scroll: {
    flex: 1,
    overflowY: "auto",
    padding: "12px 16px",
  },
  loading: { fontSize: 12, color: INK.dim, fontStyle: "italic" },
  errorText: { fontSize: 12, color: "#ff8a9c" },

  section: {
    marginBottom: 16,
    paddingBottom: 14,
    borderBottom: "1px solid rgba(79,216,255,0.1)",
  },
  sectionTitle: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    margin: "0 0 8px",
    fontSize: 12.5,
    fontWeight: 600,
    color: INK.rim,
  },
  sectionNum: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    width: 16,
    height: 16,
    borderRadius: 3,
    background: "rgba(79,216,255,0.14)",
    color: INK.brain,
    fontSize: 10,
    fontWeight: 700,
    flexShrink: 0,
  },
  factRow: {
    display: "flex",
    justifyContent: "space-between",
    gap: 8,
    fontSize: 11.5,
    padding: "3px 0",
    borderBottom: "1px solid rgba(79,216,255,0.06)",
  },
  factLabel: { color: INK.dim },
  factValue: { color: INK.text, fontWeight: 500, textAlign: "right" },
  sectionText: { margin: "8px 0 0", fontSize: 12.5, lineHeight: 1.55, color: INK.text },
  sectionTextStrong: { margin: "4px 0 0", fontSize: 12.5, lineHeight: 1.5, color: INK.rim, fontWeight: 600 },
  sectionTextDim: { margin: "6px 0 0", fontSize: 11, lineHeight: 1.5, color: INK.dim },
  sectionMeta: { margin: "8px 0 4px", fontSize: 10.5, color: INK.dim, textTransform: "uppercase", letterSpacing: "0.03em" },
  bulletList: { margin: "4px 0 0", padding: "0 0 0 16px", fontSize: 12, lineHeight: 1.6, color: INK.text },

  warningsBox: {
    marginBottom: 4,
    padding: "10px 12px",
    borderRadius: 3,
    background: "rgba(224,179,77,0.06)",
    border: "1px solid rgba(224,179,77,0.22)",
  },
  warningsLabel: {
    display: "block",
    fontSize: 9.5,
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    color: "#e0b34d",
    marginBottom: 5,
  },
  warningLine: { margin: "3px 0", fontSize: 11, lineHeight: 1.5, color: "#d9c393" },

  answerBlock: { marginBottom: 4 },
  imagingNote: {
    margin: "0 0 8px",
    fontSize: 10.5,
    fontStyle: "italic",
    color: INK.dim,
  },
  measureBoxInline: {
    padding: "8px 10px",
    borderRadius: 3,
    background: "rgba(79,216,255,0.05)",
    border: "1px solid rgba(79,216,255,0.14)",
  },
  measureLine: { margin: "3px 0", fontSize: 11.5, lineHeight: 1.5, color: INK.rim },
  flagNote: {
    margin: "4px 0 0",
    fontSize: 10.5,
    color: "#e0b34d",
  },
  claimList: { listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 12 },
  claimItem: { paddingBottom: 10, borderBottom: "1px solid rgba(79,216,255,0.08)" },
  claimText: { margin: 0, fontSize: 12.5, lineHeight: 1.55, color: INK.text },
  noEvidence: { fontSize: 12, color: INK.dim, fontStyle: "italic" },
  citeRow: { display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 },
  citeTag: {
    fontSize: 10,
    padding: "2px 7px",
    borderRadius: 3,
    border: "1px solid rgba(79,216,255,0.3)",
    color: INK.brain,
    textDecoration: "none",
    whiteSpace: "nowrap",
  },
  // Distinct from citeTag on purpose: live web results are a different
  // trust tier from the curated NCI PDQ/PubMed corpus, and should never
  // look visually identical to it.
  citeTagWeb: {
    border: "1px solid rgba(224,179,77,0.35)",
    color: "#e0b34d",
  },
  unsupportedBox: {
    marginTop: 10,
    padding: "8px 10px",
    borderRadius: 3,
    background: "rgba(255,138,156,0.06)",
    border: "1px solid rgba(255,138,156,0.18)",
  },
  unsupportedLabel: {
    display: "block",
    fontSize: 9.5,
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    color: "#ff8a9c",
    marginBottom: 4,
  },
  unsupportedText: { margin: "2px 0", fontSize: 11.5, lineHeight: 1.5, color: "#e0aab3" },
  turn: {
    marginTop: 16,
    paddingTop: 14,
    borderTop: "1px solid rgba(79,216,255,0.14)",
  },
  question: { margin: "0 0 10px", fontSize: 12.5, fontWeight: 600, color: INK.rim },
  inputRow: {
    display: "flex",
    gap: 8,
    padding: 12,
    borderTop: "1px solid rgba(79,216,255,0.14)",
  },
  input: {
    flex: 1,
    padding: "8px 10px",
    fontFamily: FONT,
    fontSize: 12.5,
    borderRadius: 3,
    border: "1px solid rgba(79,216,255,0.28)",
    background: "rgba(79,216,255,0.06)",
    color: INK.text,
    outline: "none",
  },
  askBtn: {
    padding: "8px 14px",
    fontFamily: FONT,
    fontSize: 12.5,
    fontWeight: 600,
    borderRadius: 3,
    border: "1px solid " + INK.brain,
    background: INK.brain,
    color: INK.void,
    cursor: "pointer",
  },
};
