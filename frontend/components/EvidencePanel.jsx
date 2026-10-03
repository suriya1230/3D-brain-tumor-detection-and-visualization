"use client";

import { createContext, useContext, useEffect, useRef, useState } from "react";
import StreamingText from "./StreamingText";
import { INK } from "./theme";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Word-boundary truncation, not a hard character cut - "Diagnosis of
// pseudoprogressi…" is not a citation anyone can follow or distinguish
// from a different paper with a similar opening word.
function truncateTitle(title, max = 56) {
  if (title.length <= max) return title;
  const cut = title.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  const safe = lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut;
  return `${safe}…`;
}

function citeLabel(s) {
  if (s.source_type === "nci_pdq") return "NCI PDQ";
  if (s.source_type === "web") {
    try {
      return `🌐 ${new URL(s.url).hostname.replace(/^www\./, "")}`;
    } catch {
      return "🌐 web";
    }
  }
  return truncateTitle(s.title);
}

// Real ChatGPT-style streaming isn't "several paragraphs each typing
// themselves out in parallel with a small stagger" - it's ONE continuous
// reveal, top to bottom, where the next piece doesn't start until the
// previous one finishes. SequenceProvider/Seq recreate that: every <Seq>
// under the same provider claims the next integer index in RENDER ORDER
// (top-to-bottom, matching reading order for free, however many claims or
// sections actually exist), and only the current index is visible/typing;
// everything after it isn't rendered at all yet, and it advances by
// calling StreamingText's onComplete. Content already fully validated by
// the backend either way - this only controls the ORDER it's revealed in.
const SequenceContext = createContext(null);

function SequenceProvider({ children }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const counterRef = useRef(0);
  counterRef.current = 0; // reset before this render's <Seq> children re-claim their indices
  const next = () => setActiveIndex((i) => i + 1);
  return (
    <SequenceContext.Provider value={{ activeIndex, counterRef, next }}>
      {children}
    </SequenceContext.Provider>
  );
}

// 6ms/char (~165 chars/sec) - closer to real LLM streaming pace than a
// dramatic-effect typewriter demo. This matters more here than in a
// single-message chat UI: a report with a dozen-plus sequential
// paragraphs at the wrong (slower) speed doesn't just look leisurely, it
// adds up to over a minute before the last section even starts - long
// enough that nobody's still watching by the time it finishes, which
// looks indistinguishable from "everything appeared at once."
const SEQ_SPEED = 6;

function Seq({ text, style, speed = SEQ_SPEED }) {
  const ctx = useContext(SequenceContext);
  const myIndex = useRef(null); // called unconditionally every render - Rules of Hooks
  if (!text) return null; // never claim a slot for nothing - it would never call onComplete and the sequence would stall forever waiting on it

  if (myIndex.current === null) {
    myIndex.current = ctx.counterRef.current;
    ctx.counterRef.current += 1;
  }
  const idx = myIndex.current;
  if (idx > ctx.activeIndex) return null;
  const isActive = idx === ctx.activeIndex;
  return (
    <span style={style}>
      <StreamingText
        text={text}
        speed={speed}
        cursor={isActive}
        onComplete={isActive ? ctx.next : undefined}
      />
    </span>
  );
}

// Like <Seq>, but for a claim's WHOLE block (text + its citation tags +
// its flagged-for-review note), not just the text span. Those extra
// pieces are siblings of the text in the markup, not children of it, so
// gating only the text left them free to render up front regardless of
// whether the sequence had reached that claim yet - exactly the bug that
// made citation buttons for later claims show up instantly while their
// own text was still blank. Render-prop: the caller decides what to draw
// once it's this item's turn, this only decides IF and WHEN.
function SeqBlock({ text, children }) {
  const ctx = useContext(SequenceContext);
  const myIndex = useRef(null);
  if (!text) return null;

  if (myIndex.current === null) {
    myIndex.current = ctx.counterRef.current;
    ctx.counterRef.current += 1;
  }
  const idx = myIndex.current;
  if (idx > ctx.activeIndex) return null;
  const isActive = idx === ctx.activeIndex;
  return children({ isActive, onComplete: isActive ? ctx.next : undefined });
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

// Shared between the freeform /ask answers and Phase 4's brain-effects /
// treatment sections (§2 and §5) — both are now the same underlying
// shape: {claims, sources, unsupported} from the citation-gated Phase 3
// pipeline. See clinical_agent.py's module docstring for why §2/§5 moved
// off a free-form, uncited LLM call.
function CitedClaims({ data }) {
  const claims = data?.claims || [];
  const sources = data?.sources || [];
  const unsupported = data?.unsupported || [];

  if (!claims.length && !unsupported.length) {
    return <p style={S.noEvidence}>The evidence corpus doesn't cover this.</p>;
  }

  return (
    <>
      {claims.length > 0 && (
        <ul style={S.claimList}>
          {claims.map((c, i) => {
            // A flagged claim failed the citation/hallucination check -
            // its text is exactly the thing that might be wrong, so it
            // must never render as if it were normal cited content. Swap
            // in a visible withheld-notice instead of the claim text (not
            // a silent drop - a clinician should be able to tell something
            // was removed here, not just see a shorter list) and skip its
            // citations, since they'd be citing text that isn't shown.
            const displayText = c.flagged
              ? "Claim withheld — failed automated verification against its cited source."
              : c.text;
            return (
              <SeqBlock key={i} text={displayText}>
                {({ isActive, onComplete }) => (
                  <li style={S.claimItem}>
                    <p style={c.flagged ? S.claimWithheld : S.claimText}>
                      <StreamingText text={displayText} speed={SEQ_SPEED} cursor={isActive} onComplete={onComplete} />
                    </p>
                    {c.flagged ? (
                      <p style={S.flagNote} title={c.flag_reason || ""}>
                        ⚠ {c.flag_reason || "flagged for review"}
                      </p>
                    ) : (
                      <Citations sources={sources} chunkIds={c.chunk_ids} />
                    )}
                  </li>
                )}
              </SeqBlock>
            );
          })}
        </ul>
      )}
      {/* "Not addressed by current evidence" box removed from display per
          user request - the underlying data/citation-gating is untouched,
          this is a UI-only hide. unsupported items are simply never
          rendered, not passed through Seq, so they don't occupy a slot in
          the reveal sequence either. */}
    </>
  );
}

/* --------------------------------------------- Phase 4 default analysis */
// Sections 3 and 4 are hardcoded, no model call at all. Section 1 is a
// single unsourced LLM call for genuinely general classification-level
// education. Sections 2 and 5 are citation-gated via CitedClaims above —
// see clinical_agent.py.

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
        <Fact
          label="Whole tumor (WT) volume"
          value={t.whole_tumor_volume_cc != null ? `${t.whole_tumor_volume_cc} cm³` : null}
        />
        {/* Two different metrics on two different, non-overlapping splits
            (voxel-wise on a 60-case training validation split vs.
            lesion-wise, gated, on a 293-case held-out test set) - shown as
            two separate rows, never folded into one number, so "0.71"
            can't be misread as the same kind of score as a voxel Dice. */}
        <Fact
          label="Voxel Dice (validation)"
          value={
            t.model_voxel_dice != null
              ? `${t.model_voxel_dice} — ${t.model_voxel_dice_n ?? "?"}-case training validation split`
              : null
          }
        />
        <Fact
          label="Lesion-wise Dice (held-out test)"
          value={
            t.model_holdout_dice != null
              ? `${t.model_holdout_dice} — ${t.model_holdout_dice_n ?? "?"}-case held-out test, gated`
              : null
          }
        />
        {t.model_cohort_dice_range && (
          <Fact
            label="Range by tumour-type cohort"
            value={`${t.model_cohort_dice_range[0]} – ${t.model_cohort_dice_range[1]} (model can't tell which cohort this scan is)`}
          />
        )}
        {/* Not a Fact row - this is a full cautionary sentence, not a
            short value, and was previously dominating the top of the
            panel by appearing instantly while everything else streamed. */}
        {t.scan_type_assumption && (
          <p style={S.sectionTextStrong}>
            <Seq text={t.scan_type_assumption} />
          </p>
        )}
        {/* explanation_by_category, not a single explanation string - the
            model was trained on glioma + meningioma combined with no
            tumor-type classification head, so Q1 explains both candidate
            categories rather than picking one. */}
        {t.explanation_by_category &&
          Object.entries(t.explanation_by_category).map(
            ([cat, text]) =>
              text && (
                <div key={cat} style={S.categoryBlock}>
                  <p style={S.categoryLabel}>{cat}</p>
                  <p style={S.sectionText}>
                    <Seq text={text} />
                  </p>
                </div>
              )
          )}
      </Section>

      <Section number="2" title="How can it affect the brain/body?">
        {b.locations?.length > 0 && (
          <Fact label="Location" value={b.locations.join(", ")} />
        )}
        <CitedClaims data={b} />
      </Section>

      <Section number="3" title="How long has it been growing?">
        {g.estimate && (
          <p style={S.sectionTextStrong}>
            <Seq text={g.estimate} />
          </p>
        )}
        {g.reason && (
          <p style={S.sectionTextDim}>
            <Seq text={g.reason} />
          </p>
        )}
        {g.future_capability && (
          <p style={S.sectionTextDim}>
            <Seq text={g.future_capability} />
          </p>
        )}
      </Section>

      <Section number="4" title="What is the survival outlook?">
        {p.individual_survival_probability && (
          <p style={S.sectionTextStrong}>
            <Seq text={p.individual_survival_probability} />
          </p>
        )}
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
        {p.note && (
          <p style={S.sectionTextDim}>
            <Seq text={p.note} />
          </p>
        )}
      </Section>

      <Section number="5" title="What treatments are used?">
        {/* treatment_information is now keyed by candidate category (same
            reason as Q1 above) rather than one flat {claims,sources,
            unsupported} object. */}
        {Object.entries(tr).map(([cat, data]) => (
          <div key={cat} style={S.categoryBlock}>
            <p style={S.categoryLabel}>{cat}</p>
            <CitedClaims data={data} />
          </div>
        ))}
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
          ⚠ <Seq text={w} />
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

      {!isImaging && (
        <SequenceProvider>
          <CitedClaims data={answer} />
        </SequenceProvider>
      )}
    </div>
  );
}

export default function EvidencePanel({ jobId, open = true }) {
  // Self-managed, not lifted to BrainViewer: the parent always passes
  // open (permanently, by its own comment) with no toggle of its own, so
  // closing has to live here or there'd be no way to dismiss the panel at
  // all. Dismissing never refetches - askedAnalysis below already
  // guards that, and closing/reopening just changes what's rendered.
  const [dismissed, setDismissed] = useState(false);

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

  if (dismissed) {
    return (
      <button
        type="button"
        onClick={() => setDismissed(false)}
        style={S.reopenTab}
        title="Show Lumenbrain AI"
      >
        Lumenbrain AI
      </button>
    );
  }

  return (
    <aside style={S.drawer}>
      <div style={S.header}>
        <h2 style={S.title}>Lumenbrain AI</h2>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          style={S.closeBtn}
          title="Hide"
          aria-label="Hide Lumenbrain AI panel"
        >
          ✕
        </button>
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

        {/* One shared sequence across both - warnings continue the same
            top-to-bottom reveal right after Q5 instead of appearing as a
            fully-formed box the moment the data arrives. */}
        <SequenceProvider>
          <DefaultAnalysis data={analysis} />
          <Warnings warnings={analysis?.warnings} />
        </SequenceProvider>

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
    fontFamily: FONT,
    color: INK.text,
    textShadow: "0 1px 6px rgba(0,0,0,0.85), 0 0 1px rgba(0,0,0,0.6)",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "16px 16px 0",
  },
  title: { margin: 0, fontSize: 15, fontWeight: 600 },
  closeBtn: {
    width: 26,
    height: 26,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 4,
    border: "1px solid rgba(79,216,255,0.24)",
    background: "rgba(4,10,18,0.5)",
    color: INK.dim,
    fontSize: 13,
    lineHeight: 1,
    cursor: "pointer",
  },
  reopenTab: {
    position: "absolute",
    left: 24,
    top: 92, // clears BrainViewer's back button (top:22) and hint line (top:56)
    padding: "9px 16px",
    borderRadius: 999,
    border: "1px solid rgba(79,216,255,0.3)",
    background: "rgba(4,10,18,0.7)",
    color: INK.brain,
    fontFamily: FONT,
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
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
  categoryBlock: { marginTop: 10 },
  categoryLabel: {
    margin: "0 0 4px",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    color: INK.dim,
  },

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
  claimWithheld: { margin: 0, fontSize: 12.5, lineHeight: 1.55, color: INK.dim, fontStyle: "italic" },
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
