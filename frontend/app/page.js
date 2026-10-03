"use client";

import { useEffect, useRef, useState } from "react";
import AssetBoundary from "@/components/AssetBoundary";
import BrainViewer from "@/components/BrainViewer";
import HeroBrain from "@/components/HeroBrain";
import UploadPanel from "@/components/UploadPanel";

// Fades a card up into place the first time it scrolls into view, and adds
// a hover lift via the .ne-card class (see the <style> block in the page
// component) — plain inline style objects can't express :hover or
// transitions, so this is the one place the project steps outside that
// convention.
function RevealCard({ children, style, delay = 0 }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.unobserve(el);
        }
      },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="ne-card"
      style={{
        ...style,
        transitionDelay: `${delay}ms`,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(28px)",
      }}
    >
      {children}
    </div>
  );
}

const NAV_LINKS = [
  { label: "How it works", href: "#how-it-works" },
  { label: "Upload", href: "#upload" },
  { label: "Evidence", href: "#evidence" },
  { label: "Research", href: "#research" },
];

const EVIDENCE_POINTS = [
  {
    title: "Curated sources only",
    body: "NCI PDQ clinical summaries, PubMed abstracts, PMC Open Access full text, and live web search restricted server-side to ten authoritative domains — cancer.gov, nih.gov, who.int, Cochrane, ASCO, ASTRO among them. Never open web search.",
  },
  {
    title: "Every claim cited or declined",
    body: "The model never sees a question without retrieved passages, and every claim it returns either cites the specific passage it came from or is listed under “not addressed by current evidence.” Enforced in code, not just prompted for.",
  },
  {
    title: "Validated automatically",
    body: "An Evidence Validator cross-checks each claim's numbers and key terms against its own cited passage and flags anything that doesn't match for human review, rather than silently approving it.",
  },
];

const EVIDENCE_STATS = [
  { value: "94%", label: "Citation precision" },
  { value: "95%", label: "Faithfulness to cited text" },
  { value: "5/5", label: "Correctly declined the unanswerable eval questions" },
];

const RESEARCH_POINTS = [
  {
    title: "Segmentation model",
    body: "DynUNet (MONAI), trained on BraTS 2024 glioma, meningioma, and pediatric-glioma cases combined, with a post-hoc learned gate filtering predictions. Holdout-set Dice 0.7106 on 293 held-out patients — a property of the model, reported as such, never rephrased as this scan's confidence.",
  },
  {
    title: "Retrieval, measured not assumed",
    body: "Recall@20 evaluated against a 40-question hand-written set — 15 clinical, 15 research, 5 spanning both, and 5 deliberately unanswerable — comparing two embedding models empirically rather than picking one on assumption.",
  },
  {
    title: "Re-measured on every change",
    body: "Citation precision, faithfulness, and abstention rate are re-run after any material change to the corpus or retrieval pipeline, so a change that looks like an improvement has to prove it.",
  },
];

const STEPS = [
  {
    n: "01",
    title: "Upload an MRI",
    body: "Four co-registered NIfTI volumes — T1, T1CE, T2, FLAIR — from one session.",
  },
  {
    n: "02",
    title: "3D segmentation",
    body: "A DynUNet model finds the resection cavity and FLAIR hyperintensity, rendered in an interactive 3D viewer.",
  },
  {
    n: "03",
    title: "Evidence-grounded review",
    body: "A clinical agent explains the case and answers questions from a cited NCI PDQ / PubMed / PMC corpus — reviewed by a clinician, not a diagnosis.",
  },
];

export default function LandingPage() {
  const [result, setResult] = useState(null);

  // A completed segmentation replaces the whole marketing page with the
  // full 3D viewer — a distinct "working" experience (its own dark theme),
  // not a section scrolled to within the light landing page like Upload is.
  if (result) {
    return (
      <BrainViewer
        meta={result.meta}
        brainUrl={result.brainUrl}
        tumorUrl={result.tumorUrl}
        onNewCase={() => setResult(null)}
      />
    );
  }

  return (
    <div style={S.page}>
      <style>{`
        .ne-card {
          transition: opacity 0.6s ease, transform 0.6s ease,
            box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .ne-card:hover {
          transform: translateY(-6px) !important;
          box-shadow: 0 20px 44px -18px rgba(20, 40, 80, 0.3);
          border-color: rgba(59, 130, 246, 0.35);
        }
      `}</style>

      <header style={S.heroFull}>
        <nav style={S.nav}>
          <div style={S.logo}>Lumenbrain</div>
          <div style={S.navLinks}>
            {NAV_LINKS.map((l) => (
              <a key={l.href} href={l.href} style={S.navLink}>
                {l.label}
              </a>
            ))}
          </div>
          <a href="#upload" style={S.navCta}>
            Get Started
          </a>
        </nav>

        <div style={S.heroVisual}>
          {/* Decorative only - if the borrowed brain.glb fails to load
              (network hiccup, CORS, backend restart mid-fetch), this must
              degrade to an empty hero background, not take the rest of the
              landing page's cards down with it. See AssetBoundary.jsx. */}
          <AssetBoundary>
            <HeroBrain />
          </AssetBoundary>
        </div>

        <div style={S.heroContent}>
          <h1 style={S.heading}>
            AI-Assisted Brain Tumor
            <br />
            <span style={S.headingAccent}>Segmentation &amp; Evidence</span>
          </h1>
          <p style={S.subtext}>
            Upload an MRI, get a 3D tumor segmentation, and review AI-grounded
            clinical evidence — reviewed by a clinician, not a diagnosis.
          </p>
          <p style={S.disclaimer}>Research use only. Not for clinical diagnosis.</p>
        </div>
      </header>

      <section id="how-it-works" style={S.steps}>
        {STEPS.map((s, i) => (
          <RevealCard key={s.n} style={S.stepCard} delay={i * 90}>
            <span style={S.stepNum}>{s.n}</span>
            <h3 style={S.stepTitle}>{s.title}</h3>
            <p style={S.stepBody}>{s.body}</p>
          </RevealCard>
        ))}
      </section>

      <section id="evidence" style={S.section}>
        <span style={S.sectionEyebrow}>Evidence</span>
        <h2 style={S.sectionHeading}>Grounded in cited sources, not the model's memory</h2>
        <p style={S.sectionSubtext}>
          The clinical agent's five default questions explain this case's
          own segmentation directly. Anything you ask beyond that is
          answered only from a retrieval-and-citation pipeline — the model
          never supplies a medical fact from its own weights.
        </p>

        <div style={S.cardGrid}>
          {EVIDENCE_POINTS.map((p, i) => (
            <RevealCard key={p.title} style={S.stepCard} delay={i * 90}>
              <h3 style={S.stepTitle}>{p.title}</h3>
              <p style={S.stepBody}>{p.body}</p>
            </RevealCard>
          ))}
        </div>

        <div style={S.statRow}>
          {EVIDENCE_STATS.map((s) => (
            <div key={s.label} style={S.statTile}>
              <span style={S.statValue}>{s.value}</span>
              <span style={S.statLabel}>{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      <section id="research" style={S.section}>
        <span style={S.sectionEyebrow}>Research</span>
        <h2 style={S.sectionHeading}>Built and measured like a research project</h2>
        <p style={S.sectionSubtext}>
          Every component here — the segmentation model, the retrieval
          pipeline, the answer contract — has a real number behind it,
          checked against held-out data or a hand-written eval set, not
          just shipped and assumed to work.
        </p>

        <div style={S.cardGrid}>
          {RESEARCH_POINTS.map((p, i) => (
            <RevealCard key={p.title} style={S.stepCard} delay={i * 90}>
              <h3 style={S.stepTitle}>{p.title}</h3>
              <p style={S.stepBody}>{p.body}</p>
            </RevealCard>
          ))}
        </div>
      </section>

      <section id="upload" style={S.uploadSection}>
        <h2 style={S.uploadHeading}>Start a segmentation</h2>
        <p style={S.uploadSubtext}>
          Runs the same DynUNet model described above, right in your browser session.
        </p>
        <UploadPanel onResult={setResult} />
      </section>

      <footer style={S.footer}>
        <span>© 2026 Lumenbrain. All rights reserved.</span>
        <span style={S.footerDivider}>·</span>
        <span>
          Suriya, AI Researcher —{" "}
          <a href="mailto:ssuriyas380@gmail.com" style={S.footerLink}>
            ssuriyas380@gmail.com
          </a>
        </span>
      </footer>
    </div>
  );
}

const FONT = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";

const S = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(120% 70% at 30% 0%, #eef1f5 0%, #e4e8ee 45%, #dfe3ea 100%)",
    fontFamily: FONT,
    color: "#33383f",
    overflowX: "hidden",
  },
  heroFull: {
    position: "relative",
    minHeight: "100vh",
    overflow: "hidden",
    background: "#050a13",
    display: "flex",
    flexDirection: "column",
  },
  heroVisual: {
    position: "relative",
    flex: "1 1 auto",
    minHeight: 340,
  },
  nav: {
    position: "relative",
    zIndex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    width: "100%",
    padding: "28px 32px",
    boxSizing: "border-box",
  },
  logo: {
    padding: "10px 22px",
    borderRadius: 999,
    border: "1px solid rgba(125,196,255,0.35)",
    background: "rgba(255,255,255,0.06)",
    color: "#8fd0ff",
    fontWeight: 600,
    fontSize: 15,
  },
  navLinks: {
    display: "flex",
    gap: 32,
    position: "absolute",
    left: "50%",
    top: "50%",
    transform: "translate(-50%, -50%)",
  },
  navLink: {
    color: "#cfd4de",
    textDecoration: "none",
    fontSize: 14.5,
  },
  navCta: {
    padding: "11px 22px",
    borderRadius: 999,
    background: "#fff",
    boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
    color: "#20242b",
    textDecoration: "none",
    fontWeight: 600,
    fontSize: 14,
  },
  heroContent: {
    flex: "0 0 auto",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    padding: "0 24px 56px",
  },
  heading: {
    margin: 0,
    fontSize: "clamp(36px, 6vw, 68px)",
    fontWeight: 600,
    lineHeight: 1.08,
    letterSpacing: "-0.02em",
    color: "#f4f7fb",
  },
  headingAccent: { color: "#7dc4ff" },
  subtext: {
    margin: "26px 0 0",
    fontSize: 18,
    lineHeight: 1.6,
    color: "#c7ccd6",
    maxWidth: 620,
  },
  disclaimer: { margin: "28px 0 0", fontSize: 12.5, color: "#8a94a8" },

  steps: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: 24,
    maxWidth: 1100,
    margin: "0 auto",
    padding: "20px 32px 40px",
  },
  stepCard: {
    padding: "28px 26px",
    borderRadius: 14,
    background: "rgba(255,255,255,0.65)",
    border: "1px solid rgba(30,40,60,0.08)",
  },
  stepNum: { fontSize: 13, fontWeight: 700, color: "#3b82f6" },
  stepTitle: { margin: "10px 0 8px", fontSize: 18, fontWeight: 600, color: "#2c3038" },
  stepBody: { margin: 0, fontSize: 14, lineHeight: 1.6, color: "#6b7180" },

  section: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: "56px 32px 20px",
    textAlign: "center",
    scrollMarginTop: 24,
  },
  sectionEyebrow: {
    display: "inline-block",
    fontSize: 12.5,
    fontWeight: 700,
    letterSpacing: "0.06em",
    textTransform: "uppercase",
    color: "#3b82f6",
  },
  sectionHeading: {
    margin: "10px 0 0",
    fontSize: "clamp(26px, 3.6vw, 38px)",
    fontWeight: 600,
    letterSpacing: "-0.015em",
    color: "#2c3038",
  },
  sectionSubtext: {
    margin: "14px auto 0",
    maxWidth: 640,
    fontSize: 15.5,
    lineHeight: 1.65,
    color: "#6b7180",
  },
  cardGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: 20,
    marginTop: 36,
    textAlign: "left",
  },
  statRow: {
    display: "flex",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: 36,
    marginTop: 44,
    paddingTop: 36,
    borderTop: "1px solid rgba(30,40,60,0.08)",
  },
  statTile: { display: "flex", flexDirection: "column", alignItems: "center" },
  statValue: { fontSize: 34, fontWeight: 700, color: "#3b82f6", lineHeight: 1 },
  statLabel: {
    marginTop: 8,
    maxWidth: 170,
    fontSize: 12.5,
    lineHeight: 1.5,
    color: "#6b7180",
    textAlign: "center",
  },

  uploadSection: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    padding: "40px 24px 110px",
    scrollMarginTop: 24,
  },
  uploadHeading: {
    margin: 0,
    fontSize: "clamp(28px, 4vw, 40px)",
    fontWeight: 600,
    letterSpacing: "-0.015em",
    color: "#2c3038",
  },
  uploadSubtext: {
    margin: "12px 0 36px",
    fontSize: 15,
    color: "#6b7180",
  },

  footer: {
    display: "flex",
    flexWrap: "wrap",
    justifyContent: "center",
    alignItems: "center",
    gap: 10,
    padding: "28px 24px 36px",
    borderTop: "1px solid rgba(30,40,60,0.08)",
    fontSize: 13,
    color: "#8a90a0",
    textAlign: "center",
  },
  footerDivider: { color: "#c3c8d2" },
  footerLink: { color: "#3b82f6", textDecoration: "none" },
};
