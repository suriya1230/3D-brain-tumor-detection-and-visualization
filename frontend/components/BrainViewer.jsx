"use client";

import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { Dust, NeuralNodes, Synapses, ensureNormals, useFresnel } from "./brainVisuals";
import EvidencePanel from "./EvidencePanel";
import { INK, REGION } from "./theme";

/* --------------------------------------------------- screen-space backdrop */
// Drawn in-scene rather than as CSS behind a transparent canvas: the bloom
// pass renders to its own target, and an alpha canvas comes back empty.
function Backdrop() {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uInner: { value: new THREE.Color(INK.halo) },
          uMid: { value: new THREE.Color(INK.deep) },
          uOuter: { value: new THREE.Color(INK.void) },
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position.xy, 1.0, 1.0);
          }`,
        fragmentShader: `
          uniform vec3 uInner; uniform vec3 uMid; uniform vec3 uOuter;
          varying vec2 vUv;
          void main() {
            vec2 p = (vUv - vec2(0.5, 0.52)) * vec2(1.35, 1.0);
            float d = clamp(length(p) * 1.85, 0.0, 1.0);
            vec3 c = mix(uInner, uMid, smoothstep(0.0, 0.5, d));
            c = mix(c, uOuter, smoothstep(0.45, 1.0, d));
            gl_FragColor = vec4(c, 1.0);
          }`,
        depthTest: false,
        depthWrite: false,
      }),
    []
  );

  return (
    <mesh material={material} frustumCulled={false} renderOrder={-1000}>
      <planeGeometry args={[2, 2]} />
    </mesh>
  );
}

function Brain({ url, visible, wire, glow }) {
  const { scene } = useGLTF(url);
  const shell = useFresnel(INK.brain, { power: 2.1, strength: 1.25 });
  const geometry = useMemo(() => {
    let g = null;
    scene.traverse((o) => {
      if (o.isMesh && !g) g = o.geometry;
    });
    return ensureNormals(g);
  }, [scene]);

  const radius = useMemo(() => {
    if (!geometry) return 90;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    return geometry.boundingSphere.radius;
  }, [geometry]);

  // A full wireframe of ~56k faces is a solid haze at screen size. Creased
  // edges only trace the sulcal ridges, which is what reads as a network.
  const edges = useMemo(
    () => (geometry ? new THREE.EdgesGeometry(geometry, 9) : null),
    [geometry]
  );

  if (!geometry) return null;
  return (
    <group visible={visible}>
      {/* Depth-only prepass. Without it the additive layers of a folded cortex
          stack ~15 deep along every ray and saturate the silhouette to white. */}
      <mesh geometry={geometry} renderOrder={1}>
        {/* Offset back a hair so the shell below passes the depth test cleanly
            inside narrow sulci, where equal-depth fragments speckle. */}
        <meshBasicMaterial
          colorWrite={false}
          polygonOffset
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      </mesh>

      {/* Faint body fill so the cortex reads as volume rather than outline. */}
      <mesh geometry={geometry} renderOrder={2}>
        <meshBasicMaterial
          color={INK.core}
          transparent
          opacity={0.18}
          depthWrite={false}
        />
      </mesh>

      <mesh geometry={geometry} material={shell} renderOrder={3} />

      {wire && (
        <lineSegments geometry={edges} renderOrder={4}>
          <lineBasicMaterial
            color={INK.rim}
            transparent
            opacity={0.2}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </lineSegments>
      )}

      {glow && (
        <>
          <Synapses geometry={geometry} radius={radius} />
          <NeuralNodes geometry={geometry} radius={radius} />
        </>
      )}
    </group>
  );
}

function Region({ geometry, color, visible }) {
  // depthTest off so the halo survives the brain's depth prepass: the tumour
  // sits inside the shell and would otherwise be culled by it.
  const halo = useFresnel(color, { power: 1.7, strength: 1.4, depthTest: false });
  return (
    <group visible={visible}>
      {/* Opaque, and ordered ahead of the brain prepass, so the core lands in
          the colour buffer before the shell glows over it. */}
      <mesh geometry={geometry} renderOrder={0}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.9}
          roughness={0.35}
          metalness={0.05}
        />
      </mesh>
      <mesh geometry={geometry} material={halo} scale={1.045} renderOrder={20} />
    </group>
  );
}

function Tumours({ url, shown }) {
  const { scene } = useGLTF(url);
  const parts = useMemo(() => {
    const found = [];
    scene.traverse((o) => {
      if (!o.isMesh) return;
      const key = Object.keys(REGION).find(
        (k) => o.name === k || o.name.startsWith(k) || o.parent?.name === k
      );
      if (key) found.push({ key, geometry: ensureNormals(o.geometry) });
    });
    return found;
  }, [scene]);

  return (
    <group>
      {parts.map(({ key, geometry }, i) => (
        <Region
          key={`${key}-${i}`}
          geometry={geometry}
          color={REGION[key].color}
          visible={!!shown[key]}
        />
      ))}
    </group>
  );
}

/* ----------------------------------------------------------- error guard */
// A malformed .glb throws inside Suspense, which without a boundary unmounts
// the whole page including the panel that would explain the failure.
class AssetBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(err) {
    this.props.onError?.(err?.message || "could not load the 3D assets");
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function Rig({ radius }) {
  const { camera } = useThree();

  useEffect(() => {
    const d = radius * 3.1;
    camera.position.set(d * 0.55, d * 0.28, d * 0.78);
    camera.near = radius * 0.05;
    camera.far = radius * 40;
    camera.updateProjectionMatrix();
  }, [camera, radius]);

  return null;
}

/* ------------------------------------------------------------------ view */
export default function BrainViewer({ meta, brainUrl, tumorUrl, onNewCase }) {
  const [brainOn, setBrainOn] = useState(true);
  const [wire, setWire] = useState(true);
  const [glow, setGlow] = useState(true);
  const [spin, setSpin] = useState(false);
  const [shown, setShown] = useState({});
  const [error, setError] = useState(null);
  const [resetKey, setResetKey] = useState(0);
  const wrapRef = useRef();

  useEffect(() => {
    const init = {};
    Object.entries(meta?.regions ?? {}).forEach(([k, v]) => {
      init[k] = v.default_visible !== false;
    });
    setShown(init);
    setError(null);
  }, [meta]);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setSpin(false);
    }
  }, []);

  const radius = meta?.scene_radius_mm ?? 90;
  const regions = meta?.regions ?? {};

  const fullscreen = () => {
    if (!document.fullscreenElement) wrapRef.current?.requestFullscreen?.();
    else document.exitFullscreen?.();
  };

  return (
    <div ref={wrapRef} style={S.wrap}>
      <Canvas
        key={`${brainUrl}-${resetKey}`}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
        camera={{ fov: 38 }}
        onCreated={({ gl }) => gl.setClearColor(INK.void)}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[1, 2, 3]} intensity={1.1} />
        <directionalLight
          position={[-2, -1, -2]}
          intensity={0.45}
          color={INK.brain}
        />

        <Backdrop />
        <Rig radius={radius} />
        <Dust radius={radius} />

        <AssetBoundary onError={setError}>
          <Suspense fallback={null}>
            <Brain url={brainUrl} visible={brainOn} wire={wire} glow={glow} />
            <Tumours url={tumorUrl} shown={shown} />
          </Suspense>
        </AssetBoundary>

        <OrbitControls
          makeDefault
          enablePan
          enableDamping
          dampingFactor={0.08}
          autoRotate={spin}
          autoRotateSpeed={0.55}
          minDistance={radius * 1.15}
          maxDistance={radius * 7}
        />

        <EffectComposer disableNormalPass frameBufferType={THREE.UnsignedByteType}>
          <Bloom
            intensity={0.9}
            luminanceThreshold={0.3}
            luminanceSmoothing={0.5}
            mipmapBlur
            radius={0.7}
          />
        </EffectComposer>
      </Canvas>

      <aside style={S.panel}>
        {error && <p style={S.error}>{error}</p>}

        <h1 style={S.caseId}>{meta.case_id}</h1>
        <p style={S.sub}>
          SegResNet · {meta.checkpoint} · {meta.elapsed_s}s
        </p>

        {meta.derived?.WT_cc != null && (
          <div style={S.hero}>
            <span style={S.heroNum}>{meta.derived.WT_cc.toFixed(1)}</span>
            <span style={S.heroUnit}>cm³ whole tumour</span>
          </div>
        )}

        <ul style={S.list}>
          {Object.entries(regions).map(([key, r]) => (
            <li key={key} style={S.row}>
              <button
                onClick={() => setShown((s) => ({ ...s, [key]: !s[key] }))}
                aria-pressed={!!shown[key]}
                title={shown[key] ? `Hide ${key}` : `Show ${key}`}
                style={{
                  ...S.swatch,
                  background: shown[key] ? REGION[key]?.color : "transparent",
                  borderColor: REGION[key]?.color ?? INK.dim,
                }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={S.rowTop}>
                  <span style={S.rowName}>{REGION[key]?.label ?? key}</span>
                  <span style={S.rowVol}>{r.volume_cc.toFixed(2)} cm³</span>
                </div>
                <div style={S.rowMeta}>
                  {r.location ?? "location not derived"}
                  {r.mean_probability != null &&
                    ` · mean p ${r.mean_probability.toFixed(2)}`}
                </div>
              </div>
            </li>
          ))}
        </ul>

        {meta.derived?.ET_WT_ratio != null && (
          <p style={S.ratio}>
            Enhancing fraction {(meta.derived.ET_WT_ratio * 100).toFixed(1)}%
            {meta.derived.TC_cc != null &&
              ` · tumour core ${meta.derived.TC_cc.toFixed(2)} cm³`}
          </p>
        )}

        <div style={S.links}>
          <a href={meta.assets.mask} download style={S.link}>
            Download mask (NIfTI)
          </a>
        </div>

        <p style={S.note}>{meta.notes}</p>
      </aside>

      <div style={S.controls}>
        <Btn on={brainOn} onClick={() => setBrainOn((v) => !v)}>
          Brain surface
        </Btn>
        <Btn on={wire} onClick={() => setWire((v) => !v)}>
          Mesh lines
        </Btn>
        <Btn on={glow} onClick={() => setGlow((v) => !v)}>
          Neural glow
        </Btn>
        <Btn on={spin} onClick={() => setSpin((v) => !v)}>
          Auto-rotate
        </Btn>
        <Btn onClick={() => setResetKey((k) => k + 1)}>Reset view</Btn>
        <Btn onClick={fullscreen}>Fullscreen</Btn>
        <Btn onClick={onNewCase}>New scan</Btn>
      </div>

      <button onClick={onNewCase} style={S.backBtn}>
        ← Home
      </button>
      <p style={S.hint}>Drag to rotate · scroll to zoom · right-drag to pan</p>

      <EvidencePanel jobId={meta.job_id} open />
    </div>
  );
}

function Btn({ on, onClick, children }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={on === undefined ? undefined : on}
      style={{
        ...S.btn,
        color: on ? INK.void : INK.text,
        background: on ? INK.brain : "rgba(79,216,255,0.06)",
        borderColor: on ? INK.brain : "rgba(79,216,255,0.28)",
      }}
    >
      {children}
    </button>
  );
}

const FONT = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";

const S = {
  wrap: {
    position: "relative",
    width: "100%",
    height: "100dvh",
    background: `radial-gradient(115% 85% at 58% 45%, ${INK.halo} 0%, ${INK.deep} 38%, ${INK.void} 78%)`,
    fontFamily: FONT,
    color: INK.text,
    overflow: "hidden",
  },
  panel: {
    position: "absolute",
    top: 24,
    right: 24,
    width: "min(330px, calc(100vw - 48px))",
    maxHeight: "calc(100dvh - 48px)",
    overflowY: "auto",
    padding: "22px 22px 18px",
    borderRadius: 4,
    border: "1px solid rgba(79,216,255,0.16)",
    background: "rgba(4,10,18,0.72)",
    backdropFilter: "blur(14px)",
  },
  caseId: { margin: 0, fontSize: 17, fontWeight: 600, letterSpacing: "-0.01em" },
  sub: { margin: "5px 0 18px", fontSize: 12, color: INK.dim, lineHeight: 1.5 },
  hero: {
    display: "flex",
    alignItems: "baseline",
    gap: 8,
    paddingBottom: 16,
    marginBottom: 16,
    borderBottom: "1px solid rgba(79,216,255,0.14)",
  },
  heroNum: {
    fontSize: 40,
    fontWeight: 300,
    lineHeight: 1,
    color: INK.rim,
    fontVariantNumeric: "tabular-nums",
  },
  heroUnit: { fontSize: 12, color: INK.dim },
  list: { listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 13 },
  row: { display: "flex", gap: 11, alignItems: "flex-start" },
  swatch: {
    flex: "0 0 auto",
    width: 13,
    height: 13,
    marginTop: 3,
    borderRadius: 3,
    borderWidth: 1,
    borderStyle: "solid",
    cursor: "pointer",
    padding: 0,
  },
  rowTop: { display: "flex", justifyContent: "space-between", gap: 8 },
  rowName: { fontSize: 13, fontWeight: 500 },
  rowVol: { fontSize: 13, fontVariantNumeric: "tabular-nums", color: INK.rim },
  rowMeta: { fontSize: 11, color: INK.dim, marginTop: 2, lineHeight: 1.45 },
  ratio: {
    margin: "16px 0 0",
    paddingTop: 14,
    borderTop: "1px solid rgba(79,216,255,0.14)",
    fontSize: 12,
  },
  links: { marginTop: 14 },
  link: { fontSize: 12, color: INK.brain, textDecoration: "none" },
  note: { margin: "14px 0 0", fontSize: 10.5, lineHeight: 1.55, color: "#43606f" },
  error: {
    margin: "0 0 12px",
    fontSize: 12.5,
    lineHeight: 1.55,
    color: "#ff8a9c",
  },
  controls: {
    // Top-center, not bottom-left — the Evidence panel is permanently open
    // now (no toggle button anymore) and occupies the whole left column
    // top-to-bottom, so bottom-left is hidden underneath it.
    position: "absolute",
    top: 24,
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: 8,
    maxWidth: "min(560px, calc(100vw - 48px))",
  },
  btn: {
    padding: "7px 13px",
    fontFamily: FONT,
    fontSize: 12,
    borderRadius: 3,
    borderWidth: 1,
    borderStyle: "solid",
    cursor: "pointer",
    transition: "background 140ms ease, color 140ms ease",
  },
  hint: {
    position: "absolute",
    left: 26,
    top: 56,
    margin: 0,
    fontSize: 11,
    color: INK.dim,
  },
  backBtn: {
    position: "absolute",
    left: 24,
    top: 22,
    padding: "7px 13px",
    fontFamily: FONT,
    fontSize: 12,
    fontWeight: 500,
    borderRadius: 3,
    border: "1px solid rgba(79,216,255,0.28)",
    background: "rgba(79,216,255,0.06)",
    color: INK.text,
    cursor: "pointer",
  },
};
