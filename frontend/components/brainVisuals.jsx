"use client";

// Shared rendering primitives for the "neural glow" brain look — used by
// both BrainViewer.jsx (the app's case viewer) and HeroBrain.jsx (the
// landing page). Extracted so the two don't drift into two slightly
// different, independently-maintained copies of the same shaders.
//
// Important: everything additive here (the fresnel shell, node glow,
// synapse lines) needs a DARK backdrop to read at all — additive blending
// only adds brightness, so on a light background it just washes out to
// nothing. Any consumer of this module must render against something dark
// (INK.void/deep, or an equivalent), not a light page background directly.

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ------------------------------------------------- x-ray fresnel material */
export function useFresnel(color, { power = 2.4, strength = 1.0, depthTest = true } = {}) {
  return useMemo(
    () =>
      new THREE.ShaderMaterial({
        depthTest,
        uniforms: {
          uColor: { value: new THREE.Color(color) },
          uPower: { value: power },
          uStrength: { value: strength },
        },
        vertexShader: `
          varying vec3 vN; varying vec3 vV;
          void main() {
            vec4 wp = modelMatrix * vec4(position, 1.0);
            vN = normalize(mat3(modelMatrix) * normal);
            vV = normalize(cameraPosition - wp.xyz);
            gl_Position = projectionMatrix * viewMatrix * wp;
          }`,
        fragmentShader: `
          uniform vec3 uColor; uniform float uPower; uniform float uStrength;
          varying vec3 vN; varying vec3 vV;
          void main() {
            float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), uPower);
            gl_FragColor = vec4(uColor * f * uStrength, f);
          }`,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.FrontSide,
      }),
    [color, power, strength, depthTest]
  );
}

/* --------------------------------------------------- deterministic random */
// Seeded so node placement and pulse phases stay identical across
// re-renders; a fresh Math.random() set on every render makes the glow
// visibly jump.
export function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The backend exports POSITION and indices only, so anything that shades by
// normal (the fresnel shell, a standard material) gets a zero vector and
// renders as a flat blown-out silhouette until these exist.
export function ensureNormals(geometry) {
  if (geometry && !geometry.attributes.normal) geometry.computeVertexNormals();
  return geometry;
}

export function sampleSurface(geometry, count) {
  const pos = geometry.attributes.position;
  const nor = geometry.attributes.normal;
  if (!geometry.boundingSphere) geometry.computeBoundingSphere();
  const centre = geometry.boundingSphere.center;
  const total = pos.count;
  const stride = Math.max(1, Math.floor(total / count));
  const v = new THREE.Vector3();
  const n = new THREE.Vector3();
  const out = [];
  for (let i = 0; i < total && out.length < count; i += stride) {
    v.fromBufferAttribute(pos, i);
    if (nor) n.fromBufferAttribute(nor, i).normalize();
    else n.copy(v).sub(centre).normalize();
    out.push({ p: v.clone(), n: n.clone() });
  }
  return out;
}

/* --------------------------------------------------------- glow sprite map */
export function useGlowSprite() {
  return useMemo(() => {
    const size = 128;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d");
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0.0, "rgba(255,255,255,1)");
    g.addColorStop(0.18, "rgba(255,255,255,0.85)");
    g.addColorStop(0.42, "rgba(255,255,255,0.28)");
    g.addColorStop(1.0, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }, []);
}

export const NODE_COOL = ["#a8f4ff", "#ffffff", "#6fe0ff", "#d6faff"];
// Warm accents read as "firing" without borrowing the crimson the app's ET
// (enhancing tumour) overlay uses, so a hotspot can never be mistaken for
// tumour tissue in the case viewer. Kept here too for visual consistency
// between the landing page and the app.
export const NODE_WARM = ["#ff9a3c", "#ffc85e", "#ff7d2e"];

/* ------------------------------------------------------- firing node field */
export function NeuralNodes({ geometry, radius, count = 240 }) {
  const map = useGlowSprite();

  const attrs = useMemo(() => {
    const rand = mulberry32(0x5eed);
    const pts = sampleSurface(geometry, count);
    const position = new Float32Array(pts.length * 3);
    const colour = new Float32Array(pts.length * 3);
    const phase = new Float32Array(pts.length);
    const scale = new Float32Array(pts.length);
    const c = new THREE.Color();

    pts.forEach(({ p, n }, i) => {
      const lift = radius * 0.004 + rand() * radius * 0.012;
      position[i * 3] = p.x + n.x * lift;
      position[i * 3 + 1] = p.y + n.y * lift;
      position[i * 3 + 2] = p.z + n.z * lift;

      const warm = rand() < 0.26;
      const pool = warm ? NODE_WARM : NODE_COOL;
      c.set(pool[Math.floor(rand() * pool.length)]);
      colour[i * 3] = c.r;
      colour[i * 3 + 1] = c.g;
      colour[i * 3 + 2] = c.b;

      phase[i] = rand();
      // A few oversized nodes carry the composition; the rest are filler.
      scale[i] = warm ? 0.8 + rand() * 1.5 : 0.28 + rand() * 0.55;
    });
    return { position, colour, phase, scale, n: pts.length };
  }, [geometry, radius, count]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uMap: { value: map },
          uSize: { value: radius * 0.06 },
        },
        vertexShader: `
          attribute float aPhase; attribute float aScale; attribute vec3 aColor;
          uniform float uTime; uniform float uSize;
          varying vec3 vColor; varying float vPulse;
          void main() {
            vColor = aColor;
            float p = 0.35 + 0.65 * pow(
              0.5 + 0.5 * sin(uTime * 1.7 + aPhase * 6.2831), 1.6
            );
            vPulse = p;
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = uSize * aScale * (0.55 + 0.45 * p) * (260.0 / -mv.z);
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: `
          uniform sampler2D uMap;
          varying vec3 vColor; varying float vPulse;
          void main() {
            float a = texture2D(uMap, gl_PointCoord).a;
            gl_FragColor = vec4(vColor * (0.8 + 1.9 * vPulse), 1.0) * a;
          }`,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [map, radius]
  );
  useFrame(({ clock }) => {
    material.uniforms.uTime.value = clock.elapsedTime;
  });

  return (
    <points material={material} renderOrder={6}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[attrs.position, 3]} />
        <bufferAttribute attach="attributes-aColor" args={[attrs.colour, 3]} />
        <bufferAttribute attach="attributes-aPhase" args={[attrs.phase, 1]} />
        <bufferAttribute attach="attributes-aScale" args={[attrs.scale, 1]} />
      </bufferGeometry>
    </points>
  );
}

/* --------------------------------------------------------- radiating rays */
export function Synapses({ geometry, radius, count = 60, opacity = 0.16 }) {
  const attrs = useMemo(() => {
    const rand = mulberry32(0xbeef);
    const pts = sampleSurface(geometry, count * 3).filter(() => rand() < 0.34);
    const position = new Float32Array(pts.length * 6);
    const colour = new Float32Array(pts.length * 6);
    const head = new THREE.Color("#bff4ff");
    const dir = new THREE.Vector3();

    pts.forEach(({ p, n }, i) => {
      const len = radius * (0.03 + rand() * 0.085);
      dir
        .copy(n)
        .add(new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).multiplyScalar(0.35))
        .normalize();

      position[i * 6] = p.x;
      position[i * 6 + 1] = p.y;
      position[i * 6 + 2] = p.z;
      position[i * 6 + 3] = p.x + dir.x * len;
      position[i * 6 + 4] = p.y + dir.y * len;
      position[i * 6 + 5] = p.z + dir.z * len;

      // Bright at the cortex, fading to nothing at the tip.
      colour[i * 6] = head.r;
      colour[i * 6 + 1] = head.g;
      colour[i * 6 + 2] = head.b;
      colour[i * 6 + 3] = 0;
      colour[i * 6 + 4] = 0;
      colour[i * 6 + 5] = 0;
    });
    return { position, colour };
  }, [geometry, radius, count]);

  return (
    <lineSegments renderOrder={5}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[attrs.position, 3]} />
        <bufferAttribute attach="attributes-color" args={[attrs.colour, 3]} />
      </bufferGeometry>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={opacity}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </lineSegments>
  );
}

/* ------------------------------------------------------- background bokeh */
export function Dust({ radius, count = 260, color = "#4fd8ff" }) {
  const map = useGlowSprite();
  const ref = useRef();

  const attrs = useMemo(() => {
    const rand = mulberry32(0xd057);
    const position = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = radius * (1.5 + rand() * 2.1);
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      position[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      position[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      position[i * 3 + 2] = r * Math.cos(phi);
    }
    return { position };
  }, [radius, count]);

  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.012;
  });

  return (
    <points ref={ref} renderOrder={0}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[attrs.position, 3]} />
      </bufferGeometry>
      <pointsMaterial
        map={map}
        color={color}
        size={radius * 0.05}
        sizeAttenuation
        transparent
        opacity={0.28}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
