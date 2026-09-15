"use client";

import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { Dust, NeuralNodes, Synapses, ensureNormals, useFresnel } from "./brainVisuals";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// A real segmentation mesh from an existing case, used purely as a
// decorative anatomical shape here — no tumor overlay, no measurements.
// There's no synthetic "generic brain" asset in this project; every mesh
// comes from an actual segmentation, so this borrows one rather than
// fabricating a stand-in.
const BRAIN_URL = `${API}/api/jobs/06bb40cb7fae/brain.glb`;

const STAGE = {
  void: "#050a13",
  deep: "#0a1c30",
  brain: "#4fd8ff",
};

function CameraRig({ radius }) {
  // Looks at the world origin — RotatingBrain re-centers the mesh to
  // origin itself, so "distance along Z, looking at 0,0,0" is correct
  // regardless of where the source mesh's own coordinates put its
  // bounding sphere.
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 0, radius * 2.5);
    camera.near = radius * 0.05;
    camera.far = radius * 20;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, radius]);
  return null;
}

function RotatingBrain() {
  const { scene } = useGLTF(BRAIN_URL);
  const shell = useFresnel(STAGE.brain, { power: 2.0, strength: 1.4 });
  const spinRef = useRef();

  const geometry = useMemo(() => {
    let g = null;
    scene.traverse((o) => {
      if (o.isMesh && !g) g = o.geometry;
    });
    return ensureNormals(g);
  }, [scene]);

  const edges = useMemo(() => (geometry ? new THREE.EdgesGeometry(geometry, 9) : null), [geometry]);

  const { radius, center } = useMemo(() => {
    if (!geometry) return { radius: 90, center: new THREE.Vector3() };
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    return { radius: geometry.boundingSphere.radius, center: geometry.boundingSphere.center };
  }, [geometry]);

  // Spins in place around its own center — a plain rotation on a mesh
  // whose geometry isn't centered at the origin orbits it in a wide,
  // wobbly loop instead, since three.js always rotates about local origin.
  useFrame((_, dt) => {
    if (spinRef.current) spinRef.current.rotation.y += dt * 0.3;
  });

  if (!geometry) return null;
  return (
    <>
      <CameraRig radius={radius} />
      <group ref={spinRef} rotation={[0.12, 0, 0]}>
        <group position={[-center.x, -center.y, -center.z]}>
          {/* Depth-only prepass — without it, additive layers through a
              folded cortex stack ~15 deep along one ray and blow out to
              white (the exact bug fixed in the app viewer earlier). */}
          <mesh geometry={geometry} renderOrder={1}>
            <meshBasicMaterial colorWrite={false} polygonOffset polygonOffsetFactor={1} polygonOffsetUnits={1} />
          </mesh>
          <mesh geometry={geometry} renderOrder={2}>
            <meshBasicMaterial color={STAGE.deep} transparent opacity={0.14} depthWrite={false} />
          </mesh>
          <mesh geometry={geometry} material={shell} renderOrder={3} />
          <lineSegments geometry={edges} renderOrder={4}>
            <lineBasicMaterial color="#9beeff" transparent opacity={0.3} depthWrite={false} blending={THREE.AdditiveBlending} />
          </lineSegments>
          <Synapses geometry={geometry} radius={radius} opacity={0.22} />
          <NeuralNodes geometry={geometry} radius={radius} count={180} />
        </group>
      </group>
      <Dust radius={radius} count={160} color={STAGE.brain} />
    </>
  );
}

export default function HeroBrain({ className, style }) {
  return (
    <div className={className} style={{ width: "100%", height: "100%", ...style }}>
      <Canvas
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
        camera={{ fov: 42 }}
        onCreated={({ gl }) => gl.setClearColor(STAGE.void)}
      >
        <ambientLight intensity={0.35} />
        <directionalLight position={[2, 3, 4]} intensity={1} />
        <directionalLight position={[-2, -1, -3]} intensity={0.4} color={STAGE.brain} />
        <Suspense fallback={null}>
          <RotatingBrain />
        </Suspense>
        <EffectComposer disableNormalPass frameBufferType={THREE.UnsignedByteType}>
          <Bloom intensity={1.1} luminanceThreshold={0.28} luminanceSmoothing={0.5} mipmapBlur radius={0.75} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
