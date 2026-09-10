export const INK = {
  void: "#02060e",
  deep: "#061a2e",
  halo: "#0e3a5c",
  core: "#1c6ea8",
  brain: "#4fd8ff",
  rim: "#9beeff",
  text: "#c8e4f0",
  dim: "#5d7f92",
};

export const REGION = {
  ET: { color: "#ff2d55", label: "Enhancing tissue" },
  NETC: { color: "#ffb020", label: "Non-enhancing core" },
  RC: { color: "#8b7bff", label: "Resection cavity" },
  SNFH: { color: "#2e7d8f", label: "FLAIR hyperintensity" },
};

export const MODALITIES = [
  { key: "t1n", label: "T1", hint: "no contrast" },
  { key: "t1c", label: "T1CE", hint: "with contrast" },
  { key: "t2w", label: "T2", hint: "" },
  { key: "t2f", label: "FLAIR", hint: "T2-FLAIR" },
];
