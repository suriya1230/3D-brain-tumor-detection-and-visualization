"use client";

import { useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import BrainViewer from "@/components/BrainViewer";

export default function Page() {
  const [result, setResult] = useState(null);

  if (!result) return <UploadPanel onResult={setResult} />;

  return (
    <BrainViewer
      meta={result.meta}
      brainUrl={result.brainUrl}
      tumorUrl={result.tumorUrl}
      onNewCase={() => setResult(null)}
    />
  );
}
