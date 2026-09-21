"use client";

import { useEffect, useState } from "react";

// Same shape as the @animui/ui StreamingText API (text/speed/cursor/
// onComplete) - that package isn't a real npm dependency (checked: 404 on
// the registry), it's copy-paste example code from a component site, so
// this is a local component matching that same interface rather than an
// installed one.
//
// Reveals `text` progressively (typewriter), never generates it live -
// see EvidencePanel.jsx for why: every string this renders has already
// passed this app's citation validators before it ever reaches the
// frontend, and that has to stay true. `startDelay` is an addition beyond
// the reference API, used to stagger multiple claims in a list so they
// don't all start typing at once.
export default function StreamingText({
  text,
  speed = 20,
  cursor = false,
  onComplete,
  startDelay = 0,
  style,
  className,
}) {
  const [shown, setShown] = useState("");

  useEffect(() => {
    setShown("");
    if (!text) return undefined;

    let i = 0;
    let intervalId;
    const timeoutId = setTimeout(() => {
      intervalId = setInterval(() => {
        i += 1;
        setShown(text.slice(0, i));
        if (i >= text.length) {
          clearInterval(intervalId);
          onComplete?.();
        }
      }, speed);
    }, startDelay);

    return () => {
      clearTimeout(timeoutId);
      if (intervalId) clearInterval(intervalId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, speed, startDelay]);

  const done = shown.length >= text.length;
  return (
    <span className={className} style={style}>
      {shown}
      {cursor && !done && <span aria-hidden style={{ opacity: 0.75 }}>▍</span>}
    </span>
  );
}

export { StreamingText };
