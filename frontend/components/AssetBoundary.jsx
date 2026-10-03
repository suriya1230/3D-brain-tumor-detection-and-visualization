"use client";

import React from "react";

// A malformed/unreachable .glb throws inside Suspense, which without a
// boundary unmounts the whole subtree it's in - and if that subtree isn't
// isolated from the rest of the page (e.g. HeroBrain sitting directly in
// the landing page's JSX, nothing wrapping it), React unmounts everything
// else on the page too, not just the failed 3D viewport. Originally local
// to BrainViewer.jsx; extracted so HeroBrain (page.js's decorative landing
// hero) gets the same isolation instead of being able to take the whole
// landing page down with it.
export default class AssetBoundary extends React.Component {
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
    return this.state.failed ? (this.props.fallback ?? null) : this.props.children;
  }
}
