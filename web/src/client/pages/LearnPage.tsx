import { useEffect, useRef } from "react";

import { mountLearnApp } from "../learn/main";

export function LearnPage() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    return mountLearnApp(container);
  }, []);

  return <div ref={containerRef} />;
}
