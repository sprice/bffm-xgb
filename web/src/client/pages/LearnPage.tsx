import { useEffect, useRef } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";

import { chapters } from "../learn/content";
import { mountLearnApp, type LearnAppHandle } from "../learn/main";
import {
  defaultLearnPath,
  isLearnChapterSlug,
} from "../learn/routes";
import { NotFoundPage } from "./NotFoundPage";

export function LearnPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<LearnAppHandle | null>(null);
  const navigate = useNavigate();
  const { chapterSlug } = useParams<{ chapterSlug: string }>();

  const isValidSlug = isLearnChapterSlug(chapterSlug);
  const chapter = isValidSlug
    ? chapters.find((item) => item.slug === chapterSlug)
    : undefined;

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !chapterSlug || !isValidSlug) return;

    const app = mountLearnApp(container, {
      initialSlug: chapterSlug,
      onNavigate: (slug) => navigate(`/learn/${slug}`),
    });
    appRef.current = app;

    return () => {
      app.destroy();
      appRef.current = null;
    };
  }, [navigate, isValidSlug, chapterSlug]);

  useEffect(() => {
    if (!chapter) return;
    document.title = `${chapter.title} - Big Five Learning Guide`;
  }, [chapter]);

  if (!chapterSlug) {
    return <Navigate to={defaultLearnPath} replace />;
  }

  if (!isValidSlug) {
    return <NotFoundPage />;
  }

  return <div ref={containerRef} className="h-full w-full" />;
}
