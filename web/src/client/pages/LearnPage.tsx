import { useEffect } from "react";
import { Navigate, useParams } from "react-router-dom";

import { LearnShell } from "../learn/components/LearnShell";
import { chapters } from "../learn/content";
import { defaultLearnPath } from "../learn/routes";
import { NotFoundPage } from "./NotFoundPage";

export function LearnPage() {
  const { chapterSlug } = useParams<{ chapterSlug: string }>();
  const chapter = chapterSlug
    ? chapters.find((item) => item.slug === chapterSlug)
    : undefined;

  useEffect(() => {
    if (chapter) {
      document.title = `${chapter.title} - Big Five Learning Guide`;
    }
  }, [chapter]);

  if (!chapterSlug) {
    return <Navigate to={defaultLearnPath} replace />;
  }

  if (!chapter) {
    return <NotFoundPage />;
  }

  return <LearnShell chapter={chapter} />;
}
