import { Navigate } from "react-router-dom";
import { getResumeTarget } from "../hooks/use-assessment";

export function AssessmentPage() {
  const target = getResumeTarget();
  const path =
    target === "done" ? "/assessment/done" : `/assessment/question/${target}`;
  return <Navigate to={path} replace />;
}
