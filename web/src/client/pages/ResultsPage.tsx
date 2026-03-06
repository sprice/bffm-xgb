import { Navigate, useLocation } from "react-router-dom";
import { encodeResults } from "../lib/results-codec";
import type { PredictionResult } from "../types";

export function ResultsPage() {
  const location = useLocation();
  const results = (location.state as { results?: PredictionResult })?.results;

  if (!results) {
    return <Navigate to="/" replace />;
  }

  let hash: string;
  try {
    hash = encodeResults(results);
  } catch {
    return <Navigate to="/" replace />;
  }
  return <Navigate to={`/results/${hash}`} replace />;
}
