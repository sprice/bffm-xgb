import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { PreAssessment } from "../components/PreAssessment";

export function PreAssessmentPage() {
  const navigate = useNavigate();

  useEffect(() => {
    document.title = "Before You Begin - Big Five Personality Assessment";
  }, []);

  return <div className="max-w-[540px] mx-auto"><PreAssessment onBegin={() => navigate("/assessment")} /></div>;
}
