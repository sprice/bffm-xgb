import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { PreAssessment } from "../components/PreAssessment";
import { getResumeTarget } from "../hooks/use-assessment";

export function PreAssessmentPage() {
  const navigate = useNavigate();

  useEffect(() => {
    document.title = "Before You Begin - Big Five Personality Assessment";
  }, []);

  return (
    <div className="max-w-[540px] mx-auto">
      <PreAssessment
        onBegin={() => {
          const target = getResumeTarget();
          navigate(
            target === "done" ? "/assessment/done" : `/assessment/question/${target}`,
          );
        }}
      />
    </div>
  );
}
