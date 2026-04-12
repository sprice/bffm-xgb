import { useEffect, useRef } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { ProgressBar } from "../components/ProgressBar";
import { QuestionCard } from "../components/QuestionCard";
import { useAssessment } from "../hooks/use-assessment";
import type { Likert } from "../types";

export function AssessmentQuestionPage() {
  const { questionNumber: param } = useParams<{ questionNumber: string }>();
  const navigate = useNavigate();
  const { order, totalItems, answeredCount, allAnswered, responses, answer } =
    useAssessment();

  const questionNumber = Number(param);
  const index = questionNumber - 1;

  const cardRef = useRef<HTMLDivElement>(null);
  const pendingAdvance = useRef(false);
  const advanceTimerId = useRef<ReturnType<typeof setTimeout> | null>(null);

  const currentItem = order[index] as (typeof order)[number] | undefined;

  useEffect(() => {
    document.title = `Question ${questionNumber} of ${totalItems} - Big Five Personality Assessment`;
  }, [questionNumber, totalItems]);

  useEffect(() => {
    cardRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [questionNumber]);

  // Cancel pending auto-advance when question changes (e.g. manual nav)
  useEffect(() => {
    return () => {
      if (advanceTimerId.current) {
        clearTimeout(advanceTimerId.current);
        advanceTimerId.current = null;
        pendingAdvance.current = false;
      }
    };
  }, [questionNumber]);

  // Keyboard navigation: arrow keys advance between questions,
  // EXCEPT when focus is inside the Likert radiogroup — there, the
  // QuestionCard handler owns arrow keys (moving between options).
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (
        e.target instanceof Element &&
        e.target.closest('[role="radiogroup"]')
      ) {
        return;
      }
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        if (questionNumber < totalItems)
          navigate(`/assessment/question/${questionNumber + 1}`);
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        if (questionNumber > 1)
          navigate(`/assessment/question/${questionNumber - 1}`);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [questionNumber, totalItems, navigate]);

  if (
    !param ||
    Number.isNaN(questionNumber) ||
    questionNumber < 1 ||
    questionNumber > totalItems ||
    !currentItem
  ) {
    return <Navigate to="/assessment" replace />;
  }

  function handleAnswer(value: Likert) {
    if (!currentItem) return;
    const isNewAnswer = responses[currentItem.id] === undefined;
    answer(index, value);

    if (!pendingAdvance.current) {
      let nextRoute: string | null = null;
      if (questionNumber < totalItems) {
        nextRoute = `/assessment/question/${questionNumber + 1}`;
      } else if (isNewAnswer ? answeredCount + 1 >= totalItems : allAnswered) {
        nextRoute = "/assessment/done";
      }

      if (nextRoute) {
        pendingAdvance.current = true;
        const route = nextRoute;
        advanceTimerId.current = setTimeout(() => {
          pendingAdvance.current = false;
          advanceTimerId.current = null;
          navigate(route);
        }, 200);
      }
    }
  }

  return (
    <div className="max-w-[540px] mx-auto">
      <ProgressBar current={answeredCount} total={totalItems} />

      <div ref={cardRef}>
        <QuestionCard
          item={currentItem}
          selectedValue={responses[currentItem.id]}
          onAnswer={handleAnswer}
          questionNumber={questionNumber}
          total={totalItems}
        />
      </div>

      <nav
        className="flex flex-wrap justify-between items-center mt-5 gap-3"
        aria-label="Question navigation"
      >
        <button
          type="button"
          aria-label="Previous question"
          className="min-h-[44px] min-w-[44px] px-4 py-2.5 border border-border rounded-md bg-surface text-sm text-text transition-colors hover:bg-primary-lighter active:bg-primary-light disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={() => navigate(`/assessment/question/${questionNumber - 1}`)}
          disabled={questionNumber === 1}
        >
          &larr; Previous
        </button>

        <span className="flex-1" />

        <button
          type="button"
          aria-label="Next question"
          className="min-h-[44px] min-w-[44px] px-4 py-2.5 border border-border rounded-md bg-surface text-sm text-text transition-colors hover:bg-primary-lighter active:bg-primary-light disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={() => navigate(`/assessment/question/${questionNumber + 1}`)}
          disabled={questionNumber === totalItems}
        >
          Next &rarr;
        </button>
      </nav>
    </div>
  );
}
