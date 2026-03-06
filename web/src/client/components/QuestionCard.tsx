import { useRef } from "react";
import type { Item, Likert } from "../types";
import { LIKERT_LABELS } from "../types";
import { cn } from "../lib/utils";

interface QuestionCardProps {
  item: Item;
  selectedValue: Likert | undefined;
  onAnswer: (value: Likert) => void;
  questionNumber: number;
  total: number;
}

const LIKERT_VALUES: Likert[] = [1, 2, 3, 4, 5];

export function QuestionCard({
  item,
  selectedValue,
  onAnswer,
  questionNumber,
  total,
}: QuestionCardProps) {
  const questionId = `question-${item.id}`;
  const groupRef = useRef<HTMLDivElement>(null);

  function handleKeyDown(e: React.KeyboardEvent) {
    const buttons = groupRef.current?.querySelectorAll<HTMLButtonElement>('[role="radio"]');
    if (!buttons?.length) return;

    let idx = Array.from(buttons).findIndex((b) => b === document.activeElement);
    if (idx === -1) return;

    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      idx = (idx + 1) % buttons.length;
      buttons[idx].focus();
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      idx = (idx - 1 + buttons.length) % buttons.length;
      buttons[idx].focus();
    }
  }

  const activeIndex = selectedValue ? LIKERT_VALUES.indexOf(selectedValue) : 0;

  return (
    <div className="bg-surface border border-border rounded-lg p-5 sm:p-6 shadow-sm">
      <div key={item.id} className="animate-in">
        <div className="text-sm text-text-muted mb-4 font-medium tracking-wide">
          Question {questionNumber} of {total}
        </div>

        <p id={questionId} className="font-display text-2xl font-medium mb-6 leading-snug">{item.text}</p>
      </div>

      <div
        ref={groupRef}
        className="flex flex-col gap-2"
        role="radiogroup"
        aria-labelledby={questionId}
        onKeyDown={handleKeyDown}
      >
        {LIKERT_VALUES.map((value, i) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={selectedValue === value}
            tabIndex={i === activeIndex ? 0 : -1}
            className={cn(
              "flex items-center gap-3 w-full min-h-[48px] py-3 px-4 border rounded-lg text-base transition-all",
              "border-border bg-surface text-text hover:border-primary hover:bg-primary-lighter active:scale-[0.98] active:bg-primary-light",
              selectedValue === value && "border-primary bg-primary-light font-semibold shadow-sm"
            )}
            onClick={() => onAnswer(value)}
          >
            <span
              className={cn(
                "shrink-0 w-2 h-2 rounded-full transition-all",
                selectedValue === value ? "bg-primary scale-125" : "bg-border"
              )}
              aria-hidden="true"
            />
            <span className="flex-1 text-left">{LIKERT_LABELS[value]}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
