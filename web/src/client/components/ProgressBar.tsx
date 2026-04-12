interface ProgressBarProps {
  current: number;
  total: number;
}

export function ProgressBar({ current, total }: ProgressBarProps) {
  const pct = total > 0 ? current / total : 0;
  return (
    <div className="mb-6">
      <div className="text-base text-text-muted mb-2 text-center font-medium">
        {current} of {total} answered
      </div>
      <div
        className="h-2 bg-border/40 rounded-full overflow-hidden"
        role="progressbar"
        aria-valuenow={current}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-label={`Assessment progress: ${current} of ${total} questions answered`}
      >
        <div
          className="h-full w-full origin-left bg-primary rounded-full transition-transform duration-500 ease-out"
          style={{ transform: `scaleX(${pct})` }}
        />
      </div>
    </div>
  );
}
