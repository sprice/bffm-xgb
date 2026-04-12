import { useNavigate } from "react-router-dom";

export function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div className="max-w-[540px] mx-auto flex flex-col items-center justify-center gap-4 py-20 text-center">
      <h2 className="font-display text-4xl font-semibold text-text">Page Not Found</h2>
      <p className="text-base text-text-muted">
        That page doesn't exist, or the link is broken.
      </p>
      <button
        type="button"
        className="mt-2 min-h-[48px] px-9 py-3.5 rounded-lg text-lg font-bold bg-primary text-white transition-colors hover:bg-primary-hover active:bg-primary-hover"
        onClick={() => navigate("/")}
      >
        Go Home
      </button>
    </div>
  );
}
