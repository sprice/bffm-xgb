import { Link } from "react-router-dom";
import { defaultLearnPath } from "../learn/routes";

export function HomePage() {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center gap-6 text-center px-4">
      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <Link
          to={defaultLearnPath}
          className="min-h-[52px] min-w-[170px] px-6 py-3 bg-primary text-white rounded-lg text-lg font-bold hover:bg-primary-hover active:bg-primary-hover transition-colors"
        >
          Learn
        </Link>
        <Link
          to="/assessment"
          className="min-h-[52px] min-w-[170px] px-6 py-3 border border-border rounded-lg text-lg font-bold text-text bg-surface hover:bg-primary-lighter transition-colors"
        >
          Assessment
        </Link>
      </div>
    </div>
  );
}
