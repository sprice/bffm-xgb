import { Link } from "react-router-dom";
import { ThemeSwitcher } from "./theme-switcher";
import { defaultLearnPath } from "../learn/routes";

const GITHUB_URL = "https://github.com/sprice/bffm-xgb";

export function AppHeader() {
  return (
    <header className="w-full border-b border-border bg-bg/90 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1280px] items-center justify-between px-4 py-3">
        <Link
          to="/"
          className="font-display text-xl sm:text-2xl font-semibold tracking-tight text-text hover:text-primary"
        >
          Big Five
        </Link>
        <nav className="flex items-center gap-2 text-sm sm:gap-4">
          <Link
            className="inline-flex items-center min-h-[44px] px-3 py-2 rounded-md text-text-muted hover:text-text transition-colors"
            to={defaultLearnPath}
          >
            Learn
          </Link>
          <Link
            className="inline-flex items-center min-h-[44px] px-3 py-2 rounded-md text-text-muted hover:text-text transition-colors"
            to="/assessment"
          >
            Take the Assessment
          </Link>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center min-h-[44px] px-3 py-2 rounded-md text-text-muted hover:text-text transition-colors"
          >
            GitHub
          </a>
          <ThemeSwitcher />
        </nav>
      </div>
    </header>
  );
}
