import { Moon, Sun } from "lucide-react";
import { getToggledTheme, useTheme } from "../theme";

export function ThemeSwitcher() {
  const { theme, toggleTheme } = useTheme();
  const nextTheme = getToggledTheme(theme);
  const label = `Theme is ${theme}. Click to switch to ${nextTheme}.`;

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md text-text-muted transition-colors hover:text-text active:opacity-75"
      aria-label={label}
      title={label}
    >
      {theme === "light" ? (
        <Moon aria-hidden="true" className="h-5 w-5" strokeWidth={1.5} />
      ) : (
        <Sun aria-hidden="true" className="h-5 w-5" strokeWidth={1.5} />
      )}
    </button>
  );
}
