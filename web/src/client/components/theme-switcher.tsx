import { getToggledTheme, useTheme } from "../theme";

function SunIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5 fill-none stroke-current">
      <circle cx="12" cy="12" r="4" strokeWidth="1.8" />
      <path d="M12 2.75V5.25M12 18.75V21.25M21.25 12H18.75M5.25 12H2.75M18.54 5.46L16.77 7.23M7.23 16.77L5.46 18.54M18.54 18.54L16.77 16.77M7.23 7.23L5.46 5.46" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5 fill-none stroke-current">
      <path d="M14.5 3.5a7.5 7.5 0 1 0 6 12.03A8.75 8.75 0 1 1 14.5 3.5Z" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ThemeSwitcher() {
  const { preference, resolvedTheme, toggleTheme } = useTheme();
  const nextTheme = getToggledTheme(resolvedTheme);
  const isUsingSystemTheme = preference === "system";

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex min-h-[36px] min-w-[36px] items-center justify-center text-text-muted transition-colors hover:text-text active:opacity-75"
      aria-label={`Theme is ${resolvedTheme}${isUsingSystemTheme ? " (system)" : ""}. Click to switch to ${nextTheme}.`}
      title={`Theme is ${resolvedTheme}${isUsingSystemTheme ? " (system)" : ""}. Click to switch to ${nextTheme}.`}
    >
      {resolvedTheme === "light" ? <MoonIcon /> : <SunIcon />}
    </button>
  );
}
