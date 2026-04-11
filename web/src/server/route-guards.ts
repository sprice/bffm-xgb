export function isAnalyticsPath(pathname: string): boolean {
  return pathname === "/a" || pathname.startsWith("/a/");
}

export function isApiPath(pathname: string): boolean {
  return pathname === "/api" || pathname.startsWith("/api/");
}

export function isNavigationRequest(
  pathname: string,
  accept: string,
  cfaMode?: string,
  cfaDest?: string,
): boolean {
  if (isApiPath(pathname) || isAnalyticsPath(pathname)) return false;
  if (pathname.startsWith("/assets/") || pathname.startsWith("/@") || pathname.startsWith("/__vite")) return false;
  if (/\.[^/]+$/.test(pathname)) return false;

  if (pathname === "/") return true;
  if (cfaMode === "navigate" || cfaDest === "document") return true;
  return accept.includes("text/html");
}
