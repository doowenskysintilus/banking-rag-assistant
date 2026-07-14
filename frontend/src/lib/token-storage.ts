/**
 * Token persistence + a tiny pub/sub so React components can react to
 * login/logout without prop-drilling. Tokens live in `localStorage` (not an
 * httpOnly cookie) because this is a pure SPA talking directly to the FastAPI
 * backend with no server-rendering layer to set cookies from — the
 * documented trade-off is that a successful XSS on this app could exfiltrate
 * the token. Mitigations already in place on the backend side (see
 * backend.core.security_headers) reduce that risk but don't eliminate it.
 */

const ACCESS_TOKEN_KEY = "banking_rag.access_token";
const REFRESH_TOKEN_KEY = "banking_rag.refresh_token";

type Listener = () => void;
const listeners = new Set<Listener>();

function notify() {
  for (const listener of listeners) listener();
}

export function subscribeToTokenChanges(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setTokens(accessToken: string, refreshToken: string | null): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  if (refreshToken) {
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  }
  notify();
}

export function clearTokens(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  notify();
}

export interface DecodedAccessToken {
  sub: string;
  role?: string;
  scope?: string;
  exp: number;
  iat: number;
  jti: string;
}

/** Decodes the JWT payload for UI purposes only (role/scope display, expiry
 * countdown) — this is NOT signature verification; the backend is the only
 * authority on whether a token is actually valid.
 */
export function decodeAccessToken(token: string): DecodedAccessToken | null {
  try {
    const [, payloadB64] = token.split(".");
    if (!payloadB64) return null;
    const normalized = payloadB64.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(normalized.padEnd(normalized.length + ((4 - (normalized.length % 4)) % 4), "="));
    return JSON.parse(json) as DecodedAccessToken;
  } catch {
    return null;
  }
}

export function isTokenExpired(decoded: DecodedAccessToken, skewSeconds = 10): boolean {
  return decoded.exp * 1000 < Date.now() + skewSeconds * 1000;
}

export function getScopes(): string[] {
  const token = getAccessToken();
  if (!token) return [];
  const decoded = decodeAccessToken(token);
  return decoded?.scope?.split(" ").filter(Boolean) ?? [];
}
