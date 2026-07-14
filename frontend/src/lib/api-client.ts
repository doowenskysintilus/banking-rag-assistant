import type { ApiErrorPayload, TokenPair } from "@/types/api";
import { clearTokens, getAccessToken, getRefreshToken, setTokens } from "./token-storage";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly details?: Record<string, unknown>;

  constructor(status: number, payload: ApiErrorPayload | null, fallbackMessage: string) {
    super(payload?.error.message ?? fallbackMessage);
    this.name = "ApiError";
    this.status = status;
    this.code = payload?.error.code ?? "unknown_error";
    this.requestId = payload?.error.request_id ?? null;
    this.details = payload?.error.details;
  }
}

// Only one refresh should ever be in flight at a time — concurrent 401s from
// several in-flight requests all await the same promise instead of each
// racing to rotate the refresh token (the backend only accepts a refresh
// token once; a second concurrent use would look like reuse-detection theft).
let refreshInFlight: Promise<boolean> | null = null;

async function refreshTokens(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const response = await fetch(`${API_BASE}${API_PREFIX}/auth/refresh`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!response.ok) return false;
        const tokens = (await response.json()) as TokenPair;
        setTokens(tokens.access_token, tokens.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        refreshInFlight = null;
      }
    })();
  }
  return refreshInFlight;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  formData?: FormData;
  query?: Record<string, string | number | boolean | undefined>;
  /** Internal — prevents infinite refresh loops. */
  _isRetry?: boolean;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const url = new URL(`${API_BASE}${API_PREFIX}${path}`, window.location.origin);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, formData, query, _isRetry = false } = options;

  const headers: Record<string, string> = {};
  const accessToken = getAccessToken();
  if (accessToken) headers["Authorization"] = `Bearer ${accessToken}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(buildUrl(path, query), {
    method,
    headers,
    body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
  });

  if (response.status === 401 && !_isRetry) {
    const refreshed = await refreshTokens();
    if (refreshed) {
      return apiRequest<T>(path, { ...options, _isRetry: true });
    }
    clearTokens();
    throw new ApiError(401, null, "Session expired — please sign in again.");
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json") ? await response.json() : null;

  if (!response.ok) {
    throw new ApiError(response.status, payload as ApiErrorPayload, `Request failed with status ${response.status}`);
  }

  return payload as T;
}

export const api = {
  get: <T>(path: string, query?: RequestOptions["query"]) => apiRequest<T>(path, { method: "GET", query }),
  post: <T>(path: string, body?: unknown) => apiRequest<T>(path, { method: "POST", body }),
  put: <T>(path: string, body?: unknown) => apiRequest<T>(path, { method: "PUT", body }),
  delete: <T>(path: string) => apiRequest<T>(path, { method: "DELETE" }),
  upload: <T>(path: string, formData: FormData) => apiRequest<T>(path, { method: "POST", formData }),
};

export { API_BASE, API_PREFIX };
