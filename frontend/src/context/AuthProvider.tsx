import { createContext, useCallback, useContext, useMemo, useSyncExternalStore, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import {
  clearTokens,
  decodeAccessToken,
  getAccessToken,
  setTokens,
  subscribeToTokenChanges,
} from "@/lib/token-storage";
import type { TokenPair, UserPublic } from "@/types/api";

interface AuthContextValue {
  isAuthenticated: boolean;
  isLoadingUser: boolean;
  user: UserPublic | undefined;
  scopes: string[];
  role: string | undefined;
  hasScope: (scope: string) => boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function subscribe(onChange: () => void) {
  return subscribeToTokenChanges(onChange);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  // Re-renders this subtree whenever tokens are set/cleared (login/logout,
  // or a silent refresh failure in the API client) without prop drilling.
  const accessToken = useSyncExternalStore(subscribe, getAccessToken, getAccessToken);

  const decoded = useMemo(() => (accessToken ? decodeAccessToken(accessToken) : null), [accessToken]);
  const scopes = useMemo(() => decoded?.scope?.split(" ").filter(Boolean) ?? [], [decoded]);
  const isAuthenticated = Boolean(accessToken && decoded);

  const { data: user, isLoading: isLoadingUser } = useQuery({
    queryKey: ["auth", "me", accessToken],
    queryFn: () => api.get<UserPublic>("/auth/me"),
    enabled: isAuthenticated,
    retry: false,
    staleTime: 60_000,
  });

  const login = useCallback(async (email: string, password: string) => {
    const tokens = await api.post<TokenPair>("/auth/login", { email, password });
    setTokens(tokens.access_token, tokens.refresh_token);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.post("/auth/logout", {});
    } catch {
      // Best-effort server-side revocation — proceed to clear local state
      // regardless, since staying "logged in" client-side after a failed
      // revocation call would be worse than a token that outlives its logout.
    } finally {
      clearTokens();
      queryClient.clear();
    }
  }, [queryClient]);

  const hasScope = useCallback((scope: string) => scopes.includes(scope), [scopes]);

  const value: AuthContextValue = {
    isAuthenticated,
    isLoadingUser,
    user,
    scopes,
    role: decoded?.role,
    hasScope,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
