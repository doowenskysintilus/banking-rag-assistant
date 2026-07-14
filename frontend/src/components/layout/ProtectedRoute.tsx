import type { ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/context/AuthProvider";
import { ShieldAlert } from "lucide-react";

interface ProtectedRouteProps {
  children: ReactNode;
  /** An OAuth2 scope / RBAC permission string (see backend.core.permissions.Permission) — */
  requiredScope?: string;
}

export function ProtectedRoute({ children, requiredScope }: ProtectedRouteProps) {
  const { isAuthenticated, hasScope } = useAuth();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (requiredScope && !hasScope(requiredScope)) {
    return (
      <div className="flex h-full min-h-96 flex-col items-center justify-center gap-3 p-8 text-center">
        <ShieldAlert className="size-10 text-muted-foreground" />
        <h2 className="text-lg font-semibold">You don't have access to this page</h2>
        <p className="max-w-md text-sm text-muted-foreground">
          This section requires the <code className="rounded bg-muted px-1 py-0.5">{requiredScope}</code>{" "}
          permission. Ask an administrator if you believe this is a mistake.
        </p>
      </div>
    );
  }

  return <>{children}</>;
}
