import { Suspense, lazy } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { Loader2 } from "lucide-react";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider, useAuth } from "@/context/AuthProvider";
import { AppShell } from "@/components/layout/AppShell";
import { ProtectedRoute } from "@/components/layout/ProtectedRoute";
import { LoginPage } from "@/pages/LoginPage";
import { ChatPage } from "@/pages/ChatPage";
import { ConversationsPage } from "@/pages/ConversationsPage";
import { DocumentsPage } from "@/pages/DocumentsPage";

// Recharts (dashboard) and the admin user table pull in enough weight that
// splitting them out of the main bundle is worth the extra chunk — most
// sessions only ever touch Chat/Documents/History.
const DashboardPage = lazy(() => import("@/pages/DashboardPage").then((m) => ({ default: m.DashboardPage })));
const UsersPage = lazy(() => import("@/pages/UsersPage").then((m) => ({ default: m.UsersPage })));

function RouteFallback() {
  return (
    <div className="flex h-full items-center justify-center p-12">
      <Loader2 className="size-6 animate-spin text-muted-foreground" />
    </div>
  );
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

function LoginRoute() {
  const { isAuthenticated } = useAuth();
  if (isAuthenticated) return <Navigate to="/" replace />;
  return <LoginPage />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginRoute />} />
      <Route
        element={
          <ProtectedRoute>
            <AppShell />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<ChatPage />} />
        <Route path="/conversations" element={<ConversationsPage />} />
        <Route path="/conversations/:conversationId" element={<ChatPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute requiredScope="evaluation:read">
              <Suspense fallback={<RouteFallback />}>
                <DashboardPage />
              </Suspense>
            </ProtectedRoute>
          }
        />
        <Route
          path="/users"
          element={
            <ProtectedRoute requiredScope="user:read">
              <Suspense fallback={<RouteFallback />}>
                <UsersPage />
              </Suspense>
            </ProtectedRoute>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <TooltipProvider delayDuration={200}>
            <Router>
              <AppRoutes />
            </Router>
            <Toaster position="top-right" richColors closeButton />
          </TooltipProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;
