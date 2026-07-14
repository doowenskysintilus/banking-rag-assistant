import { NavLink, Outlet } from "react-router-dom";
import { BarChart3, FileText, History, LogOut, MessageSquare, Moon, Shield, Sun, Users } from "lucide-react";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";
import { useAuth } from "@/context/AuthProvider";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";

interface NavItem {
  to: string;
  label: string;
  icon: typeof MessageSquare;
  requiredScope?: string;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Chat", icon: MessageSquare, end: true },
  { to: "/conversations", label: "History", icon: History },
  { to: "/documents", label: "Documents", icon: FileText },
  { to: "/dashboard", label: "Analytics", icon: BarChart3, requiredScope: "evaluation:read" },
  { to: "/users", label: "Users", icon: Users, requiredScope: "user:read" },
];

const ROLE_LABELS: Record<string, string> = {
  admin: "Administrator",
  analyst: "Analyst",
  customer_support: "Customer Support",
};

function initialsFor(name: string): string {
  const parts = name.trim().split(/\s+/);
  return parts
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase())
    .join("");
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      <Sun className="size-4 scale-100 rotate-0 dark:scale-0 dark:-rotate-90" />
      <Moon className="absolute size-4 scale-0 rotate-90 dark:scale-100 dark:rotate-0" />
    </Button>
  );
}

export function AppShell() {
  const { user, role, hasScope, logout } = useAuth();

  return (
    <div className="flex h-screen w-full overflow-hidden bg-muted/30">
      <aside className="flex w-60 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground">
        <div className="flex items-center gap-2 border-b px-4 py-4">
          <Shield className="size-6 text-sidebar-primary" />
          <div>
            <p className="text-sm font-semibold leading-tight">Banking RAG</p>
            <p className="text-xs text-muted-foreground leading-tight">Assistant</p>
          </div>
        </div>

        <nav className="flex-1 space-y-1 px-2 py-4">
          {NAV_ITEMS.filter((item) => !item.requiredScope || hasScope(item.requiredScope)).map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                )
              }
            >
              <item.icon className="size-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t p-3">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex w-full items-center gap-2 rounded-md p-2 text-left hover:bg-sidebar-accent/60">
                <Avatar className="size-8">
                  <AvatarFallback className="text-xs">
                    {user ? initialsFor(user.full_name) : "…"}
                  </AvatarFallback>
                </Avatar>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{user?.full_name ?? "Loading…"}</p>
                  <p className="truncate text-xs text-muted-foreground">{user?.email}</p>
                </div>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuLabel className="flex items-center justify-between font-normal">
                <span className="text-xs text-muted-foreground">Signed in as</span>
                <Badge variant="secondary" className="text-[10px]">
                  {role ? (ROLE_LABELS[role] ?? role) : "—"}
                </Badge>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => void logout()} variant="destructive">
                <LogOut />
                Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-end border-b bg-background px-4">
          <ThemeToggle />
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
