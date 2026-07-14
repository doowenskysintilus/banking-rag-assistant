import { useState } from "react";
import { toast } from "sonner";
import { formatDistanceToNow } from "date-fns";
import { LockOpen, UserCog, UserX } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UserRoleSelect } from "@/components/users/UserRoleSelect";
import { UserStatusBadge } from "@/components/users/UserStatusBadge";
import { useSetUserActive, useUnlockUser, useUsers, type UserListFilters } from "@/hooks/useUsers";
import { useAuth } from "@/context/AuthProvider";
import { ApiError } from "@/lib/api-client";
import type { UserRole } from "@/types/api";

const ROLE_FILTER_OPTIONS: { value: UserRole | "all"; label: string }[] = [
  { value: "all", label: "All roles" },
  { value: "admin", label: "Administrator" },
  { value: "analyst", label: "Analyst" },
  { value: "customer_support", label: "Customer Support" },
];

function isLocked(lockedUntil: string | null): boolean {
  return lockedUntil !== null && new Date(lockedUntil).getTime() > Date.now();
}

export function UsersPage() {
  const { user: currentUser } = useAuth();
  const [roleFilter, setRoleFilter] = useState<UserRole | "all">("all");

  const filters: UserListFilters = roleFilter === "all" ? {} : { role: roleFilter };
  const { data, isLoading } = useUsers(filters);
  const setActive = useSetUserActive();
  const unlockUser = useUnlockUser();

  async function handleToggleActive(userId: string, isActive: boolean, name: string) {
    const action = isActive ? "deactivate" : "reactivate";
    if (!window.confirm(`Are you sure you want to ${action} ${name}?`)) return;
    try {
      await setActive.mutateAsync({ userId, isActive: !isActive });
      toast.success(`${name} was ${action}d.`);
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : `Failed to ${action} user.`);
    }
  }

  async function handleUnlock(userId: string, name: string) {
    try {
      await unlockUser.mutateAsync({ userId, reason: "Unlocked via admin console" });
      toast.success(`${name}'s account was unlocked.`);
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : "Failed to unlock user.");
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">User management</h1>
          <p className="text-sm text-muted-foreground">
            Manage roles, account status, and brute-force lockouts. Every change here is written to the audit
            trail.
          </p>
        </div>
        <Select value={roleFilter} onValueChange={(v) => setRoleFilter(v as UserRole | "all")}>
          <SelectTrigger className="w-52">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ROLE_FILTER_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      )}

      {!isLoading && data && (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>User</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Joined</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.map((user) => {
                const isSelf = user.id === currentUser?.id;
                return (
                  <TableRow key={user.id}>
                    <TableCell>
                      <p className="font-medium">{user.full_name}</p>
                      <p className="text-xs text-muted-foreground">{user.email}</p>
                    </TableCell>
                    <TableCell>
                      <UserRoleSelect userId={user.id} role={user.role} disabled={isSelf} />
                    </TableCell>
                    <TableCell>
                      <UserStatusBadge user={user} />
                      {user.failed_login_attempts > 0 && !isLocked(user.locked_until) && (
                        <p className="mt-1 text-xs text-muted-foreground">
                          {user.failed_login_attempts} failed login attempt(s)
                        </p>
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDistanceToNow(new Date(user.created_at), { addSuffix: true })}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        {isLocked(user.locked_until) && (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => handleUnlock(user.id, user.full_name)}
                            aria-label={`Unlock ${user.full_name}`}
                          >
                            <LockOpen />
                          </Button>
                        )}
                        {!isSelf && (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => handleToggleActive(user.id, user.is_active, user.full_name)}
                            aria-label={user.is_active ? `Deactivate ${user.full_name}` : `Reactivate ${user.full_name}`}
                          >
                            {user.is_active ? <UserX className="text-destructive" /> : <UserCog />}
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </Card>
      )}
    </div>
  );
}
