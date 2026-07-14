import { Lock, ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { UserAdminView } from "@/types/api";

function isLocked(user: UserAdminView): boolean {
  return user.locked_until !== null && new Date(user.locked_until).getTime() > Date.now();
}

export function UserStatusBadge({ user }: { user: UserAdminView }) {
  if (isLocked(user)) {
    return (
      <Badge variant="outline" className="bg-destructive/15 text-destructive">
        <Lock className="size-3" />
        Locked
      </Badge>
    );
  }
  if (!user.is_active) {
    return (
      <Badge variant="outline" className="bg-muted text-muted-foreground">
        <ShieldAlert className="size-3" />
        Deactivated
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="bg-success/15 text-success">
      <ShieldCheck className="size-3" />
      Active
    </Badge>
  );
}
