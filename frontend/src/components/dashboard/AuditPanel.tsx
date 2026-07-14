import { formatDistanceToNow } from "date-fns";
import { ShieldCheck, ShieldX } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuditChainVerification, useAuditLogs } from "@/hooks/useAudit";

export function AuditPanel() {
  const { data: logs, isLoading: isLoadingLogs } = useAuditLogs({ limit: 10 });
  const { data: verification, isLoading: isLoadingVerification } = useAuditChainVerification(true);

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="text-base">Audit trail</CardTitle>
        {!isLoadingVerification && verification && (
          <Badge
            variant="outline"
            className={verification.intact ? "bg-success/15 text-success" : "bg-destructive/15 text-destructive"}
          >
            {verification.intact ? <ShieldCheck className="size-3" /> : <ShieldX className="size-3" />}
            {verification.intact ? "Chain intact" : "Tampering detected"}
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-2">
        {isLoadingLogs && <Skeleton className="h-32 w-full" />}
        {!isLoadingLogs && logs?.items.length === 0 && (
          <p className="py-4 text-center text-sm text-muted-foreground">No audit events yet.</p>
        )}
        {logs?.items.map((entry) => (
          <div key={entry.id} className="flex items-center justify-between border-b py-2 text-sm last:border-0">
            <div className="min-w-0">
              <p className="truncate font-mono text-xs">{entry.action}</p>
              {entry.detail && <p className="truncate text-xs text-muted-foreground">{entry.detail}</p>}
            </div>
            <p className="shrink-0 text-xs text-muted-foreground">
              {formatDistanceToNow(new Date(entry.created_at), { addSuffix: true })}
            </p>
          </div>
        ))}
        {verification && !verification.intact && (
          <ul className="mt-2 space-y-1 rounded-md bg-destructive/10 p-2 text-xs text-destructive">
            {verification.problems.slice(0, 5).map((problem, i) => (
              <li key={i}>{problem}</li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
