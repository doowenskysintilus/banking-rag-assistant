import { CheckCircle2, Clock, Loader2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { DocumentStatus } from "@/types/api";

const CONFIG: Record<DocumentStatus, { label: string; icon: typeof Clock; className: string }> = {
  pending: { label: "Pending", icon: Clock, className: "bg-muted text-muted-foreground" },
  processing: { label: "Processing", icon: Loader2, className: "bg-warning/15 text-warning" },
  indexed: { label: "Indexed", icon: CheckCircle2, className: "bg-success/15 text-success" },
  failed: { label: "Failed", icon: XCircle, className: "bg-destructive/15 text-destructive" },
};

export function DocumentStatusBadge({ status }: { status: DocumentStatus }) {
  const { label, icon: Icon, className } = CONFIG[status];
  return (
    <Badge variant="outline" className={className}>
      <Icon className={status === "processing" ? "size-3 animate-spin" : "size-3"} />
      {label}
    </Badge>
  );
}
