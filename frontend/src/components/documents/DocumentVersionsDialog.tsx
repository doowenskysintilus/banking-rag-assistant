import { format } from "date-fns";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DocumentStatusBadge } from "./DocumentStatusBadge";
import { useDocumentVersions } from "@/hooks/useDocuments";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

interface DocumentVersionsDialogProps {
  documentId: string | null;
  onOpenChange: (open: boolean) => void;
}

export function DocumentVersionsDialog({ documentId, onOpenChange }: DocumentVersionsDialogProps) {
  const { data: versions, isLoading } = useDocumentVersions(documentId);

  return (
    <Dialog open={documentId !== null} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Version history</DialogTitle>
        </DialogHeader>
        <div className="space-y-2">
          {isLoading && <Skeleton className="h-24 w-full" />}
          {versions?.map((version) => (
            <div key={version.id} className="flex items-center justify-between rounded-md border p-3 text-sm">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium">v{version.version}</span>
                  {version.is_current && <Badge variant="secondary">Current</Badge>}
                </div>
                <p className="text-xs text-muted-foreground">
                  {format(new Date(version.created_at), "PPp")}
                </p>
              </div>
              <DocumentStatusBadge status={version.status} />
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
