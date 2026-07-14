import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { toast } from "sonner";
import { FileQuestion, Trash2 } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent } from "@/components/ui/card";
import { DocumentUploadDialog } from "@/components/documents/DocumentUploadDialog";
import { DocumentStatusBadge } from "@/components/documents/DocumentStatusBadge";
import { DocumentVersionsDialog } from "@/components/documents/DocumentVersionsDialog";
import { useDeleteDocument, useDocuments } from "@/hooks/useDocuments";
import { ApiError } from "@/lib/api-client";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentsPage() {
  const { data, isLoading } = useDocuments();
  const deleteDocument = useDeleteDocument();
  const [versionsFor, setVersionsFor] = useState<string | null>(null);

  async function handleDelete(documentId: string, filename: string) {
    if (!window.confirm(`Delete "${filename}"? This removes it and its embeddings from the knowledge base.`)) {
      return;
    }
    try {
      await deleteDocument.mutateAsync(documentId);
      toast.success(`${filename} was deleted.`);
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : "Failed to delete document.");
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Knowledge base documents</h1>
          <p className="text-sm text-muted-foreground">
            Upload policies, product terms, and regulations for the assistant to answer from.
          </p>
        </div>
        <DocumentUploadDialog />
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {!isLoading && data?.items.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center text-muted-foreground">
            <FileQuestion className="size-8" />
            <p className="text-sm">No documents uploaded yet.</p>
          </CardContent>
        </Card>
      )}

      {!isLoading && data && data.items.length > 0 && (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Language</TableHead>
                <TableHead>Uploaded</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.map((document) => (
                <TableRow key={document.id}>
                  <TableCell className="max-w-56 truncate font-medium">{document.filename}</TableCell>
                  <TableCell>
                    <DocumentStatusBadge status={document.status} />
                    {document.status === "failed" && document.error_message && (
                      <p className="mt-1 max-w-56 truncate text-xs text-destructive">{document.error_message}</p>
                    )}
                  </TableCell>
                  <TableCell>
                    <button onClick={() => setVersionsFor(document.id)}>
                      <Badge variant="outline" className="cursor-pointer hover:bg-accent">
                        v{document.version}
                      </Badge>
                    </button>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">{formatBytes(document.size_bytes)}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{document.language ?? "—"}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDistanceToNow(new Date(document.created_at), { addSuffix: true })}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => handleDelete(document.id, document.filename)}
                      aria-label={`Delete ${document.filename}`}
                    >
                      <Trash2 className="text-destructive" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      <DocumentVersionsDialog documentId={versionsFor} onOpenChange={(open) => !open && setVersionsFor(null)} />
    </div>
  );
}
