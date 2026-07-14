import { FileText } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Citation, DocumentUsed } from "@/types/api";

interface CitationListProps {
  citations: Citation[];
  documentsUsed: DocumentUsed[];
}

export function CitationList({ citations, documentsUsed }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-3 space-y-2 border-t pt-3">
      <p className="text-xs font-medium text-muted-foreground">
        Sources ({documentsUsed.length} document{documentsUsed.length === 1 ? "" : "s"})
      </p>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation) => (
          <Tooltip key={citation.chunk_id}>
            <TooltipTrigger asChild>
              <button className="flex max-w-64 items-center gap-1.5 rounded-md border bg-muted/50 px-2 py-1 text-left text-xs hover:bg-muted">
                <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate">
                  [{citation.index}] {citation.document_filename}
                </span>
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-sm">
              <p className="font-medium">
                {citation.document_filename}
                {citation.header_path ? ` — ${citation.header_path}` : ""}
              </p>
              <p className="mt-1 text-muted-foreground">{citation.excerpt}</p>
              <p className="mt-1 text-[10px] text-muted-foreground">
                Relevance score: {(citation.score * 100).toFixed(0)}%
              </p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </div>
  );
}
