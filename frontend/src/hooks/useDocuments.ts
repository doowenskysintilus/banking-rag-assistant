import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { DocumentList, DocumentPublic } from "@/types/api";

export const documentKeys = {
  all: ["documents"] as const,
  list: (limit: number, offset: number) => ["documents", "list", limit, offset] as const,
  versions: (id: string) => ["documents", id, "versions"] as const,
};

export function useDocuments(limit = 50, offset = 0) {
  return useQuery({
    queryKey: documentKeys.list(limit, offset),
    queryFn: () => api.get<DocumentList>("/documents", { limit, offset }),
    // Documents move through pending -> processing -> indexed asynchronously
    // server-side (background ingestion) — poll while any are still in flight.
    refetchInterval: (query) => {
      const data = query.state.data as DocumentList | undefined;
      const hasInFlight = data?.items.some((d) => d.status === "pending" || d.status === "processing");
      return hasInFlight ? 3000 : false;
    },
  });
}

export function useDocumentVersions(documentId: string | null) {
  return useQuery({
    queryKey: documentId ? documentKeys.versions(documentId) : ["documents", "none", "versions"],
    queryFn: () => api.get<DocumentPublic[]>(`/documents/${documentId}/versions`),
    enabled: Boolean(documentId),
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return api.upload<DocumentPublic>("/documents/upload", formData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentKeys.all });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => api.delete(`/documents/${documentId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentKeys.all });
    },
  });
}
