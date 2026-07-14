import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { ConversationDetail, ConversationPublic } from "@/types/api";

export const conversationKeys = {
  all: ["conversations"] as const,
  detail: (id: string) => ["conversations", id] as const,
};

export function useConversations() {
  return useQuery({
    queryKey: conversationKeys.all,
    queryFn: () => api.get<ConversationPublic[]>("/chat/conversations"),
  });
}

export function useConversationDetail(conversationId: string | null) {
  return useQuery({
    queryKey: conversationId ? conversationKeys.detail(conversationId) : ["conversations", "none"],
    queryFn: () => api.get<ConversationDetail>(`/chat/conversations/${conversationId}`),
    enabled: Boolean(conversationId),
  });
}
