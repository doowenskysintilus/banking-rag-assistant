import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import { conversationKeys } from "./useConversations";
import type { ChatAnswer } from "@/types/api";

export interface SendQueryInput {
  question: string;
  conversationId: string | null;
}

/** Non-streaming send — used as the fallback when the browser can't stream
 * (or streaming failed) since /chat/query returns the exact same ChatAnswer
 * shape the streamed path assembles piecemeal via SSE (see chat-stream.ts).
 */
export function useSendQuery() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ question, conversationId }: SendQueryInput) =>
      api.post<ChatAnswer>("/chat/query", { question, conversation_id: conversationId }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: conversationKeys.all });
      queryClient.invalidateQueries({ queryKey: conversationKeys.detail(data.conversation_id) });
    },
  });
}
