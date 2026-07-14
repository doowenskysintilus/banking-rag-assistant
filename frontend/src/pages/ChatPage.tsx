import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { streamChatQuery } from "@/lib/chat-stream";
import { useConversationDetail, conversationKeys } from "@/hooks/useConversations";
import { useSendQuery } from "@/hooks/useChat";
import { ApiError } from "@/lib/api-client";
import { Bot } from "lucide-react";
import type { ChatAnswer, Citation, ConfidenceLevel, DocumentUsed, MessageRole } from "@/types/api";

export interface ChatMessageView {
  id: string;
  role: MessageRole;
  content: string;
  isStreaming?: boolean;
  citations?: Citation[];
  documentsUsed?: DocumentUsed[];
  confidenceScore?: number;
  confidenceLevel?: ConfidenceLevel;
  warnings?: string[];
}

function applyAnswerToMessage(message: ChatMessageView, answer: ChatAnswer): ChatMessageView {
  return {
    ...message,
    content: answer.answer,
    isStreaming: false,
    citations: answer.citations,
    documentsUsed: answer.documents_used,
    confidenceScore: answer.confidence_score,
    confidenceLevel: answer.confidence_level,
    warnings: answer.warnings,
  };
}

export function ChatPage() {
  const { conversationId: routeConversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [conversationId, setConversationId] = useState<string | null>(routeConversationId ?? null);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isSending, setIsSending] = useState(false);
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  const { data: conversationDetail } = useConversationDetail(routeConversationId ?? null);
  const sendQuery = useSendQuery();

  // Loading an existing conversation from history replaces local state;
  // starting a brand new chat (no route param) clears it.
  useEffect(() => {
    setConversationId(routeConversationId ?? null);
    if (routeConversationId && conversationDetail) {
      setMessages(
        conversationDetail.messages.map((m) => ({ id: m.id, role: m.role, content: m.content })),
      );
    } else if (!routeConversationId) {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeConversationId, conversationDetail?.id]);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(question: string) {
    const userMessage: ChatMessageView = { id: crypto.randomUUID(), role: "user", content: question };
    const assistantId = crypto.randomUUID();
    setMessages((prev) => [...prev, userMessage, { id: assistantId, role: "assistant", content: "", isStreaming: true }]);
    setIsSending(true);

    let sawStreamEvent = false;
    try {
      for await (const event of streamChatQuery(question, conversationId)) {
        sawStreamEvent = true;
        if (event.type === "conversation") {
          if (!conversationId) {
            setConversationId(event.conversationId);
            navigate(`/conversations/${event.conversationId}`, { replace: true });
          }
        } else if (event.type === "delta") {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + event.text } : m)),
          );
        } else if (event.type === "done") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    isStreaming: false,
                    citations: event.payload.citations,
                    documentsUsed: event.payload.documents_used,
                    confidenceScore: event.payload.confidence_score,
                    confidenceLevel: event.payload.confidence_level,
                    warnings: event.payload.warnings,
                  }
                : m,
            ),
          );
          queryClient.invalidateQueries({ queryKey: conversationKeys.all });
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
    } catch {
      // Streaming can fail for reasons unrelated to the question itself (a
      // corporate proxy buffering/blocking SSE, a dropped connection) — fall
      // back to the plain request/response endpoint once before giving up,
      // rather than leaving the user with a stuck placeholder.
      try {
        const answer = await sendQuery.mutateAsync({ question, conversationId });
        if (!conversationId) {
          setConversationId(answer.conversation_id);
          navigate(`/conversations/${answer.conversation_id}`, { replace: true });
        }
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? applyAnswerToMessage(m, answer) : m)));
      } catch (fallbackError) {
        const message =
          fallbackError instanceof ApiError
            ? fallbackError.message
            : sawStreamEvent
              ? "The response was interrupted."
              : "Could not reach the assistant.";
        toast.error(message);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      }
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl space-y-6 p-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center gap-3 py-24 text-center text-muted-foreground">
              <Bot className="size-10" />
              <p className="text-sm">
                Ask a question about internal policies, product terms, or regulations. Every answer is grounded
                in your organization's documents, with sources cited.
              </p>
            </div>
          )}
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          <div ref={scrollAnchorRef} />
        </div>
      </ScrollArea>
      <div className="mx-auto w-full max-w-3xl">
        <ChatInput onSend={handleSend} disabled={isSending} />
      </div>
    </div>
  );
}
