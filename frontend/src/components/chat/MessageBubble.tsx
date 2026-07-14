import { Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationList } from "./CitationList";
import { GuardrailWarnings } from "./GuardrailWarnings";
import type { ChatMessageView } from "@/pages/ChatPage";

export function MessageBubble({ message }: { message: ChatMessageView }) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
        )}
      >
        {isUser ? <User className="size-4" /> : <Bot className="size-4" />}
      </div>

      <div className={cn("max-w-[75%] rounded-lg border bg-card px-4 py-3 shadow-sm", isUser && "bg-primary/5")}>
        <p className="whitespace-pre-wrap text-sm leading-relaxed">
          {message.content}
          {message.isStreaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-foreground/50 align-text-bottom" />}
        </p>

        {!isUser && !message.isStreaming && message.confidenceLevel && (
          <div className="mt-3 flex items-center gap-2">
            <ConfidenceBadge level={message.confidenceLevel} score={message.confidenceScore ?? 0} />
          </div>
        )}

        {!isUser && message.citations && message.documentsUsed && (
          <CitationList citations={message.citations} documentsUsed={message.documentsUsed} />
        )}

        {!isUser && message.warnings && <GuardrailWarnings warnings={message.warnings} />}
      </div>
    </div>
  );
}
