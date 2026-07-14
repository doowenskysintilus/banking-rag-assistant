import { useState, type KeyboardEvent } from "react";
import { SendHorizonal } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSend: (question: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="flex items-end gap-2 border-t bg-background p-4">
      <Textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about a policy, product term, or regulation… (Shift+Enter for a new line)"
        rows={2}
        disabled={disabled}
        className="max-h-40 min-h-16 resize-none"
      />
      <Button size="icon" onClick={submit} disabled={disabled || !value.trim()} aria-label="Send message">
        <SendHorizonal className="size-4" />
      </Button>
    </div>
  );
}
