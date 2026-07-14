import { API_BASE, API_PREFIX } from "./api-client";
import { getAccessToken } from "./token-storage";
import type { ChatStreamDonePayload } from "@/types/api";

export type ChatStreamEvent =
  | { type: "conversation"; conversationId: string }
  | { type: "delta"; text: string }
  | { type: "done"; payload: ChatStreamDonePayload }
  | { type: "error"; code: string; message: string };

/**
 * POSTs to /chat/stream and yields parsed SSE events as they arrive.
 *
 * The browser's built-in `EventSource` can't be used here — it only supports
 * GET requests with no custom headers, and this endpoint needs a POST body
 * (the question) plus an `Authorization: Bearer` header — so this parses the
 * `text/event-stream` framing (blank-line-delimited `event:`/`data:` pairs)
 * by hand over a regular streamed `fetch` response.
 */
export async function* streamChatQuery(
  question: string,
  conversationId: string | null,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const accessToken = getAccessToken();
  const response = await fetch(`${API_BASE}${API_PREFIX}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: JSON.stringify({ question, conversation_id: conversationId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat stream request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const parsed = parseSseFrame(frame);
        if (parsed) yield parsed;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseSseFrame(frame: string): ChatStreamEvent | null {
  let eventName = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;

  try {
    const json = JSON.parse(data);
    switch (eventName) {
      case "conversation":
        return { type: "conversation", conversationId: json.conversation_id };
      case "delta":
        return { type: "delta", text: json.text };
      case "done":
        return { type: "done", payload: json as ChatStreamDonePayload };
      case "error":
        return { type: "error", code: json.code, message: json.message };
      default:
        return null;
    }
  } catch {
    return null;
  }
}
