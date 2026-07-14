import { Link } from "react-router-dom";
import { formatDistanceToNow } from "date-fns";
import { MessageSquarePlus, MessagesSquare } from "lucide-react";
import { useConversations } from "@/hooks/useConversations";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export function ConversationsPage() {
  const { data: conversations, isLoading } = useConversations();

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Conversation history</h1>
          <p className="text-sm text-muted-foreground">Every question you've asked, and the assistant's answers.</p>
        </div>
        <Button asChild>
          <Link to="/">
            <MessageSquarePlus />
            New chat
          </Link>
        </Button>
      </div>

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      )}

      {!isLoading && conversations?.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center text-muted-foreground">
            <MessagesSquare className="size-8" />
            <p className="text-sm">No conversations yet — start one from the Chat page.</p>
          </CardContent>
        </Card>
      )}

      <div className="space-y-2">
        {conversations?.map((conversation) => (
          <Link key={conversation.id} to={`/conversations/${conversation.id}`}>
            <Card className="transition-colors hover:bg-accent/50">
              <CardContent className="flex items-center justify-between py-4">
                <div className="min-w-0">
                  <p className="truncate font-medium">{conversation.title || "Untitled conversation"}</p>
                  <p className="text-xs text-muted-foreground">
                    Updated {formatDistanceToNow(new Date(conversation.updated_at), { addSuffix: true })}
                  </p>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
