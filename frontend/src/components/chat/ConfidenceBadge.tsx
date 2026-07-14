import { cn } from "@/lib/utils";
import type { ConfidenceLevel } from "@/types/api";

const STYLES: Record<ConfidenceLevel, string> = {
  high: "bg-success/15 text-success border-success/30",
  medium: "bg-warning/15 text-warning border-warning/30",
  low: "bg-destructive/15 text-destructive border-destructive/30",
};

const LABELS: Record<ConfidenceLevel, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
};

interface ConfidenceBadgeProps {
  level: ConfidenceLevel;
  score: number;
}

export function ConfidenceBadge({ level, score }: ConfidenceBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium",
        STYLES[level],
      )}
      title={`Retrieval-grounded confidence score: ${(score * 100).toFixed(0)}%`}
    >
      {LABELS[level]} · {(score * 100).toFixed(0)}%
    </span>
  );
}
