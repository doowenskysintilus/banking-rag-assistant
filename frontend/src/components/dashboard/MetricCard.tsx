import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { bandFor, BAND_DOT_CLASS, BAND_TEXT_CLASS, METRIC_LABELS } from "@/lib/metrics";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  metric: string;
  score: number | undefined;
}

export function MetricCard({ metric, score }: MetricCardProps) {
  const label = METRIC_LABELS[metric] ?? metric;

  if (score === undefined) {
    return (
      <Card className="opacity-60">
        <CardHeader className="pb-1">
          <p className="text-xs text-muted-foreground">{label}</p>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold text-muted-foreground">n/a</p>
          <p className="text-xs text-muted-foreground">skipped — no provider available</p>
        </CardContent>
      </Card>
    );
  }

  const band = bandFor(score);
  return (
    <Card>
      <CardHeader className="pb-1">
        <p className="text-xs text-muted-foreground">{label}</p>
      </CardHeader>
      <CardContent>
        <p className={cn("text-2xl font-semibold", BAND_TEXT_CLASS[band])}>{score.toFixed(2)}</p>
        <div className="mt-1 flex items-center gap-1.5">
          <span className={cn("size-1.5 rounded-full", BAND_DOT_CLASS[band])} />
          <p className={cn("text-xs uppercase tracking-wide", BAND_TEXT_CLASS[band])}>{band}</p>
        </div>
      </CardContent>
    </Card>
  );
}
