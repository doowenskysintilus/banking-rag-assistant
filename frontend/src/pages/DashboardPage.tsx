import { useState } from "react";
import { toast } from "sonner";
import { Loader2, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { MetricTrendChart } from "@/components/dashboard/MetricTrendChart";
import { AuditPanel } from "@/components/dashboard/AuditPanel";
import { useEvaluationRuns, useTriggerEvaluationRun } from "@/hooks/useEvaluation";
import { useAuth } from "@/context/AuthProvider";
import { METRIC_ORDER } from "@/lib/metrics";
import { ApiError } from "@/lib/api-client";

const DATASETS = [
  { value: "golden-fixture-set", label: "Golden fixture set" },
];

export function DashboardPage() {
  const { hasScope } = useAuth();
  const [datasetName, setDatasetName] = useState(DATASETS[0].value);
  const { data: runs, isLoading } = useEvaluationRuns(datasetName);
  const triggerRun = useTriggerEvaluationRun();

  const latestRun = runs?.[0];

  async function handleTriggerRun() {
    try {
      await triggerRun.mutateAsync(datasetName);
      toast.success("Evaluation run completed.");
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : "Failed to run evaluation.");
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">RAG quality analytics</h1>
          <p className="text-sm text-muted-foreground">
            Faithfulness, retrieval precision/recall, answer relevancy, and hallucination rate — see
            backend.evaluation for how these are computed.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={datasetName} onValueChange={setDatasetName}>
            <SelectTrigger className="w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DATASETS.map((d) => (
                <SelectItem key={d.value} value={d.value}>
                  {d.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {hasScope("evaluation:run") && (
            <Button onClick={handleTriggerRun} disabled={triggerRun.isPending}>
              {triggerRun.isPending ? <Loader2 className="animate-spin" /> : <PlayCircle />}
              Run evaluation
            </Button>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      )}

      {!isLoading && !latestRun && (
        <p className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
          No evaluation runs yet for this dataset — click "Run evaluation" to generate the first one.
        </p>
      )}

      {latestRun && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            {METRIC_ORDER.map((metric) => (
              <MetricCard key={metric} metric={metric} score={latestRun.aggregate_scores[metric]} />
            ))}
          </div>

          {latestRun.warnings.length > 0 && (
            <ul className="rounded-md border border-warning/40 bg-warning/10 p-3 text-sm">
              {latestRun.warnings.map((warning, i) => (
                <li key={i}>{warning}</li>
              ))}
            </ul>
          )}

          <MetricTrendChart runs={runs ?? []} />
        </>
      )}

      {hasScope("audit:read") && <AuditPanel />}
    </div>
  );
}
