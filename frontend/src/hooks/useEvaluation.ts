import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { EvalReportPublic, EvalRunSummary } from "@/types/api";

export const evaluationKeys = {
  runs: (datasetName?: string) => ["evaluation-runs", datasetName ?? "all"] as const,
  run: (runId: string) => ["evaluation-runs", "detail", runId] as const,
};

export function useEvaluationRuns(datasetName?: string, limit = 20) {
  return useQuery({
    queryKey: evaluationKeys.runs(datasetName),
    queryFn: () => api.get<EvalRunSummary[]>("/evaluation/runs", { dataset_name: datasetName, limit }),
  });
}

export function useEvaluationRun(runId: string | null) {
  return useQuery({
    queryKey: runId ? evaluationKeys.run(runId) : ["evaluation-runs", "none"],
    queryFn: () => api.get<EvalReportPublic>(`/evaluation/runs/${runId}`),
    enabled: Boolean(runId),
  });
}

export function useTriggerEvaluationRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (datasetName: string) => api.post<EvalReportPublic>("/evaluation/runs", { dataset_name: datasetName }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["evaluation-runs"] }),
  });
}
