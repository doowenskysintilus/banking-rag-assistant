/** Mirrors backend.evaluation.report's band thresholds — a "green" score
 * should mean the same thing here as it does in the backend-rendered
 * dashboard/report.
 */
export type MetricBand = "good" | "warning" | "critical";

const HIGH_THRESHOLD = 0.75;
const MEDIUM_THRESHOLD = 0.4;

export function bandFor(score: number): MetricBand {
  if (score >= HIGH_THRESHOLD) return "good";
  if (score >= MEDIUM_THRESHOLD) return "warning";
  return "critical";
}

export const METRIC_LABELS: Record<string, string> = {
  faithfulness: "Faithfulness",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
  answer_relevancy: "Answer Relevancy",
  hallucination_rate: "Non-Hallucination Rate",
};

export const METRIC_ORDER = [
  "faithfulness",
  "context_precision",
  "context_recall",
  "answer_relevancy",
  "hallucination_rate",
];

export const BAND_TEXT_CLASS: Record<MetricBand, string> = {
  good: "text-success",
  warning: "text-warning",
  critical: "text-destructive",
};

export const BAND_DOT_CLASS: Record<MetricBand, string> = {
  good: "bg-success",
  warning: "bg-warning",
  critical: "bg-destructive",
};
