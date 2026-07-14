import { format } from "date-fns";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { METRIC_LABELS, METRIC_ORDER } from "@/lib/metrics";
import type { EvalRunSummary } from "@/types/api";

const LINE_COLORS: Record<string, string> = {
  faithfulness: "var(--chart-1)",
  context_precision: "var(--chart-2)",
  context_recall: "var(--chart-3)",
  answer_relevancy: "var(--chart-4)",
  hallucination_rate: "var(--chart-5)",
};

interface MetricTrendChartProps {
  runs: EvalRunSummary[];
}

export function MetricTrendChart({ runs }: MetricTrendChartProps) {
  const chartData = [...runs]
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((run) => ({
      date: format(new Date(run.created_at), "MMM d, HH:mm"),
      ...run.aggregate_scores,
    }));

  const availableMetrics = METRIC_ORDER.filter((metric) => chartData.some((row) => metric in row));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Metric trend across runs</CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length < 2 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            Run the evaluation at least twice to see a trend.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData} margin={{ left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--popover)",
                  color: "var(--popover-foreground)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {availableMetrics.map((metric) => (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  name={METRIC_LABELS[metric] ?? metric}
                  stroke={LINE_COLORS[metric]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
