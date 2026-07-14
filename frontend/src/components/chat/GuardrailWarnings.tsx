import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

/** Surfaces backend.rag.guardrails' rule-based anti-hallucination flags —
 * these are heuristics, not certainties, so this reads as "worth a second
 * look" rather than "this answer is wrong".
 */
export function GuardrailWarnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;

  return (
    <Alert variant="default" className="mt-3 border-warning/40 bg-warning/10">
      <AlertTriangle className="size-4 text-warning" />
      <AlertDescription className="text-foreground">
        <ul className="list-disc space-y-1 pl-4">
          {warnings.map((warning, i) => (
            <li key={i}>{warning}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}
