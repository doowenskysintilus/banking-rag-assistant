import { toast } from "sonner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useUpdateUserRole } from "@/hooks/useUsers";
import { ApiError } from "@/lib/api-client";
import type { UserRole } from "@/types/api";

const ROLE_OPTIONS: { value: UserRole; label: string }[] = [
  { value: "admin", label: "Administrator" },
  { value: "analyst", label: "Analyst" },
  { value: "customer_support", label: "Customer Support" },
];

interface UserRoleSelectProps {
  userId: string;
  role: UserRole;
  disabled?: boolean;
}

export function UserRoleSelect({ userId, role, disabled }: UserRoleSelectProps) {
  const updateRole = useUpdateUserRole();

  async function handleChange(newRole: string) {
    try {
      await updateRole.mutateAsync({ userId, role: newRole as UserRole });
      toast.success("Role updated.");
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : "Failed to update role.");
    }
  }

  return (
    <Select value={role} onValueChange={handleChange} disabled={disabled || updateRole.isPending}>
      <SelectTrigger size="sm" className="w-44">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {ROLE_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
