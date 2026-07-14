import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { AuditChainVerification, AuditLogList } from "@/types/api";

export interface AuditLogFilters {
  userId?: string;
  action?: string;
  limit?: number;
  offset?: number;
}

export function useAuditLogs(filters: AuditLogFilters = {}) {
  const { userId, action, limit = 50, offset = 0 } = filters;
  return useQuery({
    queryKey: ["audit-logs", filters],
    queryFn: () => api.get<AuditLogList>("/audit/logs", { user_id: userId, action, limit, offset }),
  });
}

export function useAuditChainVerification(enabled: boolean) {
  return useQuery({
    queryKey: ["audit-verify"],
    queryFn: () => api.get<AuditChainVerification>("/audit/verify"),
    enabled,
  });
}
