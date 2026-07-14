import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { UserAdminView, UserList, UserRole } from "@/types/api";

export const userKeys = {
  all: ["admin-users"] as const,
  list: (filters: UserListFilters) => ["admin-users", "list", filters] as const,
};

export interface UserListFilters {
  role?: UserRole;
  isActive?: boolean;
  limit?: number;
  offset?: number;
}

export function useUsers(filters: UserListFilters = {}) {
  const { role, isActive, limit = 50, offset = 0 } = filters;
  return useQuery({
    queryKey: userKeys.list(filters),
    queryFn: () =>
      api.get<UserList>("/users", {
        role,
        is_active: isActive,
        limit,
        offset,
      }),
  });
}

export function useUpdateUserRole() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: UserRole }) =>
      api.put<UserAdminView>(`/users/${userId}/role`, { role }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: userKeys.all }),
  });
}

export function useSetUserActive() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, isActive, reason }: { userId: string; isActive: boolean; reason?: string }) =>
      api.put<UserAdminView>(`/users/${userId}/active`, { is_active: isActive, reason }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: userKeys.all }),
  });
}

export function useUnlockUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, reason }: { userId: string; reason?: string }) =>
      api.post<UserAdminView>(`/users/${userId}/unlock`, { reason }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: userKeys.all }),
  });
}
