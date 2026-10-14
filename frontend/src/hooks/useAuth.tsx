import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { ApiError, api, getToken, setToken } from "../api/client";
import type { AuthResponse, User } from "../lib/types";

interface AuthValue {
  user: User | null;
  loading: boolean;
  /** The server did not answer while restoring the session; the sign-in is kept and can be retried. */
  unreachable: boolean;
  retry: () => void;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(getToken() !== null);
  const [unreachable, setUnreachable] = useState(false);

  const restore = useCallback(() => {
    if (!getToken()) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setUnreachable(false);
    api<User>("/auth/me")
      .then(setUser)
      .catch((e: unknown) => {
        // Only a 401 means the sign-in is no longer valid (the client already dropped the token).
        // A network error or a 5xx while the server restarts must not log anybody out.
        if (!(e instanceof ApiError && e.status === 401)) setUnreachable(true);
      })
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => restore(), [restore]);

  useEffect(() => {
    const onLogout = () => {
      queryClient.clear(); // nothing of the previous session may be shown to the next one
      setUser(null);
    };
    window.addEventListener("auth:logout", onLogout);
    return () => window.removeEventListener("auth:logout", onLogout);
  }, [queryClient]);

  const authenticate = useCallback(
    async (path: string, email: string, password: string) => {
      const result = await api<AuthResponse>(path, { method: "POST", body: JSON.stringify({ email, password }) });
      queryClient.clear();
      setToken(result.access_token);
      setUnreachable(false);
      setUser(result.user);
    },
    [queryClient],
  );

  const value = useMemo<AuthValue>(
    () => ({
      user,
      loading,
      unreachable,
      retry: restore,
      login: (email, password) => authenticate("/auth/login", email, password),
      signup: (email, password) => authenticate("/auth/signup", email, password),
      logout: () => {
        setToken(null);
        queryClient.clear();
        setUser(null);
      },
    }),
    [user, loading, unreachable, restore, authenticate, queryClient],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
