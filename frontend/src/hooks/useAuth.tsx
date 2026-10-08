import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, getToken, setToken } from "../api/client";
import type { AuthResponse, User } from "../lib/types";

interface AuthValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(getToken() !== null);

  useEffect(() => {
    if (!getToken()) return;
    api<User>("/auth/me")
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const onLogout = () => setUser(null);
    window.addEventListener("auth:logout", onLogout);
    return () => window.removeEventListener("auth:logout", onLogout);
  }, []);

  const authenticate = useCallback(async (path: string, email: string, password: string) => {
    const result = await api<AuthResponse>(path, { method: "POST", body: JSON.stringify({ email, password }) });
    setToken(result.access_token);
    setUser(result.user);
  }, []);

  const value = useMemo<AuthValue>(
    () => ({
      user,
      loading,
      login: (email, password) => authenticate("/auth/login", email, password),
      signup: (email, password) => authenticate("/auth/signup", email, password),
      logout: () => {
        setToken(null);
        setUser(null);
      },
    }),
    [user, loading, authenticate],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
