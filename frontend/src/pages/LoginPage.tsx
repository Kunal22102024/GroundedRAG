import { useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

export default function LoginPage({ mode }: { mode: "login" | "signup" }) {
  const { user, login, signup } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/chat" replace />;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (mode === "login" ? login : signup)(email, password);
      navigate("/chat");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-xl border bg-white p-6 shadow-sm">
        <h1 className="text-xl font-semibold text-indigo-700">GroundedRAG</h1>
        <p className="text-sm text-slate-500">
          {mode === "login" ? "Sign in to ask questions about your documents." : "Create an account. The first account becomes the administrator."}
        </p>
        <label className="block text-sm">
          Email
          <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="mt-1 w-full rounded border px-3 py-2" />
        </label>
        <label className="block text-sm">
          Password
          <input
            type="password"
            required
            minLength={mode === "signup" ? 8 : 1}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1 w-full rounded border px-3 py-2"
          />
          {mode === "signup" && <span className="text-xs text-slate-500">At least 8 characters.</span>}
        </label>
        {error && <p className="rounded bg-red-50 p-2 text-sm text-red-700">{error}</p>}
        <button disabled={busy} className="w-full rounded bg-indigo-600 py-2 font-medium text-white hover:bg-indigo-700 disabled:opacity-60">
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
        <p className="text-center text-sm text-slate-500">
          {mode === "login" ? (
            <>
              No account? <Link to="/signup" className="text-indigo-600 underline">Sign up</Link>
            </>
          ) : (
            <>
              Have an account? <Link to="/login" className="text-indigo-600 underline">Sign in</Link>
            </>
          )}
        </p>
      </form>
    </div>
  );
}
