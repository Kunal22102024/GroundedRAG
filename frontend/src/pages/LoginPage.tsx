import { useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { AlertCircleIcon, BookOpenIcon, EyeIcon, EyeOffIcon, LockIcon, LogoMark, MailIcon, SearchIcon, ShieldIcon } from "../components/ui/icons";
import { Spinner } from "../components/ui/primitives";
import { useAuth } from "../hooks/useAuth";
import { btnPrimary, inputBase } from "../lib/ui";

function Feature({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <li className="flex gap-3">
      <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-white/15 text-white ring-1 ring-white/20">{icon}</span>
      <div>
        <p className="font-medium text-white">{title}</p>
        <p className="text-sm text-indigo-100/80">{text}</p>
      </div>
    </li>
  );
}

export default function LoginPage({ mode }: { mode: "login" | "signup" }) {
  const { user, login, signup } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
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
      setAttempt((n) => n + 1);
    } finally {
      setBusy(false);
    }
  };
  const signingUp = mode === "signup";

  return (
    <div className="grid min-h-full lg:grid-cols-2">
      <aside className="relative hidden overflow-hidden bg-linear-to-br from-indigo-600 via-indigo-700 to-violet-700 p-12 lg:flex lg:flex-col lg:justify-between">
        <div className="pointer-events-none absolute -left-24 -top-24 h-96 w-96 rounded-full bg-violet-400/30 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-32 -right-16 h-96 w-96 rounded-full bg-indigo-300/20 blur-3xl" />
        <div className="relative flex items-center gap-3">
          <LogoMark size={36} />
          <span className="text-lg font-semibold text-white">GroundedRAG</span>
        </div>
        <div className="relative max-w-md animate-fade-up">
          <h2 className="text-3xl font-semibold leading-tight tracking-tight text-white">Answers you can verify, from documents you trust.</h2>
          <ul className="mt-8 space-y-5">
            <Feature icon={<BookOpenIcon size={18} />} title="Every claim is cited" text="Click a citation to see the exact passage and the page it came from." />
            <Feature icon={<SearchIcon size={18} />} title="Finds the right passage" text="Meaning-based and exact-keyword search together, so error codes and IDs are found too." />
            <Feature icon={<ShieldIcon size={18} />} title="Says so when it doesn't know" text="If your documents don't contain the answer, it tells you instead of guessing." />
          </ul>
        </div>
        <p className="relative text-xs text-indigo-200/70">PDF · Word · Markdown · HTML · Text</p>
      </aside>

      <div className="flex items-center justify-center bg-linear-to-b from-slate-50 to-white p-6">
        <form onSubmit={submit} className="w-full max-w-sm animate-fade-up">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <LogoMark size={36} />
            <span className="text-lg font-semibold">GroundedRAG</span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{signingUp ? "Create your account" : "Welcome back"}</h1>
          <p className="mt-1.5 text-sm text-slate-500">
            {signingUp ? "The first account becomes the administrator." : "Sign in to ask questions about your documents."}
          </p>

          <div className="mt-8 space-y-4">
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-slate-700">Email</span>
              <span className="relative block">
                <MailIcon size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="email"
                  required
                  autoComplete="email"
                  autoFocus
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className={`${inputBase} py-2.5 pl-9`}
                />
              </span>
            </label>
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-slate-700">Password</span>
              <span className="relative block">
                <LockIcon size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  autoComplete={signingUp ? "new-password" : "current-password"}
                  minLength={signingUp ? 8 : 1}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={signingUp ? "At least 8 characters" : "Your password"}
                  className={`${inputBase} py-2.5 pl-9 pr-10`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                >
                  {showPassword ? <EyeOffIcon size={16} /> : <EyeIcon size={16} />}
                </button>
              </span>
            </label>
          </div>

          {error && (
            <p key={attempt} role="alert" className="mt-4 flex animate-shake items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">
              <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> {error}
            </p>
          )}

          <button disabled={busy} className={`${btnPrimary} mt-6 w-full py-2.5`}>
            {busy && <Spinner size={15} />}
            {busy ? "Please wait…" : signingUp ? "Create account" : "Sign in"}
          </button>

          <p className="mt-6 text-center text-sm text-slate-500">
            {signingUp ? "Already have an account? " : "No account yet? "}
            <Link to={signingUp ? "/login" : "/signup"} className="font-medium text-indigo-600 hover:text-indigo-500">
              {signingUp ? "Sign in" : "Create one"}
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
