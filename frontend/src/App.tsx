import { Link, Navigate, NavLink, Outlet, Route, Routes, useLocation } from "react-router-dom";
import { AlertCircleIcon, ChatIcon, LibraryIcon, LogOutIcon, LogoMark } from "./components/ui/icons";
import { Spinner } from "./components/ui/primitives";
import { useAuth } from "./hooks/useAuth";
import { btnPrimary, iconBtn } from "./lib/ui";
import ChatPage from "./pages/ChatPage";
import LibraryPage from "./pages/LibraryPage";
import LoginPage from "./pages/LoginPage";

function FullPageLoading() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-slate-500">
      <span className="animate-float">
        <LogoMark size={44} />
      </span>
      <span className="flex items-center gap-2 text-sm">
        <Spinner className="text-indigo-500" /> Loading…
      </span>
    </div>
  );
}

function ProtectedLayout() {
  const { user, loading, unreachable, retry, logout } = useAuth();
  const location = useLocation();
  if (loading) return <FullPageLoading />;
  if (unreachable)
    return (
      <div className="flex h-full items-center justify-center p-4">
        <div role="alert" className="w-full max-w-md animate-scale-in space-y-4 rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
          <span className="mx-auto grid h-12 w-12 place-items-center rounded-full bg-amber-50 text-amber-500">
            <AlertCircleIcon size={24} />
          </span>
          <h1 className="text-lg font-semibold">Cannot reach the server</h1>
          <p className="text-sm text-slate-500">You are still signed in. The server may be restarting.</p>
          <button onClick={retry} className={btnPrimary}>
            Try again
          </button>
        </div>
      </div>
    );
  if (!user) return <Navigate to="/login" replace />;

  const tab = ({ isActive }: { isActive: boolean }) =>
    `relative flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-all ${
      isActive ? "bg-white text-indigo-700 shadow-sm" : "text-slate-600 hover:text-slate-900"
    }`;
  const section = location.pathname.split("/")[1] || "chat";
  return (
    <div className="flex h-full flex-col">
      <header className="z-20 flex items-center gap-3 border-b border-slate-200/80 bg-white/80 px-3 py-2 backdrop-blur-md sm:gap-6 sm:px-5">
        <Link to="/chat" className="flex items-center gap-2.5" aria-label="GroundedRAG home">
          <LogoMark size={30} />
          <span className="hidden text-[15px] font-semibold tracking-tight text-slate-900 sm:block">
            Grounded<span className="bg-linear-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">RAG</span>
          </span>
        </Link>
        <nav className="flex gap-1 rounded-xl bg-slate-100 p-1" aria-label="Main">
          <NavLink to="/chat" className={tab}>
            <ChatIcon size={16} /> Chat
          </NavLink>
          <NavLink to="/library" className={tab}>
            <LibraryIcon size={16} /> Library
          </NavLink>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <div className="hidden items-center gap-2.5 rounded-full border border-slate-200 bg-white py-1 pl-1 pr-3 sm:flex">
            <span className="grid h-7 w-7 place-items-center rounded-full bg-linear-to-br from-indigo-500 to-violet-500 text-xs font-semibold uppercase text-white">
              {user.email.slice(0, 1)}
            </span>
            <span className="max-w-48 truncate text-sm text-slate-700" title={user.email}>
              {user.email}
            </span>
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${user.role === "admin" ? "bg-violet-100 text-violet-700" : "bg-slate-100 text-slate-600"}`}
            >
              {user.role}
            </span>
          </div>
          <button onClick={logout} aria-label="Log out" title="Log out" className={`${iconBtn} h-9 w-9`}>
            <LogOutIcon size={18} />
          </button>
        </div>
      </header>
      <main key={section} className="min-h-0 flex-1 animate-fade-in">
        <Outlet />
      </main>
    </div>
  );
}

function NotFoundPage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-10 text-center">
      <LogoMark size={44} />
      <p className="mt-2 bg-linear-to-r from-indigo-600 to-violet-600 bg-clip-text text-5xl font-bold text-transparent">404</p>
      <h1 className="text-lg font-semibold text-slate-800">This page does not exist</h1>
      <Link to="/chat" className={`${btnPrimary} mt-2`}>
        Back to chat
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage mode="login" />} />
      <Route path="/signup" element={<LoginPage mode="signup" />} />
      <Route element={<ProtectedLayout />}>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:conversationId" element={<ChatPage />} />
        <Route path="/library" element={<LibraryPage />} />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
