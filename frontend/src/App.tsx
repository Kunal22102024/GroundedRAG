import { Link, Navigate, NavLink, Outlet, Route, Routes } from "react-router-dom";
import { useAuth } from "./hooks/useAuth";
import ChatPage from "./pages/ChatPage";
import LibraryPage from "./pages/LibraryPage";
import LoginPage from "./pages/LoginPage";

function ProtectedLayout() {
  const { user, loading, logout } = useAuth();
  if (loading) return <p className="p-8 text-slate-500">Loading…</p>;
  if (!user) return <Navigate to="/login" replace />;
  const tab = ({ isActive }: { isActive: boolean }) =>
    `rounded px-3 py-1.5 text-sm font-medium ${isActive ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-slate-100"}`;
  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-4 border-b bg-white px-4 py-2">
        <span className="text-lg font-semibold text-indigo-700">GroundedRAG</span>
        <nav className="flex gap-1">
          <NavLink to="/chat" className={tab}>
            Chat
          </NavLink>
          <NavLink to="/library" className={tab}>
            Library
          </NavLink>
        </nav>
        <div className="ml-auto flex items-center gap-3 text-sm text-slate-600">
          <span>
            {user.email} <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">{user.role}</span>
          </span>
          <button onClick={logout} className="rounded border px-2 py-1 hover:bg-slate-100">
            Log out
          </button>
        </div>
      </header>
      <main className="min-h-0 flex-1">
        <Outlet />
      </main>
    </div>
  );
}

function NotFoundPage() {
  return (
    <div className="p-10 text-center">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <Link to="/chat" className="mt-4 inline-block text-indigo-600 underline">
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
