import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { AlertCircleIcon, ShieldIcon, TrashIcon, UsersIcon } from "../components/ui/icons";
import { Modal, Skeleton, Spinner } from "../components/ui/primitives";
import { useToast } from "../components/ui/Toast";
import { useAuth } from "../hooks/useAuth";
import { formatDate } from "../lib/format";
import type { Role, UserAdminItem } from "../lib/types";
import { btnDanger, btnPrimary, btnSecondary, card, iconBtn } from "../lib/ui";

type Pending = { kind: "delete"; user: UserAdminItem } | { kind: "role"; user: UserAdminItem; role: Role };

const ROLE_PILL: Record<Role, string> = {
  admin: "bg-violet-100 text-violet-700",
  user: "bg-slate-100 text-slate-600",
};

/** Accounts: who is an administrator, promote, demote or remove (administrators only). */
export default function UsersPage() {
  const { user: me, retry } = useAuth();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [pending, setPending] = useState<Pending | null>(null);
  const cancel = useRef<HTMLButtonElement>(null);

  const list = useQuery({ queryKey: ["users"], queryFn: () => api<UserAdminItem[]>("/users"), enabled: me?.role === "admin" });

  const changeRole = useMutation({
    mutationFn: ({ user, role }: { user: UserAdminItem; role: Role }) => api<UserAdminItem>(`/users/${user.id}`, { method: "PATCH", body: JSON.stringify({ role }) }),
    onSuccess: (updated, { user }) => {
      setPending(null);
      toast({ tone: "success", title: updated.role === "admin" ? "Now an administrator" : "Now a regular user", description: user.email });
      void queryClient.invalidateQueries({ queryKey: ["users"] });
      if (user.id === me?.id) {
        retry(); // your own role changed: reload who you are, then leave a page you may no longer open
        navigate("/chat");
      }
    },
    onError: (e: unknown) => {
      setPending(null);
      toast({ tone: "error", title: "Could not change the role", description: e instanceof Error ? e.message : undefined });
    },
  });
  const remove = useMutation({
    mutationFn: (user: UserAdminItem) => api<void>(`/users/${user.id}`, { method: "DELETE" }),
    onSuccess: (_, user) => {
      setPending(null);
      toast({ tone: "success", title: "Account deleted", description: user.email });
      void queryClient.invalidateQueries({ queryKey: ["users"] });
    },
    onError: (e: unknown) => {
      setPending(null);
      toast({ tone: "error", title: "Could not delete the account", description: e instanceof Error ? e.message : undefined });
    },
  });

  if (me && me.role !== "admin") return <Navigate to="/chat" replace />;

  const users = list.data ?? [];
  const admins = users.filter((u) => u.role === "admin").length;
  const busy = changeRole.isPending || remove.isPending;

  /** Promote at once; demoting yourself asks first, because it takes this page away. */
  const setRole = (user: UserAdminItem, role: Role) => {
    if (role === user.role) return;
    if (user.id === me?.id && role === "user") setPending({ kind: "role", user, role });
    else changeRole.mutate({ user, role });
  };

  return (
    <div className="h-full overflow-y-auto bg-linear-to-b from-slate-50 to-white">
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight text-slate-900">
            <UsersIcon size={22} className="text-indigo-500" /> Users
          </h1>
          <p className="max-w-3xl text-sm text-slate-500">
            Everyone signs in on the same page; the role decides what they can do. Administrators can read the Restricted documents, see the Evaluation dashboard and manage accounts. A change applies at once.
          </p>
        </div>

        {list.isError && (
          <p role="alert" className="flex items-start gap-2 rounded-xl bg-rose-50 p-3.5 text-sm text-rose-700">
            <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> Could not load the accounts. {list.error instanceof Error ? list.error.message : ""}
          </p>
        )}

        <section className={`${card} animate-fade-up overflow-hidden`} aria-label="Accounts">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50/70 text-xs font-medium text-slate-500">
                <tr>
                  <th className="px-4 py-2.5">Account</th>
                  <th className="px-4 py-2.5">Role</th>
                  <th className="hidden px-4 py-2.5 sm:table-cell">Joined</th>
                  <th className="hidden px-4 py-2.5 text-right sm:table-cell">Documents</th>
                  <th className="hidden px-4 py-2.5 text-right sm:table-cell">Conversations</th>
                  <th className="px-4 py-2.5" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {list.isLoading &&
                  [0, 1, 2].map((i) => (
                    <tr key={i}>
                      <td colSpan={6} className="px-4 py-3">
                        <Skeleton className="h-6" />
                      </td>
                    </tr>
                  ))}
                {users.map((u) => {
                  const isMe = u.id === me?.id;
                  const onlyAdmin = u.role === "admin" && admins <= 1;
                  return (
                    <tr key={u.id} className="hover:bg-slate-50/60">
                      <td className="max-w-[11rem] px-4 py-3 sm:max-w-none">
                        <div className="flex items-center gap-2.5">
                          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-linear-to-br from-indigo-500 to-violet-500 text-xs font-semibold uppercase text-white">{u.email.slice(0, 1)}</span>
                          <span className="min-w-0 truncate font-medium text-slate-900" title={u.email}>
                            {u.email}
                          </span>
                          {isMe && <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-[11px] font-medium text-indigo-700">you</span>}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <label className="sr-only" htmlFor={`role-${u.id}`}>
                          Role of {u.email}
                        </label>
                        <select
                          id={`role-${u.id}`}
                          value={u.role}
                          disabled={busy || onlyAdmin}
                          title={onlyAdmin ? "The only administrator: make someone else an administrator first" : undefined}
                          onChange={(e) => setRole(u, e.target.value as Role)}
                          className={`cursor-pointer rounded-full border-0 py-1 pl-3 pr-7 text-xs font-medium disabled:cursor-not-allowed ${ROLE_PILL[u.role]}`}
                        >
                          <option value="user">user</option>
                          <option value="admin">admin</option>
                        </select>
                      </td>
                      <td className="hidden whitespace-nowrap px-4 py-3 text-slate-500 sm:table-cell" title={new Date(u.created_at).toLocaleString()}>
                        {formatDate(u.created_at.slice(0, 10))}
                      </td>
                      <td className="hidden px-4 py-3 text-right tabular-nums text-slate-600 sm:table-cell">{u.documents}</td>
                      <td className="hidden px-4 py-3 text-right tabular-nums text-slate-600 sm:table-cell">{u.conversations}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => setPending({ kind: "delete", user: u })}
                          disabled={isMe || busy}
                          aria-label={`Delete ${u.email}`}
                          title={isMe ? "You cannot delete your own account" : "Delete account"}
                          className={`${iconBtn} hover:bg-rose-50 hover:text-rose-600`}
                        >
                          <TrashIcon size={16} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
        <p className="flex items-center gap-1.5 text-xs text-slate-400">
          <ShieldIcon size={13} /> There is always at least one administrator. New sign-ups start as regular users.
        </p>
      </div>

      <Modal open={pending !== null} onClose={() => setPending(null)} label={pending?.kind === "delete" ? "Delete account" : "Remove your administrator role"} initialFocus={cancel}>
        {pending && (
          <div className="space-y-5">
            <div className="flex items-start gap-4">
              <span className={`grid h-11 w-11 shrink-0 place-items-center rounded-full ${pending.kind === "delete" ? "bg-rose-50 text-rose-600" : "bg-amber-50 text-amber-600"}`}>
                {pending.kind === "delete" ? <TrashIcon size={20} /> : <ShieldIcon size={20} />}
              </span>
              <div className="min-w-0 pr-6">
                <h2 className="font-semibold text-slate-900">{pending.kind === "delete" ? "Delete this account?" : "Remove your own administrator role?"}</h2>
                <p className="mt-1 break-words text-sm text-slate-500">
                  {pending.kind === "delete" ? (
                    <>
                      <b className="text-slate-700">{pending.user.email}</b> will no longer be able to sign in, and their {pending.user.conversations} conversation(s) are deleted. The {pending.user.documents} document(s) they uploaded stay in the library. This cannot be undone.
                    </>
                  ) : (
                    <>You will become a regular user at once and lose this page, the Evaluation dashboard and access to Restricted documents. Another administrator can give it back.</>
                  )}
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <button ref={cancel} onClick={() => setPending(null)} className={btnSecondary}>
                Cancel
              </button>
              {pending.kind === "delete" ? (
                <button onClick={() => remove.mutate(pending.user)} disabled={remove.isPending} className={btnDanger}>
                  {remove.isPending ? <Spinner size={14} /> : <TrashIcon size={15} />} {remove.isPending ? "Deleting…" : "Delete"}
                </button>
              ) : (
                <button onClick={() => changeRole.mutate({ user: pending.user, role: pending.role })} disabled={changeRole.isPending} className={btnPrimary}>
                  {changeRole.isPending ? <Spinner size={14} /> : null} Make me a regular user
                </button>
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
