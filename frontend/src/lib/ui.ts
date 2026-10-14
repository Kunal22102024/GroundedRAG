// Shared class names, so buttons, fields and cards look the same everywhere.

const btn =
  "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-50";

export const btnPrimary = `${btn} bg-linear-to-r from-indigo-600 to-violet-600 px-4 py-2 text-white shadow-sm shadow-indigo-500/25 hover:from-indigo-500 hover:to-violet-500 hover:shadow-md hover:shadow-indigo-500/25 active:scale-[0.98] disabled:active:scale-100`;
export const btnSecondary = `${btn} border border-slate-200 bg-white px-3 py-2 text-slate-700 shadow-xs hover:border-slate-300 hover:bg-slate-50 active:scale-[0.98]`;
export const btnDanger = `${btn} bg-rose-600 px-4 py-2 text-white shadow-sm hover:bg-rose-500 active:scale-[0.98]`;
export const btnGhost = `${btn} px-2.5 py-1.5 text-slate-600 hover:bg-slate-100 hover:text-slate-900`;
export const iconBtn =
  "inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40";

export const inputBase =
  "w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-xs transition placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10";
export const selectBase = `${inputBase} cursor-pointer pr-8`;
export const fieldLabel = "mb-1 block text-xs font-medium text-slate-500";

export const card = "rounded-2xl border border-slate-200/80 bg-white shadow-sm";
