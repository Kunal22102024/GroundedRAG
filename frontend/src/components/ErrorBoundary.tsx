import { Component, type ErrorInfo, type ReactNode } from "react";

/** Without this, one error while rendering a page leaves a blank screen with no way out. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The message and component stack only: it can be copied into a bug report.
    console.error("The page failed to render:", error.message, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div role="alert" className="mx-auto mt-24 max-w-md space-y-3 rounded-xl border bg-white p-6 text-center">
        <h1 className="text-lg font-semibold">Something went wrong</h1>
        <p className="text-sm text-slate-500">The page could not be displayed. Your documents and conversations are safe.</p>
        <button onClick={() => window.location.reload()} className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700">
          Reload the page
        </button>
      </div>
    );
  }
}
