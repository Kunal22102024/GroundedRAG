import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// `npm run dev` proxies the API to the Docker stack's published port. If HTTP_PORT in .env is not
// the default 8081, start it as `VITE_PROXY_TARGET=http://localhost:<port> npm run dev`.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  return {
    plugins: [react(), tailwindcss()],
    server: { port: 5173, proxy: { "/api": env.VITE_PROXY_TARGET ?? "http://localhost:8081" } },
  };
});
