import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // The main chunk (react-router + react-query + radix primitives + react-hook-form)
  // settles around ~600 kB minified for this app; Dashboard's recharts and the
  // Users page are already split into lazy-loaded chunks (see src/App.tsx).
  // Raising this just silences the cosmetic build warning for the remainder.
  build: {
    chunkSizeWarningLimit: 700,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    // The FastAPI backend's default CORS_ORIGINS allowlist (see ../.env.example)
    // expects the frontend on :3000 — keep this in sync if you change one side.
    proxy: {
      "/api/v1": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
