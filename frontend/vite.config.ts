import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const apiTarget = process.env.VITE_DEV_API_TARGET ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': apiTarget,
      '/model': apiTarget,
      '/generate': apiTarget,
      '/chat': apiTarget,
    },
  },
});
