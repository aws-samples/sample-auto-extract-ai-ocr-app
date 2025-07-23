import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      './runtimeConfig': './runtimeConfig.browser', // ensures browser compatible version of AWS JS SDK is used
    },
  },
  server: {
    port: 3000,
  },
})
