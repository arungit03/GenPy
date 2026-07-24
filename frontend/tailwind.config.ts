import type { Config } from 'tailwindcss';

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          50: '#f8faf9',
          100: '#eef2ef',
          200: '#dbe3de',
          700: '#33413a',
          800: '#202a25',
          900: '#121815',
          950: '#0b100e',
        },
        brand: {
          400: '#2fb9a1',
          500: '#159783',
          600: '#0f7769',
        },
        ember: {
          400: '#f98a67',
          500: '#ef6f4a',
        },
      },
      boxShadow: {
        soft: '0 18px 55px -36px rgba(15, 23, 42, 0.55)',
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'sans-serif',
        ],
      },
    },
  },
  plugins: [],
} satisfies Config;
