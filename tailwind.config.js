/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './frontend/templates/**/*.html',
    './frontend/static/js/**/*.js',
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme color palette
        dark: {
          bg: '#0f172a',        // Main background (slate-900)
          card: '#1e293b',      // Card background (slate-800)
          hover: '#334155',     // Hover state (slate-700)
          border: '#475569',    // Borders (slate-600)
          text: '#e2e8f0',      // Primary text (slate-200)
          muted: '#94a3b8',     // Muted text (slate-400)
        },
        accent: {
          primary: '#3b82f6',   // Primary accent (blue-500)
          hover: '#2563eb',     // Primary hover (blue-600)
          success: '#22c55e',   // Success (green-500)
          warning: '#eab308',   // Warning (yellow-500)
          error: '#ef4444',     // Error (red-500)
          info: '#06b6d4',      // Info (cyan-500)
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(59, 130, 246, 0.3)',
        'glow-success': '0 0 20px rgba(34, 197, 94, 0.3)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
