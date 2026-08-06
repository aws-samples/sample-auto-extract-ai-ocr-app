/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: 'var(--color-bg)',
        overlay: 'var(--color-overlay)',
        'on-primary': 'var(--color-on-primary)',
        primary: {
          DEFAULT: 'var(--color-primary)',
          hover: 'var(--color-primary-hover)',
        },
        danger: {
          DEFAULT: 'var(--color-danger)',
          hover: 'var(--color-danger-hover)',
          light: 'var(--color-danger-light)',
          border: 'var(--color-danger-border)',
          text: 'var(--color-danger-text)',
        },
        success: {
          DEFAULT: 'var(--color-success)',
          hover: 'var(--color-success-hover)',
          light: 'var(--color-success-light)',
          border: 'var(--color-success-border)',
          text: 'var(--color-success-text)',
        },
        warning: {
          DEFAULT: 'var(--color-warning)',
          light: 'var(--color-warning-light)',
          border: 'var(--color-warning-border)',
          text: 'var(--color-warning-text)',
        },
        info: {
          DEFAULT: 'var(--color-info)',
          hover: 'var(--color-info-hover)',
          light: 'var(--color-info-light)',
          border: 'var(--color-info-border)',
          text: 'var(--color-info-text)',
        },
        surface: {
          DEFAULT: 'var(--color-surface)',
          alt: 'var(--color-surface-alt)',
        },
        header: {
          DEFAULT: 'var(--color-header)',
          hover: 'var(--color-header-hover)',
        },
        muted: 'var(--color-muted)',
        neutral: {
          50: 'var(--color-neutral-50)',
          100: 'var(--color-neutral-100)',
          200: 'var(--color-neutral-200)',
          300: 'var(--color-neutral-300)',
          400: 'var(--color-neutral-400)',
          500: 'var(--color-neutral-500)',
          600: 'var(--color-neutral-600)',
          700: 'var(--color-neutral-700)',
          800: 'var(--color-neutral-800)',
          900: 'var(--color-neutral-900)',
        },
        accent: {
          light: 'var(--color-accent-light)',
          text: 'var(--color-accent-text)',
          border: 'var(--color-accent-border)',
        },
      },
      textColor: {
        default: 'var(--color-text)',
        light: 'var(--color-text-light)',
      },
      borderColor: {
        default: 'var(--color-border)',
        light: 'var(--color-border-light)',
      },
    },
  },
  plugins: [],
}
