import { AnimatePresence, motion } from 'framer-motion';
import { Moon, Monitor, Sun, X } from 'lucide-react';

import type { GenerationSettings, ThemePreference } from '../types/chat';
import { IconButton } from './IconButton';

interface SettingsPanelProps {
  open: boolean;
  settings: GenerationSettings;
  theme: ThemePreference;
  onUpdateSettings: (settings: Partial<GenerationSettings>) => void;
  onThemeChange: (theme: ThemePreference) => void;
  onClose: () => void;
}

export function SettingsPanel({
  open,
  settings,
  theme,
  onUpdateSettings,
  onThemeChange,
  onClose,
}: SettingsPanelProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex justify-end bg-surface-950/35"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <button type="button" className="absolute inset-0" aria-label="Close settings" onClick={onClose} />
          <motion.aside
            initial={{ x: 420 }}
            animate={{ x: 0 }}
            exit={{ x: 420 }}
            transition={{ type: 'spring', damping: 30, stiffness: 280 }}
            className="panel relative h-full w-full max-w-md overflow-y-auto"
          >
            <div className="flex h-16 items-center justify-between border-b border-surface-200 px-5 dark:border-surface-800">
              <h2 className="text-sm font-semibold text-surface-950 dark:text-white">Settings</h2>
              <IconButton label="Close settings" onClick={onClose}>
                <X size={18} />
              </IconButton>
            </div>

            <div className="space-y-7 p-5">
              <section>
                <h3 className="mb-4 text-xs font-semibold uppercase text-surface-700 dark:text-surface-200">
                  Sampling
                </h3>
                <Slider
                  label="Temperature"
                  min={0.1}
                  max={2}
                  step={0.1}
                  value={settings.temperature}
                  onChange={(temperature) => onUpdateSettings({ temperature })}
                />
                <Slider
                  label="Top P"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={settings.topP}
                  onChange={(topP) => onUpdateSettings({ topP })}
                />
                <Slider
                  label="Max Tokens"
                  min={16}
                  max={4096}
                  step={16}
                  value={settings.maxNewTokens}
                  onChange={(maxNewTokens) => onUpdateSettings({ maxNewTokens })}
                />
              </section>

              <section>
                <h3 className="mb-4 text-xs font-semibold uppercase text-surface-700 dark:text-surface-200">
                  Theme
                </h3>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { value: 'light' as const, label: 'Light', icon: Sun },
                    { value: 'dark' as const, label: 'Dark', icon: Moon },
                    { value: 'system' as const, label: 'System', icon: Monitor },
                  ].map((item) => {
                    const Icon = item.icon;
                    const active = theme === item.value;

                    return (
                      <button
                        key={item.value}
                        type="button"
                        onClick={() => onThemeChange(item.value)}
                        className={`flex h-20 flex-col items-center justify-center gap-2 rounded-md border text-sm font-medium transition ${
                          active
                            ? 'border-brand-500 bg-brand-500/10 text-brand-600 dark:text-brand-400'
                            : 'border-surface-200 text-surface-700 hover:bg-surface-100 dark:border-surface-700 dark:text-surface-200 dark:hover:bg-surface-800'
                        }`}
                      >
                        <Icon size={20} />
                        {item.label}
                      </button>
                    );
                  })}
                </div>
              </section>
            </div>
          </motion.aside>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

interface SliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}

function Slider({ label, min, max, step, value, onChange }: SliderProps) {
  return (
    <label className="mb-5 block">
      <span className="mb-2 flex items-center justify-between gap-4 text-sm text-surface-800 dark:text-surface-100">
        <span>{label}</span>
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="h-9 w-24 rounded-md border border-surface-200 bg-white px-2 text-right text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:border-surface-700 dark:bg-surface-900"
        />
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-2 w-full accent-brand-500"
      />
    </label>
  );
}
