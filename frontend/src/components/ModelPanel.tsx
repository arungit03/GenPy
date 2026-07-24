import { AnimatePresence, motion } from 'framer-motion';
import { Cpu, Database, HardDrive, Info, type LucideIcon, X } from 'lucide-react';

import type { HealthResponse, ModelResponse } from '../types/chat';
import { formatCompactNumber, formatDateTime } from '../utils/format';
import { IconButton } from './IconButton';

interface ModelPanelProps {
  open: boolean;
  health: HealthResponse | null;
  model: ModelResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onClose: () => void;
}

export function ModelPanel({ open, health, model, loading, error, onRefresh, onClose }: ModelPanelProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex justify-end bg-surface-950/35"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <button type="button" className="absolute inset-0" aria-label="Close model panel" onClick={onClose} />
          <motion.aside
            initial={{ x: 420 }}
            animate={{ x: 0 }}
            exit={{ x: 420 }}
            transition={{ type: 'spring', damping: 30, stiffness: 280 }}
            className="panel relative h-full w-full max-w-md overflow-y-auto"
          >
            <div className="flex h-16 items-center justify-between border-b border-surface-200 px-5 dark:border-surface-800">
              <h2 className="text-sm font-semibold text-surface-950 dark:text-white">Model</h2>
              <IconButton label="Close model panel" onClick={onClose}>
                <X size={18} />
              </IconButton>
            </div>

            <div className="space-y-5 p-5">
              {loading && <ModelSkeleton />}

              {!loading && error && (
                <div className="rounded-md border border-ember-500/40 bg-ember-500/10 p-4 text-sm text-surface-900 dark:text-surface-100">
                  <p>{error}</p>
                  <button
                    type="button"
                    onClick={onRefresh}
                    className="mt-3 rounded-md bg-ember-500 px-3 py-2 text-sm font-medium text-white"
                  >
                    Retry
                  </button>
                </div>
              )}

              {!loading && !error && model && (
                <>
                  <InfoCard icon={Info} label="Name" value={model.model_name} />
                  <InfoCard icon={Cpu} label="Device" value={health?.device ?? model.device} />
                  <InfoCard icon={Database} label="Parameters" value={formatCompactNumber(model.parameter_count)} />
                  <InfoCard icon={HardDrive} label="Context" value={`${model.context_length} tokens`} />

                  <dl className="grid gap-3 rounded-md border border-surface-200 p-4 text-sm dark:border-surface-800">
                    <Row label="Vocabulary" value={formatCompactNumber(model.vocabulary_size)} />
                    <Row label="Quantization" value={model.quantization ?? 'None'} />
                    <Row label="LoRA" value={model.lora_enabled ? model.lora_adapter ?? 'Enabled' : 'Disabled'} />
                    <Row label="Loaded" value={formatDateTime(model.loaded_at)} />
                    <Row label="Checkpoint" value={model.checkpoint_path} />
                    <Row label="Tokenizer" value={model.tokenizer_path} />
                  </dl>
                </>
              )}

              {!loading && !error && !model && (
                <div className="rounded-md border border-surface-200 p-4 text-sm text-surface-700 dark:border-surface-800 dark:text-surface-200">
                  Model information is not available.
                </div>
              )}
            </div>
          </motion.aside>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

interface InfoCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
}

function InfoCard({ icon: Icon, label, value }: InfoCardProps) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-surface-200 bg-surface-50 p-4 dark:border-surface-800 dark:bg-surface-950">
      <div className="grid h-10 w-10 place-items-center rounded-md bg-brand-500/12 text-brand-600 dark:text-brand-400">
        <Icon size={19} />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase text-surface-700 dark:text-surface-200">{label}</p>
        <p className="truncate text-sm font-medium text-surface-950 dark:text-white" title={value}>
          {value}
        </p>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1">
      <dt className="text-xs font-semibold uppercase text-surface-700 dark:text-surface-200">{label}</dt>
      <dd className="break-words text-surface-950 dark:text-white">{value}</dd>
    </div>
  );
}

function ModelSkeleton() {
  return (
    <div className="space-y-4">
      {[0, 1, 2, 3].map((item) => (
        <div
          key={item}
          className="h-20 animate-pulse rounded-md border border-surface-200 bg-surface-100 dark:border-surface-800 dark:bg-surface-800"
        />
      ))}
    </div>
  );
}
