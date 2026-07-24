import { Menu, PanelRight, RefreshCw, RotateCcw, Settings } from 'lucide-react';

import type { HealthResponse, ModelResponse } from '../types/chat';
import { IconButton } from './IconButton';

interface TopBarProps {
  health: HealthResponse | null;
  model: ModelResponse | null;
  isGenerating: boolean;
  canRegenerate: boolean;
  onOpenSidebar: () => void;
  onRefreshStatus: () => void;
  onRegenerate: () => void;
  onOpenSettings: () => void;
  onOpenModel: () => void;
}

export function TopBar({
  health,
  model,
  isGenerating,
  canRegenerate,
  onOpenSidebar,
  onRefreshStatus,
  onRegenerate,
  onOpenSettings,
  onOpenModel,
}: TopBarProps) {
  const isHealthy = health?.status === 'healthy' && health.model_loaded;

  return (
    <header className="flex h-16 shrink-0 items-center gap-3 border-b border-surface-200 bg-surface-50/90 px-3 backdrop-blur dark:border-surface-800 dark:bg-surface-950/90 sm:px-6">
      <IconButton label="Open sidebar" className="lg:hidden" onClick={onOpenSidebar}>
        <Menu size={19} />
      </IconButton>

      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="truncate text-sm font-semibold text-surface-950 dark:text-white">
            {model?.model_name ?? 'GenPy GPT'}
          </h2>
          <span
            className={`h-2 w-2 shrink-0 rounded-full ${isHealthy ? 'bg-brand-500' : 'bg-ember-500'}`}
            aria-label={isHealthy ? 'Model ready' : 'Model unavailable'}
            title={isHealthy ? 'Model ready' : 'Model unavailable'}
          />
        </div>
        <p className="truncate text-xs text-surface-700 dark:text-surface-200">
          {health ? `${health.device} · ${health.model_loaded ? 'Loaded' : 'Not loaded'}` : 'Checking status'}
        </p>
      </div>

      <div className="flex items-center gap-1">
        <IconButton label="Refresh status" onClick={onRefreshStatus}>
          <RefreshCw size={17} />
        </IconButton>
        <IconButton label="Regenerate response" onClick={onRegenerate} disabled={!canRegenerate || isGenerating}>
          <RotateCcw size={17} />
        </IconButton>
        <IconButton label="Open settings" onClick={onOpenSettings}>
          <Settings size={17} />
        </IconButton>
        <IconButton label="Open model information" onClick={onOpenModel}>
          <PanelRight size={17} />
        </IconButton>
      </div>
    </header>
  );
}
