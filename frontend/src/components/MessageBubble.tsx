import { motion } from 'framer-motion';
import { Check, Copy, RefreshCw } from 'lucide-react';
import { useState } from 'react';

import type { ChatMessage } from '../types/chat';
import { formatDuration } from '../utils/format';
import { IconButton } from './IconButton';
import { MarkdownContent } from './MarkdownContent';
import { TypingIndicator } from './TypingIndicator';

interface MessageBubbleProps {
  message: ChatMessage;
  isDark: boolean;
  onRetry: (messageId: string) => void;
}

export function MessageBubble({ message, isDark, onRetry }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const isPending = message.status === 'pending';
  const isError = message.status === 'error';

  const copyMessage = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      className={`group flex w-full gap-3 px-4 py-6 sm:px-6 ${isUser ? 'bg-transparent' : 'bg-white/70 dark:bg-surface-900/45'}`}
    >
      <div
        className={`grid h-9 w-9 shrink-0 place-items-center rounded-md text-sm font-semibold ${
          isUser
            ? 'bg-surface-900 text-white dark:bg-surface-100 dark:text-surface-950'
            : isError
              ? 'bg-ember-500 text-white'
              : 'bg-brand-500 text-white'
        }`}
      >
        {isUser ? 'You' : 'GP'}
      </div>

      <div className="min-w-0 flex-1">
        <div className="mb-2 flex min-h-9 items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-surface-900 dark:text-white">
              {isUser ? 'You' : 'GenPy'}
            </p>
            {message.metrics && (
              <p className="text-xs text-surface-700 dark:text-surface-200">
                {message.metrics.tokensGenerated} tokens in {formatDuration(message.metrics.generationTime)} ·{' '}
                {Math.round(message.metrics.tokensPerSecond)} tok/s
              </p>
            )}
            {message.status === 'stopped' && (
              <p className="text-xs text-surface-700 dark:text-surface-200">Stopped before completion</p>
            )}
          </div>

          <div className="flex shrink-0 items-center gap-1 opacity-100 sm:opacity-0 sm:transition sm:group-hover:opacity-100">
            {isError && (
              <IconButton label="Retry message" onClick={() => onRetry(message.id)}>
                <RefreshCw size={16} />
              </IconButton>
            )}
            {message.content && (
              <IconButton label="Copy message" onClick={copyMessage}>
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </IconButton>
            )}
          </div>
        </div>

        {isPending ? (
          <TypingIndicator />
        ) : isUser ? (
          <p className="whitespace-pre-wrap text-[15px] leading-7 text-surface-900 dark:text-surface-100">
            {message.content}
          </p>
        ) : isError ? (
          <div className="rounded-md border border-ember-500/40 bg-ember-500/10 p-3 text-sm leading-6 text-surface-900 dark:text-surface-100">
            {message.content}
          </div>
        ) : (
          <MarkdownContent content={message.content} isDark={isDark} />
        )}
      </div>
    </motion.article>
  );
}
