import { AnimatePresence } from 'framer-motion';
import { Bot, Sparkles } from 'lucide-react';
import { useEffect, useRef } from 'react';

import type { ChatMessage } from '../types/chat';
import { MessageBubble } from './MessageBubble';

interface MessageListProps {
  messages: ChatMessage[];
  isDark: boolean;
  onRetry: (messageId: string) => void;
}

export function MessageList({ messages, isDark, onRetry }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex min-h-full flex-col items-center justify-center px-6 py-16 text-center">
        <div className="mb-5 grid h-16 w-16 place-items-center rounded-md bg-brand-500 text-white shadow-soft">
          <Bot size={28} />
        </div>
        <h1 className="text-3xl font-semibold text-surface-950 dark:text-white sm:text-4xl">GenPy Chat</h1>
        <p className="mt-3 max-w-xl text-base leading-7 text-surface-700 dark:text-surface-200">
          Ask about Python, inspect generated code, or iterate on an implementation with your local model.
        </p>
        <div className="mt-8 grid w-full max-w-3xl gap-3 sm:grid-cols-3">
          {[
            'Write a FastAPI endpoint for user profiles.',
            'Explain this dynamic programming solution.',
            'Generate pytest cases for a tokenizer.',
          ].map((prompt) => (
            <div
              key={prompt}
              className="rounded-md border border-surface-200 bg-white p-4 text-left text-sm leading-6 text-surface-800 shadow-sm dark:border-surface-800 dark:bg-surface-900 dark:text-surface-100"
            >
              <Sparkles className="mb-3 text-brand-500" size={18} />
              {prompt}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-full">
      <AnimatePresence initial={false}>
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} isDark={isDark} onRetry={onRetry} />
        ))}
      </AnimatePresence>
      <div ref={bottomRef} className="h-3" />
    </div>
  );
}
