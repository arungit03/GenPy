import { CornerDownLeft, Square } from 'lucide-react';
import { type KeyboardEvent, useEffect, useRef, useState } from 'react';

interface ComposerProps {
  disabled: boolean;
  isGenerating: boolean;
  onSubmit: (content: string) => void;
  onStop: () => void;
}

export function Composer({ disabled, isGenerating, onSubmit, onStop }: ComposerProps) {
  const [content, setContent] = useState('');
  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const element = textAreaRef.current;
    if (!element) {
      return;
    }

    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }, [content]);

  const submit = () => {
    const trimmed = content.trim();
    if (!trimmed || disabled || isGenerating) {
      return;
    }

    onSubmit(trimmed);
    setContent('');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-surface-200 bg-surface-50/90 px-3 py-3 backdrop-blur dark:border-surface-800 dark:bg-surface-950/90 sm:px-6">
      <div className="mx-auto flex max-w-4xl items-end gap-2 rounded-md border border-surface-200 bg-white p-2 shadow-soft transition focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/20 dark:border-surface-700 dark:bg-surface-900">
        <textarea
          ref={textAreaRef}
          value={content}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="Message GenPy"
          className="min-h-11 flex-1 resize-none bg-transparent px-2 py-2.5 text-[15px] leading-6 text-surface-950 outline-none placeholder:text-surface-700/70 disabled:cursor-not-allowed dark:text-white dark:placeholder:text-surface-200/55"
        />

        {isGenerating ? (
          <button
            type="button"
            onClick={onStop}
            className="grid h-11 w-11 shrink-0 place-items-center rounded-md bg-ember-500 text-white transition hover:bg-ember-400 focus:outline-none focus:ring-2 focus:ring-ember-500/40"
            aria-label="Stop generation"
            title="Stop generation"
          >
            <Square size={17} />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!content.trim() || disabled}
            className="grid h-11 w-11 shrink-0 place-items-center rounded-md bg-brand-500 text-white transition hover:bg-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-500/40 disabled:cursor-not-allowed disabled:bg-surface-300 disabled:text-surface-700 dark:disabled:bg-surface-700 dark:disabled:text-surface-300"
            aria-label="Send message"
            title="Send message"
          >
            <CornerDownLeft size={18} />
          </button>
        )}
      </div>
    </div>
  );
}
