import { AnimatePresence, motion } from 'framer-motion';
import { Edit3, MessageSquare, Plus, Search, Trash2, X } from 'lucide-react';
import { type RefObject, useMemo, useState } from 'react';

import type { Conversation } from '../types/chat';
import { formatRelativeTime } from '../utils/format';
import { IconButton } from './IconButton';

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  isOpen: boolean;
  searchInputRef: RefObject<HTMLInputElement | null>;
  onNewChat: () => void;
  onSelectChat: (conversationId: string) => void;
  onRenameChat: (conversationId: string, title: string) => void;
  onDeleteChat: (conversationId: string) => void;
  onClose: () => void;
}

export function Sidebar({
  conversations,
  activeConversationId,
  isOpen,
  searchInputRef,
  onNewChat,
  onSelectChat,
  onRenameChat,
  onDeleteChat,
  onClose,
}: SidebarProps) {
  const [query, setQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');

  const visibleConversations = useMemo(() => {
    const normalized = query.trim().toLowerCase();

    if (!normalized) {
      return conversations;
    }

    return conversations.filter((conversation) => {
      const haystack = [
        conversation.title,
        ...conversation.messages.map((message) => message.content),
      ]
        .join(' ')
        .toLowerCase();

      return haystack.includes(normalized);
    });
  }, [conversations, query]);

  const beginRename = (conversation: Conversation) => {
    setEditingId(conversation.id);
    setDraftTitle(conversation.title);
  };

  const finishRename = () => {
    if (editingId) {
      onRenameChat(editingId, draftTitle);
    }

    setEditingId(null);
    setDraftTitle('');
  };

  const body = (
    <aside className="flex h-full w-80 max-w-[86vw] flex-col border-r border-surface-200 bg-surface-100/95 backdrop-blur dark:border-surface-800 dark:bg-surface-950/95">
      <div className="flex h-16 items-center gap-2 px-4">
        <div className="grid h-9 w-9 place-items-center rounded-md bg-brand-500 text-sm font-bold text-white">
          GP
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-surface-900 dark:text-white">GenPy Chat</p>
          <p className="truncate text-xs text-surface-700 dark:text-surface-200">Local coding assistant</p>
        </div>
        <IconButton label="Close sidebar" className="lg:hidden" onClick={onClose}>
          <X size={18} />
        </IconButton>
      </div>

      <div className="space-y-3 px-3 pb-3">
        <button
          type="button"
          onClick={onNewChat}
          className="flex h-10 w-full items-center justify-center gap-2 rounded-md bg-surface-900 px-3 text-sm font-medium text-white transition hover:bg-surface-700 focus:outline-none focus:ring-2 focus:ring-brand-500/40 dark:bg-brand-500 dark:hover:bg-brand-400"
        >
          <Plus size={17} />
          New Chat
        </button>

        <label className="flex h-10 items-center gap-2 rounded-md border border-surface-200 bg-white px-3 text-sm text-surface-700 focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/20 dark:border-surface-800 dark:bg-surface-900 dark:text-surface-200">
          <Search size={16} />
          <input
            ref={searchInputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search chats"
            className="min-w-0 flex-1 bg-transparent text-surface-900 outline-none placeholder:text-surface-700/70 dark:text-white dark:placeholder:text-surface-200/60"
          />
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        <AnimatePresence initial={false}>
          {visibleConversations.map((conversation) => {
            const isActive = conversation.id === activeConversationId;
            const preview = conversation.messages.at(-1)?.content ?? 'No messages yet';

            return (
              <motion.div
                key={conversation.id}
                layout
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                className={`group mb-2 rounded-md border transition ${
                  isActive
                    ? 'border-brand-500/50 bg-white shadow-sm dark:border-brand-400/60 dark:bg-surface-900'
                    : 'border-transparent hover:border-surface-200 hover:bg-white/80 dark:hover:border-surface-800 dark:hover:bg-surface-900/80'
                }`}
              >
                <div
                  onClick={() => {
                    if (editingId !== conversation.id) {
                      onSelectChat(conversation.id);
                    }
                  }}
                  className="flex w-full cursor-pointer gap-3 px-3 py-3 text-left"
                >
                  <MessageSquare className="mt-0.5 shrink-0 text-brand-600 dark:text-brand-400" size={17} />
                  <span className="min-w-0 flex-1">
                    {editingId === conversation.id ? (
                      <input
                        autoFocus
                        value={draftTitle}
                        onChange={(event) => setDraftTitle(event.target.value)}
                        onBlur={finishRename}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') {
                            finishRename();
                          }

                          if (event.key === 'Escape') {
                            setEditingId(null);
                          }
                        }}
                        className="w-full rounded border border-brand-500 bg-white px-2 py-1 text-sm font-medium text-surface-900 outline-none dark:bg-surface-800 dark:text-white"
                      />
                    ) : (
                      <span className="block truncate text-sm font-medium text-surface-900 dark:text-white">
                        {conversation.title}
                      </span>
                    )}
                    <span className="mt-1 block truncate text-xs text-surface-700 dark:text-surface-200">
                      {preview}
                    </span>
                    <span className="mt-2 block text-[11px] uppercase text-surface-700/75 dark:text-surface-200/70">
                      {formatRelativeTime(conversation.updatedAt)}
                    </span>
                  </span>
                </div>

                <div className="flex items-center justify-end gap-1 px-2 pb-2 opacity-100 sm:opacity-0 sm:transition sm:group-hover:opacity-100">
                  <IconButton label="Rename chat" onClick={() => beginRename(conversation)}>
                    <Edit3 size={16} />
                  </IconButton>
                  <IconButton label="Delete chat" onClick={() => onDeleteChat(conversation.id)}>
                    <Trash2 size={16} />
                  </IconButton>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {visibleConversations.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-surface-700 dark:text-surface-200">
            No chats match your search.
          </div>
        )}
      </div>
    </aside>
  );

  return (
    <>
      <div className="hidden h-screen lg:block">{body}</div>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-40 flex lg:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <button
              type="button"
              className="absolute inset-0 bg-surface-950/45"
              aria-label="Close sidebar"
              onClick={onClose}
            />
            <motion.div
              initial={{ x: -340 }}
              animate={{ x: 0 }}
              exit={{ x: -340 }}
              transition={{ type: 'spring', damping: 28, stiffness: 280 }}
              className="relative h-full"
            >
              {body}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
