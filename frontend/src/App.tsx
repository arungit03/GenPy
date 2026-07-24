import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { chat, getHealth, getModel, isCanceled, toApiErrorMessage } from './api/client';
import { Composer } from './components/Composer';
import { MessageList } from './components/MessageList';
import { ModelPanel } from './components/ModelPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { getActiveConversation, makeMessage, useChatStore } from './store/chatStore';
import type { ChatMessage, ThemePreference } from './types/chat';

function App() {
  const conversations = useChatStore((state) => state.conversations);
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const settings = useChatStore((state) => state.settings);
  const theme = useChatStore((state) => state.theme);
  const health = useChatStore((state) => state.health);
  const model = useChatStore((state) => state.model);
  const createConversation = useChatStore((state) => state.createConversation);
  const setActiveConversation = useChatStore((state) => state.setActiveConversation);
  const renameConversation = useChatStore((state) => state.renameConversation);
  const deleteConversation = useChatStore((state) => state.deleteConversation);
  const appendMessage = useChatStore((state) => state.appendMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);
  const replaceMessages = useChatStore((state) => state.replaceMessages);
  const updateSettings = useChatStore((state) => state.updateSettings);
  const setTheme = useChatStore((state) => state.setTheme);
  const setHealth = useChatStore((state) => state.setHealth);
  const setModel = useChatStore((state) => state.setModel);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [statusLoading, setStatusLoading] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const activeConversation = useMemo(() => {
    const activeId = activeConversationId ?? conversations[0]?.id;
    return conversations.find((conversation) => conversation.id === activeId) ?? conversations[0] ?? null;
  }, [activeConversationId, conversations]);

  useEffect(() => {
    if (!activeConversationId && conversations[0]) {
      setActiveConversation(conversations[0].id);
    }
  }, [activeConversationId, conversations, setActiveConversation]);

  useEffect(() => {
    const applyTheme = () => {
      const shouldUseDark =
        theme === 'dark' ||
        (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

      document.documentElement.classList.toggle('dark', shouldUseDark);
      setIsDark(shouldUseDark);
    };

    applyTheme();

    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', applyTheme);
    return () => media.removeEventListener('change', applyTheme);
  }, [theme]);

  const loadStatus = useCallback(
    async (signal?: AbortSignal) => {
      setStatusLoading(true);
      setStatusError(null);

      try {
        const [healthResponse, modelResponse] = await Promise.all([getHealth(signal), getModel(signal)]);
        setHealth(healthResponse);
        setModel(modelResponse);
      } catch (error) {
        if (!isCanceled(error)) {
          setHealth(null);
          setModel(null);
          setStatusError(toApiErrorMessage(error));
        }
      } finally {
        setStatusLoading(false);
      }
    },
    [setHealth, setModel],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadStatus(controller.signal);
    return () => controller.abort();
  }, [loadStatus]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const modifier = event.metaKey || event.ctrlKey;

      if (modifier && event.key.toLowerCase() === 'n') {
        event.preventDefault();
        createConversation();
        setSidebarOpen(false);
      }

      if (modifier && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setSidebarOpen(true);
        window.setTimeout(() => searchInputRef.current?.focus(), 50);
      }

      if (modifier && event.key === ',') {
        event.preventDefault();
        setSettingsOpen(true);
      }

      if (event.key === 'Escape') {
        setSidebarOpen(false);
        setSettingsOpen(false);
        setModelOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [createConversation]);

  const requestAssistantResponse = useCallback(
    async (conversationId: string, messages: ChatMessage[]) => {
      const assistantMessage = makeMessage('assistant', '', 'pending');
      const controller = new AbortController();
      controllerRef.current = controller;
      setIsGenerating(true);
      appendMessage(conversationId, assistantMessage);

      try {
        const response = await chat(
          messages
            .filter((message) => message.content.trim() && message.status !== 'error' && message.status !== 'pending')
            .map((message) => ({ role: message.role, content: message.content })),
          useChatStore.getState().settings,
          controller.signal,
        );

        updateMessage(conversationId, assistantMessage.id, {
          content: response.generated_text,
          status: 'sent',
          metrics:
            response.tokens_generated !== undefined &&
            response.generation_time !== undefined &&
            response.tokens_per_second !== undefined
              ? {
                  tokensGenerated: response.tokens_generated,
                  generationTime: response.generation_time,
                  tokensPerSecond: response.tokens_per_second,
                }
              : undefined,
        });
      } catch (error) {
        if (isCanceled(error)) {
          updateMessage(conversationId, assistantMessage.id, {
            content: 'Generation stopped.',
            status: 'stopped',
          });
        } else {
          updateMessage(conversationId, assistantMessage.id, {
            content: toApiErrorMessage(error),
            status: 'error',
          });
        }
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
        setIsGenerating(false);
      }
    },
    [appendMessage, updateMessage],
  );

  const handleSubmit = (content: string) => {
    let conversation = getActiveConversation();
    let conversationId = conversation?.id;

    if (!conversationId) {
      conversationId = createConversation();
      conversation = useChatStore
        .getState()
        .conversations.find((candidate) => candidate.id === conversationId) ?? null;
    }

    const userMessage = makeMessage('user', content);
    const nextMessages = [...(conversation?.messages ?? []), userMessage];
    appendMessage(conversationId, userMessage);
    void requestAssistantResponse(conversationId, nextMessages);
  };

  const handleRetry = (messageId: string) => {
    if (!activeConversation || isGenerating) {
      return;
    }

    const messageIndex = activeConversation.messages.findIndex((message) => message.id === messageId);
    if (messageIndex < 0) {
      return;
    }

    const retainedMessages = activeConversation.messages.slice(0, messageIndex);
    if (!retainedMessages.some((message) => message.role === 'user')) {
      return;
    }

    replaceMessages(activeConversation.id, retainedMessages);
    void requestAssistantResponse(activeConversation.id, retainedMessages);
  };

  const handleRegenerate = () => {
    if (!activeConversation || isGenerating) {
      return;
    }

    const lastAssistantIndex = [...activeConversation.messages]
      .map((message, index) => ({ message, index }))
      .reverse()
      .find(({ message }) => message.role === 'assistant')?.index;

    const retainedMessages =
      lastAssistantIndex === undefined
        ? activeConversation.messages
        : activeConversation.messages.slice(0, lastAssistantIndex);

    if (!retainedMessages.some((message) => message.role === 'user')) {
      return;
    }

    replaceMessages(activeConversation.id, retainedMessages);
    void requestAssistantResponse(activeConversation.id, retainedMessages);
  };

  const handleStop = () => {
    controllerRef.current?.abort();
  };

  const handleNewChat = () => {
    createConversation();
    setSidebarOpen(false);
  };

  const handleSelectChat = (conversationId: string) => {
    setActiveConversation(conversationId);
    setSidebarOpen(false);
  };

  const canRegenerate =
    !!activeConversation?.messages.some((message) => message.role === 'user') &&
    activeConversation.messages.at(-1)?.status !== 'pending';

  return (
    <div className="flex h-screen overflow-hidden bg-surface-50 text-surface-950 dark:bg-surface-950 dark:text-white">
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversation?.id ?? null}
        isOpen={sidebarOpen}
        searchInputRef={searchInputRef}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onRenameChat={renameConversation}
        onDeleteChat={deleteConversation}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar
          health={health}
          model={model}
          isGenerating={isGenerating}
          canRegenerate={canRegenerate}
          onOpenSidebar={() => setSidebarOpen(true)}
          onRefreshStatus={() => void loadStatus()}
          onRegenerate={handleRegenerate}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenModel={() => setModelOpen(true)}
        />

        {statusError && (
          <div className="border-b border-ember-500/25 bg-ember-500/10 px-4 py-3 text-sm text-surface-900 dark:text-surface-100 sm:px-6">
            <div className="mx-auto flex max-w-4xl items-center justify-between gap-3">
              <span className="min-w-0 whitespace-pre-wrap">{statusError}</span>
              <button
                type="button"
                onClick={() => void loadStatus()}
                className="shrink-0 rounded-md bg-ember-500 px-3 py-2 text-sm font-medium text-white"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        <section className="min-h-0 flex-1 overflow-y-auto">
          <MessageList
            messages={activeConversation?.messages ?? []}
            isDark={isDark}
            onRetry={handleRetry}
          />
        </section>

        <Composer
          disabled={statusLoading || health?.model_loaded === false}
          isGenerating={isGenerating}
          onSubmit={handleSubmit}
          onStop={handleStop}
        />
      </main>

      <SettingsPanel
        open={settingsOpen}
        settings={settings}
        theme={theme}
        onUpdateSettings={updateSettings}
        onThemeChange={(nextTheme: ThemePreference) => setTheme(nextTheme)}
        onClose={() => setSettingsOpen(false)}
      />

      <ModelPanel
        open={modelOpen}
        health={health}
        model={model}
        loading={statusLoading}
        error={statusError}
        onRefresh={() => void loadStatus()}
        onClose={() => setModelOpen(false)}
      />
    </div>
  );
}

export default App;
