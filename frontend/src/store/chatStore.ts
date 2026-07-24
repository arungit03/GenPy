import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type {
  ChatMessage,
  Conversation,
  GenerationSettings,
  HealthResponse,
  ModelResponse,
  ThemePreference,
} from '../types/chat';
import { estimateTitle } from '../utils/format';
import { createId } from '../utils/id';

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  settings: GenerationSettings;
  theme: ThemePreference;
  health: HealthResponse | null;
  model: ModelResponse | null;
  createConversation: () => string;
  setActiveConversation: (conversationId: string) => void;
  renameConversation: (conversationId: string, title: string) => void;
  deleteConversation: (conversationId: string) => void;
  appendMessage: (conversationId: string, message: ChatMessage) => void;
  updateMessage: (conversationId: string, messageId: string, patch: Partial<ChatMessage>) => void;
  replaceMessages: (conversationId: string, messages: ChatMessage[]) => void;
  updateSettings: (settings: Partial<GenerationSettings>) => void;
  setTheme: (theme: ThemePreference) => void;
  setHealth: (health: HealthResponse | null) => void;
  setModel: (model: ModelResponse | null) => void;
}

const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.7,
  topP: 0.9,
  maxNewTokens: 256,
};

function now(): string {
  return new Date().toISOString();
}

function createConversation(): Conversation {
  const timestamp = now();

  return {
    id: createId('chat'),
    title: 'New chat',
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [],
  };
}

function sortByUpdatedAt(conversations: Conversation[]): Conversation[] {
  return [...conversations].sort(
    (left, right) => new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime(),
  );
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      conversations: [createConversation()],
      activeConversationId: null,
      settings: DEFAULT_SETTINGS,
      theme: 'system',
      health: null,
      model: null,
      createConversation: () => {
        const conversation = createConversation();
        set((state) => ({
          conversations: sortByUpdatedAt([conversation, ...state.conversations]),
          activeConversationId: conversation.id,
        }));
        return conversation.id;
      },
      setActiveConversation: (conversationId) => {
        set({ activeConversationId: conversationId });
      },
      renameConversation: (conversationId, title) => {
        const trimmed = title.trim() || 'Untitled chat';
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? { ...conversation, title: trimmed, updatedAt: now() }
              : conversation,
          ),
        }));
      },
      deleteConversation: (conversationId) => {
        set((state) => {
          const remaining = state.conversations.filter((conversation) => conversation.id !== conversationId);
          const conversations = remaining.length > 0 ? remaining : [createConversation()];
          const activeConversationId =
            state.activeConversationId === conversationId ? conversations[0]?.id ?? null : state.activeConversationId;

          return { conversations, activeConversationId };
        });
      },
      appendMessage: (conversationId, message) => {
        set((state) => ({
          conversations: sortByUpdatedAt(
            state.conversations.map((conversation) => {
              if (conversation.id !== conversationId) {
                return conversation;
              }

              const shouldName =
                conversation.messages.length === 0 &&
                conversation.title === 'New chat' &&
                message.role === 'user';

              return {
                ...conversation,
                title: shouldName ? estimateTitle(message.content) : conversation.title,
                updatedAt: now(),
                messages: [...conversation.messages, message],
              };
            }),
          ),
        }));
      },
      updateMessage: (conversationId, messageId, patch) => {
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  updatedAt: now(),
                  messages: conversation.messages.map((message) =>
                    message.id === messageId ? { ...message, ...patch } : message,
                  ),
                }
              : conversation,
          ),
        }));
      },
      replaceMessages: (conversationId, messages) => {
        set((state) => ({
          conversations: sortByUpdatedAt(
            state.conversations.map((conversation) =>
              conversation.id === conversationId
                ? {
                    ...conversation,
                    updatedAt: now(),
                    messages,
                  }
                : conversation,
            ),
          ),
        }));
      },
      updateSettings: (settings) => {
        set((state) => ({ settings: { ...state.settings, ...settings } }));
      },
      setTheme: (theme) => {
        set({ theme });
      },
      setHealth: (health) => {
        set({ health });
      },
      setModel: (model) => {
        set({ model });
      },
    }),
    {
      name: 'genpy-chat-state',
      version: 1,
      partialize: (state) => ({
        conversations: state.conversations,
        activeConversationId: state.activeConversationId,
        settings: state.settings,
        theme: state.theme,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) {
          return;
        }

        if (state.conversations.length === 0) {
          const conversation = createConversation();
          state.conversations = [conversation];
          state.activeConversationId = conversation.id;
        }

        if (!state.activeConversationId) {
          state.activeConversationId = state.conversations[0]?.id ?? null;
        }
      },
    },
  ),
);

export function makeMessage(role: ChatMessage['role'], content: string, status: ChatMessage['status'] = 'sent'): ChatMessage {
  return {
    id: createId('msg'),
    role,
    content,
    status,
    createdAt: now(),
  };
}

export function getActiveConversation(): Conversation | null {
  const state = useChatStore.getState();
  const activeId = state.activeConversationId ?? state.conversations[0]?.id;
  return state.conversations.find((conversation) => conversation.id === activeId) ?? null;
}
