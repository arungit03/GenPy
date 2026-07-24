export type ChatRole = 'system' | 'user' | 'assistant';

export type MessageStatus = 'sent' | 'pending' | 'error' | 'stopped';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  status: MessageStatus;
  metrics?: GenerationMetrics;
}

export interface GenerationMetrics {
  tokensGenerated: number;
  generationTime: number;
  tokensPerSecond: number;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

export interface GenerationSettings {
  temperature: number;
  topP: number;
  maxNewTokens: number;
}

export type ThemePreference = 'light' | 'dark' | 'system';

export interface HealthResponse {
  status: 'healthy';
  device: string;
  model_loaded: boolean;
}

export interface ModelResponse {
  model_name: string;
  parameter_count: number;
  checkpoint_path: string;
  quantization: string | null;
  lora_enabled: boolean;
  lora_adapter: string | null;
  device: string;
  tokenizer_path: string;
  context_length: number;
  vocabulary_size: number;
  loaded_at: string;
}

export interface GenerationResponse {
  generated_text: string;
  tokens_generated?: number;
  generation_time?: number;
  tokens_per_second?: number;
}

export interface ApiChatMessage {
  role: ChatRole;
  content: string;
}
