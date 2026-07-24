import axios, { AxiosError } from 'axios';

import type {
  ApiChatMessage,
  GenerationResponse,
  GenerationSettings,
  HealthResponse,
  ModelResponse,
} from '../types/chat';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 120_000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health', { signal });
  return data;
}

export async function getModel(signal?: AbortSignal): Promise<ModelResponse> {
  const { data } = await api.get<ModelResponse>('/model', { signal });
  return data;
}

export async function chat(
  messages: ApiChatMessage[],
  settings: GenerationSettings,
  signal?: AbortSignal,
): Promise<GenerationResponse> {
  const { data } = await api.post<GenerationResponse>(
    '/chat',
    {
      messages,
      max_new_tokens: settings.maxNewTokens,
      temperature: settings.temperature,
      top_p: settings.topP,
    },
    { signal },
  );
  return data;
}

export async function generate(
  prompt: string,
  settings: GenerationSettings,
  signal?: AbortSignal,
): Promise<GenerationResponse> {
  const { data } = await api.post<GenerationResponse>(
    '/generate',
    {
      prompt,
      max_new_tokens: settings.maxNewTokens,
      temperature: settings.temperature,
      top_p: settings.topP,
    },
    { signal },
  );
  return data;
}

export function isCanceled(error: unknown): boolean {
  return axios.isCancel(error) || (error instanceof AxiosError && error.code === 'ERR_CANCELED');
}

export function toApiErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;

    if (typeof detail === 'string') {
      return detail;
    }

    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) => {
          const location = Array.isArray(item.loc) ? item.loc.join('.') : 'request';
          return `${location}: ${item.msg ?? 'Invalid value'}`;
        })
        .join('\n');
    }

    if (error.response?.status) {
      return `Request failed with status ${error.response.status}.`;
    }

    if (error.code === 'ECONNABORTED') {
      return 'The model took too long to respond.';
    }
  }

  return error instanceof Error ? error.message : 'Something went wrong while contacting GenPy.';
}
