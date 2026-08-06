/** OCR / Agent / 抽出関連 API */
import api from './api';
import { AgentJobResponse, AgentRunResult, ToolsResponse } from '../types/agent';

export const runAgent = async (imageId: string): Promise<AgentRunResult> => {
  const startResponse = await api.post<{ jobId: string }>(`/images/${imageId}/agent`);
  const jobId = startResponse.data.jobId;
  return pollAgentJobStatus(jobId);
};

export const pollAgentJobStatus = async (
  jobId: string,
  maxAttempts = 60,
  interval = 2000
): Promise<AgentRunResult> => {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const response = await api.get<AgentJobResponse>(`/jobs/${jobId}`);
    const { status, suggestions, error } = response.data;

    if (status === 'completed' || status === 'skipped') {
      return { status: 'success', suggestions: suggestions || [] };
    }

    if (status === 'failed') {
      throw new Error(error || 'Agent processing failed');
    }

    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error('Agent processing timed out');
};

export const getAgentToolsForImage = async (imageId: string): Promise<ToolsResponse> => {
  const response = await api.get<ToolsResponse>(`/images/${imageId}/agent/tools`);
  return response.data;
};

export const getAgentJobByImage = async (imageId: string): Promise<AgentJobResponse> => {
  const response = await api.get<AgentJobResponse>(`/images/${imageId}/agent`);
  return response.data;
};

export const updateSuggestionStatus = async (
  imageId: string,
  suggestionIndex: number,
  status: 'accepted' | 'rejected'
): Promise<{ ok: boolean; pending_count: number }> => {
  const response = await api.patch(`/images/${imageId}/agent/suggestions/${suggestionIndex}`, { status });
  return response.data;
};
