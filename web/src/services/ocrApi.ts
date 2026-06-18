/** OCR / Agent / 抽出関連 API */
import api from './api';

export const runAgent = async (imageId: string) => {
  const startResponse = await api.post(`/images/${imageId}/agent`);
  const jobId = startResponse.data.jobId;
  return pollAgentJobStatus(jobId);
};

export const pollAgentJobStatus = async (
  jobId: string,
  maxAttempts = 60,
  interval = 2000
): Promise<any> => {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const response = await api.get(`/jobs/${jobId}`);
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

export const getAgentToolsForImage = async (imageId: string) => {
  const response = await api.get(`/images/${imageId}/agent/tools`);
  return response.data;
};

export const getAgentJobByImage = async (imageId: string) => {
  const response = await api.get(`/images/${imageId}/agent`);
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
