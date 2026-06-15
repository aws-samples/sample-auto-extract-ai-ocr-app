/** OCR / Agent / 抽出関連 API */
import api from './api';

export const runAgent = async (imageId: string) => {
  const startResponse = await api.post(`/ocr/agent/${imageId}`);
  const jobId = startResponse.data.jobId;
  return pollAgentJobStatus(jobId);
};

export const pollAgentJobStatus = async (
  jobId: string,
  maxAttempts = 60,
  interval = 2000
): Promise<any> => {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const response = await api.get(`/ocr/agent/status/${jobId}`);
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

export const getAgentTools = async (imageId?: string) => {
  const params = imageId ? { image_id: imageId } : {};
  const response = await api.get('/ocr/agent/tools', { params });
  return response.data;
};

export const getAgentJobByImage = async (imageId: string) => {
  const response = await api.get(`/ocr/agent/image/${imageId}`);
  return response.data;
};

export const updateSuggestionStatus = async (
  imageId: string,
  suggestionIndex: number,
  status: 'accepted' | 'rejected'
): Promise<{ ok: boolean; pending_count: number }> => {
  const response = await api.patch(`/ocr/agent/image/${imageId}/suggestions/${suggestionIndex}`, { status });
  return response.data;
};
