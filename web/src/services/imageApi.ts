/** 画像 CRUD API */
import api from './api';

export const deleteImage = async (imageId: string) => {
  return api.delete(`/images/${imageId}`);
};

export const updateVerificationStatus = async (imageId: string, completed: boolean) => {
  return api.post(`/ocr/extract/verification/${imageId}`, {
    verification_completed: completed
  });
};
