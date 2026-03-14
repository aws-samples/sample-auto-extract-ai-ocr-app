import React from "react";
import { Modal } from "./ui/Modal";
import { Button } from "./ui/Button";

interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
}

const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = "OK",
  cancelText = "キャンセル",
}) => {
  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="max-w-md w-full">
      <div className="p-6">
        <h3 className="text-lg font-semibold text-neutral-900 mb-4">{title}</h3>
        <p className="text-neutral-600 whitespace-pre-line mb-6">{message}</p>
        <div className="flex justify-end gap-3">
          <Button variant="secondary" onClick={onClose}>{cancelText}</Button>
          <Button variant="primary" onClick={handleConfirm}>{confirmText}</Button>
        </div>
      </div>
    </Modal>
  );
};

export default ConfirmModal;
