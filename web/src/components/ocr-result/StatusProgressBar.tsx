import { Check, X, Loader2 } from 'lucide-react';

type StepStatus = 'completed' | 'processing' | 'failed' | 'idle';

interface Step {
  label: string;
  status: StepStatus;
  clickable?: boolean;
  onClick?: () => void;
}

interface StatusProgressBarProps {
  steps: Step[];
}

export default function StatusProgressBar({ steps }: StatusProgressBarProps) {
  const getStepStyle = (status: StepStatus) => {
    switch (status) {
      case 'completed':
        return { circle: 'bg-success text-white', line: 'bg-success' };
      case 'processing':
        return { circle: 'bg-info text-white status-circle-processing', line: 'bg-neutral-200' };
      case 'failed':
        return { circle: 'bg-danger text-white', line: 'bg-neutral-200' };
      default:
        return { circle: 'bg-neutral-200 text-neutral-400', line: 'bg-neutral-200' };
    }
  };

  // 線: 両方 completed→緑、completed→processing→shimmer(流入)、それ以外→灰
  const getLineClass = (currentStatus: StepStatus, nextStatus: StepStatus | undefined) => {
    if (currentStatus === 'completed' && nextStatus === 'completed') return 'bg-success';
    if (currentStatus === 'completed' && nextStatus === 'processing') return 'status-line-processing';
    return 'bg-neutral-200';
  };

  const renderIcon = (status: StepStatus) => {
    switch (status) {
      case 'completed':
        return <Check size={14} />;
      case 'processing':
        return <Loader2 size={14} className="animate-spin" />;
      case 'failed':
        return <X size={14} />;
      default:
        return <span className="w-2 h-2 rounded-full bg-neutral-400" />;
    }
  };

  return (
    <div className="flex items-center w-full">
      {steps.map((step, i) => {
        const style = getStepStyle(step.status);
        const isLast = i === steps.length - 1;
        const lineClass = isLast ? '' : getLineClass(step.status, steps[i + 1]?.status);
        return (
          <div key={i} className={`flex items-center ${isLast ? '' : 'flex-1'}`}>
            <div
              className={`flex items-center gap-1.5 ${step.clickable ? 'cursor-pointer hover:opacity-80' : ''}`}
              onClick={step.clickable ? step.onClick : undefined}
            >
              <div className={`w-6 h-6 rounded-full flex items-center justify-center ${style.circle}`}>
                {renderIcon(step.status)}
              </div>
              <span className="text-xs font-medium text-neutral-700 whitespace-nowrap">{step.label}</span>
            </div>
            {!isLast && (
              <div className={`flex-1 h-0.5 mx-2 ${lineClass}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
