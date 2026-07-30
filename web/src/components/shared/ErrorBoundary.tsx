import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  // 例外時に表示する内容。未指定なら既定のメッセージを出す。
  fallback?: React.ReactNode;
  // ここに渡した値のいずれかが変わると、エラー状態を解除して再描画を試みる。
  // 例: 表示対象データを渡すと、データが更新されたときに自動復帰する。
  resetKeys?: unknown[];
}

interface ErrorBoundaryState {
  hasError: boolean;
}

// 子の描画中に投げられた例外を捕捉し、画面全体のクラッシュを防ぐ。
// 捕捉範囲を限定して使うことで、エラー箇所だけをフォールバック表示にし、
// 周囲の UI は生存させる。イベントハンドラ内の例外は React の仕様上捕捉しない。
class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // 握りつぶすと将来の不具合が検知不能になるため、必ずログに残す。
    console.error('ErrorBoundary caught an error:', error, info.componentStack);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    // エラー中に resetKeys が変化したら（データ更新・再抽出など）エラー状態を解除する。
    // これがないと、原因データが直っても fallback に貼り付いたままになる。
    if (!this.state.hasError) return;
    const prev = prevProps.resetKeys ?? [];
    const next = this.props.resetKeys ?? [];
    if (prev.length !== next.length || next.some((k, i) => !Object.is(k, prev[i]))) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="p-4 text-sm text-neutral-600 bg-neutral-50 border border-neutral-200 rounded">
            表示中にエラーが発生しました。
          </div>
        )
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
