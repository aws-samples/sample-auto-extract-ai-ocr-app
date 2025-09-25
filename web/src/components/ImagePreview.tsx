import React, { useRef, useEffect, useState } from 'react';
import { OcrBoundingBox } from '../types/ocr';

interface ImagePreviewProps {
  imageSrc: string;
  boundingBoxes: OcrBoundingBox[];
  selectedIndex: number | null;
  onSelectBox: (index: number) => void;
  onImageLoad: () => void;
  onImageError: (error: any) => void;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({
  imageSrc,
  boundingBoxes,
  selectedIndex,
  onSelectBox,
  onImageLoad,
  onImageError
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const [originalImageWidth, setOriginalImageWidth] = useState(0);
  const [originalImageHeight, setOriginalImageHeight] = useState(0);
  const [resizeCounter, setResizeCounter] = useState(0);

  // 画像がロードされたときのハンドラ
  const handleImageLoad = () => {
    if (imageRef.current) {
      const naturalWidth = imageRef.current.naturalWidth;
      const naturalHeight = imageRef.current.naturalHeight;
      
      setOriginalImageWidth(naturalWidth);
      setOriginalImageHeight(naturalHeight);
      
      updateBoundingBoxes();
    }
    onImageLoad();
  };

  // 画像のエラー時のハンドラ
  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    onImageError(e);
  };

  // バウンディングボックスの更新
  const updateBoundingBoxes = () => {
    if (!imageRef.current || originalImageWidth <= 0) return;
    setResizeCounter(prev => prev + 1);
  };

  // バウンディングボックスのスタイル計算
  const getBoxStyle = (box: OcrBoundingBox) => {
    if (imageRef.current && originalImageWidth > 0 && imageContainerRef.current) {
      // 画像の実際の表示サイズを取得
      const displayedWidth = imageRef.current.clientWidth;
      const displayedHeight = imageRef.current.clientHeight;
      
      // 画像コンテナの位置を取得
      const containerRect = imageContainerRef.current.getBoundingClientRect();
      const imageRect = imageRef.current.getBoundingClientRect();
      
      // スケール計算
      const scaleX = displayedWidth / originalImageWidth;
      const scaleY = displayedHeight / originalImageHeight;
      
      // 画像の左上からの相対位置を計算
      const offsetX = imageRect.left - containerRect.left;
      const offsetY = imageRect.top - containerRect.top;
      
      return {
        position: 'absolute' as const,
        top: `${offsetY + box.top * scaleY}px`,
        left: `${offsetX + box.left * scaleX}px`,
        width: `${box.width * scaleX}px`,
        height: `${box.height * scaleY}px`,
        pointerEvents: 'auto' as const
      };
    }
    return {};
  };

  // リサイズ監視
  useEffect(() => {
    const handleResize = () => {
      updateBoundingBoxes();
    };

    // ResizeObserverの設定
    let resizeObserver: ResizeObserver | null = null;
    if (containerRef.current) {
      resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(containerRef.current);
      
      if (imageRef.current) {
        resizeObserver.observe(imageRef.current);
      }
      
      if (imageContainerRef.current) {
        resizeObserver.observe(imageContainerRef.current);
      }
    }

    // ウィンドウリサイズイベントも監視
    window.addEventListener('resize', handleResize);

    // 初期表示時に少し遅れて強制更新
    const timer = setTimeout(() => {
      updateBoundingBoxes();
    }, 500);

    // 定期的に更新するタイマー（バックアップ）
    const intervalTimer = setInterval(() => {
      updateBoundingBoxes();
    }, 1000);

    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      window.removeEventListener('resize', handleResize);
      clearTimeout(timer);
      clearInterval(intervalTimer);
    };
  }, [originalImageWidth, originalImageHeight]);

  // 選択されたボックスが表示領域内に入るようにスクロール
  useEffect(() => {
    if (selectedIndex !== null && containerRef.current && boundingBoxes[selectedIndex]) {
      const box = boundingBoxes[selectedIndex];
      const container = containerRef.current;
      
      if (imageRef.current && originalImageWidth > 0) {
        // 現在の画像表示サイズを取得
        const scaleX = imageRef.current.clientWidth / originalImageWidth;
        const scaleY = imageRef.current.clientHeight / originalImageHeight;
        
        // スケールされたボックスの中心位置を計算
        const boxCenterX = (box.left + box.width / 2) * scaleX;
        const boxCenterY = (box.top + box.height / 2) * scaleY;
        
        // コンテナの表示領域
        const containerRect = container.getBoundingClientRect();
        
        // 新しいスクロール位置を計算
        const newScrollLeft = boxCenterX - containerRect.width / 2 + container.scrollLeft;
        const newScrollTop = boxCenterY - containerRect.height / 2 + container.scrollTop;
        
        // スムーズにスクロール
        container.scrollTo({
          left: newScrollLeft,
          top: newScrollTop,
          behavior: 'smooth'
        });
      }
    }
  }, [selectedIndex, boundingBoxes, originalImageWidth, originalImageHeight]);

  return (
    <div ref={containerRef} className="w-full h-full overflow-auto">
      {!imageSrc ? (
        <div className="flex justify-center items-center h-full text-gray-500">
          画像が読み込めませんでした
        </div>
      ) : (
        <div ref={imageContainerRef} className="relative w-full h-full flex items-start justify-center p-2">
          <img
            ref={imageRef}
            src={imageSrc}
            alt="OCR画像"
            className="w-full h-auto object-contain"
            crossOrigin="anonymous"  // CORS対応
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
          
          {/* バウンディングボックスのオーバーレイ */}
          {boundingBoxes.map((box, index) => (
            <div
              key={`box-${index}-${resizeCounter}`}
              className={`absolute cursor-pointer ${
                selectedIndex === index 
                  ? 'border-2 border-rose-500 bg-rose-100 bg-opacity-25' 
                  : 'border border-blue-400 bg-blue-100 bg-opacity-20'
              }`}
              style={getBoxStyle(box)}
              onClick={() => onSelectBox(index)}
              title={`テキスト: ${box.text}`}
            ></div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImagePreview;
