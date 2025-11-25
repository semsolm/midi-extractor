// frontend/src/components/ResultDisplay.jsx
import { pop } from '../utils/particleEffect';

import React from 'react'; // React 임포트
import * as api from '../services/api'; // api.js 임포트

export function ResultDisplay({ results, onReset }) {
  // App.jsx로부터 받은 results 객체에서 URL을 추출합니다.
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

  // 지연 후 새 탭 열기 헬퍼 함수(파티클 효과 확보용)
  const handleOpenNewTab = (e, url) => {
    e.preventDefault(); // 1. 링크 바로 이동 방지
    pop(e, "circle");   // 2. 파티클 효과 실행
    
    // 3. 파티클 효과 발생 시간 확보 후 새 탭 열기
    setTimeout(() => {
      window.open(url, '_blank', 'noopener,noreferrer');
    }, 650);
  };

  return (
    <div className="status-container">
      {/* --- 2열 레이아웃: PDF 뷰어(왼쪽) + 다운로드 버튼(오른쪽) --- */}
      <div className="result-layout">
        {/* --- 왼쪽: PDF 뷰어 (크게) --- */}
        <div className="pdf-viewer-section">
          <div id="pdfViewerContainer" className="pdf-viewer-desktop-only">
            <iframe
              id="pdfViewer"
              title="PDF Viewer"
              src={pdfDownloadUrl}
            ></iframe>
          </div>
          
          {/* 모바일용 PDF 보기 버튼 */}
          <a
            href={pdfDownloadUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="button-primary download-link pdf-viewer-mobile-only"
            onClick={(e) => handleOpenNewTab(e, pdfDownloadUrl)} //파티클 효과
          >
            새 탭에서 악보 보기
          </a>
        </div>

        {/* --- 오른쪽: 다운로드 버튼들 --- */}
        <div className="download-section">
          <div className="controls">
            {/* 데스크톱용 PDF 보기 버튼 (새 탭) */}
            <a
              href={pdfDownloadUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="button-primary download-link pdf-viewer-desktop-only"
              onClick={(e) => handleOpenNewTab(e, pdfDownloadUrl)} //파티클 효과
            >
              새 탭에서 악보 보기
            </a>

            {/* PDF 다운로드 버튼 (모든 화면에서 보임) */}
            <a
              href={`${pdfDownloadUrl}?download=true`}
              className="button-primary download-link"
              download
              onClick={(e) => pop(e, "circle")} //파티클 효과
            >
              PDF 악보(.pdf) 다운로드
            </a>

            {/* MIDI 다운로드 버튼 */}
            <a
              href={midiDownloadUrl}
              className="button-primary download-link"
              download
              onClick={(e) => pop(e, "circle")} //파티클 효과
            >
              MIDI 악보(.mid) 다운로드
            </a>
          </div>
        </div>
      </div>

      {/* 완료 메시지를 최하단에 배치 */}
      <div id="statusMessageElement" className="status-success">
        ✅ 변환이 완료되었습니다!
      </div>
      
      {/* '처음으로 돌아가기' 버튼 */}
      <button
        id="resetButton"
        onClick={(e) => {
            pop(e, "square"); // 파티클 효과
            setTimeout(() => {
            onReset();
          }, 700);
        }}
        className="reset-button"
      >
        처음으로 돌아가기
      </button>
    </div>
  );
}