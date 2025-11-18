// frontend/src/components/ResultDisplay.jsx

import React from 'react'; // React 임포트
import * as api from '../services/api'; // api.js 임포트

export function ResultDisplay({ results, onReset }) {
  // App.jsx로부터 받은 results 객체에서 URL을 추출합니다.
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

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
            >
              새 탭에서 악보 보기
            </a>

            {/* PDF 다운로드 버튼 (모든 화면에서 보임) */}
            <a
              href={pdfDownloadUrl}
              className="button-primary download-link"
              download
            >
              PDF 악보(.pdf) 다운로드
            </a>

            {/* MIDI 다운로드 버튼 */}
            <a
              href={midiDownloadUrl}
              className="button-primary download-link"
              download
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
        onClick={onReset}
        className="reset-button"
      >
        처음으로 돌아가기
      </button>
    </div>
  );
}