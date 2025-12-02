// frontend/src/components/ResultDisplay.jsx
import { pop } from '../utils/particleEffect';
import React from 'react';
import * as api from '../services/api';

export function ResultDisplay({ results, onReset }) {
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

  // 지연 후 새 탭 열기 (파티클 효과 확보용)
  const handleOpenNewTab = (e, url) => {
    e.preventDefault();
    pop(e, "circle");
    setTimeout(() => {
      window.open(url, '_blank', 'noopener,noreferrer');
    }, 650);
  };

  return (
    <div className="result-container">
      {/* 완료 메시지 - 상단에 배치 */}
      <p className="result-complete-text">✅ 변환이 완료되었습니다!</p>

      {/* --- 2열 레이아웃: PDF 뷰어(왼쪽) + 다운로드 버튼(오른쪽) --- */}
      <div className="result-layout">

        {/* --- 왼쪽: PDF 뷰어 --- */}
        <div className="pdf-viewer-section">
          <div className="pdf-viewer-wrapper">
            <iframe
              className="pdf-viewer-frame"
              title="PDF Viewer"
              src={pdfDownloadUrl}
            />
          </div>
        </div>

        {/* --- 오른쪽: 다운로드 버튼들 --- */}
        <div className="download-section">
          {/* 새 탭에서 보기 */}
          <a
            href={pdfDownloadUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-action-primary"
            onClick={(e) => handleOpenNewTab(e, pdfDownloadUrl)}
          >
            <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/>
              <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            새 탭에서 악보 보기
          </a>

          {/* PDF 다운로드 */}
          <a
            href={`${pdfDownloadUrl}?download=true`}
            className="btn-action-secondary"
            download
            onClick={(e) => pop(e, "circle")}
          >
            <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="12" y1="18" x2="12" y2="12"/>
              <polyline points="9 15 12 18 15 15"/>
            </svg>
            PDF 악보 다운로드
          </a>

          {/* MIDI 다운로드 */}
          <a
            href={midiDownloadUrl}
            className="btn-action-secondary"
            download
            onClick={(e) => pop(e, "circle")}
          >
            <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18V5l12-2v13"/>
              <circle cx="6" cy="18" r="3"/>
              <circle cx="18" cy="16" r="3"/>
            </svg>
            MIDI 파일 다운로드
          </a>
        </div>
      </div>

      {/* 모바일 전용: 버튼들 */}
      <div className="mobile-buttons">
        <a
          href={pdfDownloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-action-primary"
          onClick={(e) => handleOpenNewTab(e, pdfDownloadUrl)}
        >
          새 탭에서 악보 보기
        </a>
        <a
          href={`${pdfDownloadUrl}?download=true`}
          className="btn-action-secondary"
          download
          onClick={(e) => pop(e, "circle")}
        >
          PDF 다운로드
        </a>
        <a
          href={midiDownloadUrl}
          className="btn-action-secondary"
          download
          onClick={(e) => pop(e, "circle")}
        >
          MIDI 다운로드
        </a>
      </div>

      {/* 처음으로 돌아가기 버튼 */}
      <button
        className="btn-reset-ghost"
        onClick={(e) => {
          pop(e, "square");
          setTimeout(() => onReset(), 700);
        }}
      >
        <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '16px', height: '16px' }}>
          <polyline points="1 4 1 10 7 10"/>
          <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
        </svg>
        처음으로 돌아가기
      </button>
    </div>
  );
}