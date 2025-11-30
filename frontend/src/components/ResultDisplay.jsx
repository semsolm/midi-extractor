import React from 'react';
import * as api from '../services/api';

export function ResultDisplay({ results, onReset }) {
  // URL 추출
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

  return (
    <div className="result-container">
      {/* 완료 메시지 섹션 */}
      <div style={{ marginBottom: '15px', textAlign: 'center' }}>
        <h3 style={{ fontSize: '1.3rem', margin: '0 0 8px 0', color: '#0F172A' }}>
          ✅ 변환이 완료되었습니다!
        </h3>
        <p style={{ color: '#64748B', fontSize: '0.9rem', margin: 0 }}>
          AI가 분석한 결과물을 아래에서 확인하세요.
        </p>
      </div>

      {/* --- PDF 뷰어 (데스크톱에서만 보임) --- */}
      <div className="pdf-viewer-container">
        <iframe
          title="PDF Score Viewer"
          src={pdfDownloadUrl}
        ></iframe>
      </div>

      {/* 🎛️ 액션 버튼 */}
      <div className="action-grid">

        {/* PDF 악보 보기 (새 탭) - 항상 표시 */}
        <a
          href={pdfDownloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-action-primary"
        >
          <svg className="btn-icon" viewBox="0 0 24 24">
            <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
          </svg>
          PDF 악보 보기
        </a>

        {/* 다운로드 버튼들 (가로 배치) */}
        <div className="action-grid-row">
          {/* MIDI 다운로드 */}
          <a
            href={midiDownloadUrl}
            download
            className="btn-action-secondary"
          >
            <svg className="btn-icon" viewBox="0 0 24 24">
              <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
            MIDI 다운로드
          </a>

          {/* PDF 다운로드 */}
          <a
            href={pdfDownloadUrl}
            download
            className="btn-action-secondary"
          >
            <svg className="btn-icon" viewBox="0 0 24 24">
              <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
            PDF 다운로드
          </a>
        </div>
      </div>

      {/* 🔄 처음으로 돌아가기 */}
      <button onClick={onReset} className="btn-reset-ghost">
        <svg className="btn-icon" viewBox="0 0 24 24" style={{width: '16px', height:'16px'}}>
          <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
        </svg>
        처음으로
      </button>
    </div>
  );
}