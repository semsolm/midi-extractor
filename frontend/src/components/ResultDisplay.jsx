// frontend/src/components/ResultDisplay.jsx

import React from 'react'; // React 임포트
import * as api from '../services/api'; // api.js 임포트

export function ResultDisplay({ results, onReset }) {
  // App.jsx로부터 받은 results 객체에서 URL을 추출합니다.
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

  return (
    <div className="status-container">
      <div id="statusMessageElement" className="status-success">
        변환 완료!
      </div>

      {/* --- Start: PDF 뷰어 --- */}
      {/* 이 iframe이 "pretty_print" 상자를 대체합니다.
        src={pdfDownloadUrl}는 백엔드 API (routes.py)의
        /download/pdf/<job_id> 경로를 호출합니다.
      */}
      <div id="pdfViewerContainer" style={{ display: 'block' }}>
        <iframe
          id="pdfViewer"
          title="PDF Viewer"
          src={pdfDownloadUrl}
        ></iframe>
      </div>
      {/* --- End: PDF 뷰어 --- */}

      <div className="controls" style={{ marginTop: '20px' }}>
        <a
          href={midiDownloadUrl}
          className="button-primary download-link"
          download
        >
          MIDI 악보(.mid) 다운로드
        </a>
        <a
          href={pdfDownloadUrl}
          className="button-primary download-link"
          download
          style={{ flexGrow: 1, maxWidth: '240px', backgroundColor: '#3b82f6' }} // 다른 색상으로 구분
        >
          PDF 악보(.pdf) 다운로드
        </a>
      </div>

      {/* '처음으로 돌아가기' 버튼 */}
      <button
        id="resetButton"
        onClick={onReset}
        style={{ display: 'block', maxWidth: '490px', margin: '30px auto 0 auto' }}
      >
        처음으로 돌아가기
      </button>
    </div>
  );
}