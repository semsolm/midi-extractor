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
        ✅ 변환이 완료되었습니다!
      </div>

      {/* --- Start: PDF 뷰어 --- */}
      {/* 데스크톱에서는 iframe 뷰어를 보여주고,
        모바일에서는 CSS(App.css)에 의해 이 뷰어가 숨겨집니다.
      */}
      <div id="pdfViewerContainer" className="pdf-viewer-desktop-only" style={{ display: 'block' }}>
        <iframe
          id="pdfViewer"
          title="PDF Viewer"
          src={pdfDownloadUrl}
        ></iframe>
      </div>
      {/* --- End: PDF 뷰어 --- */}

      <div className="controls" style={{ marginTop: '20px', gap: '12px' }}>
        {/* --- 모바일용 PDF 보기 버튼 --- */}
        {/* 이 버튼은 모바일에서만 보입니다.
          target="_blank"는 새 탭에서 PDF를 엽니다.
        */}
        <a
          href={pdfDownloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="button-primary pdf-viewer-mobile-only"
        >
          PDF 악보 보기 (새 탭)
        </a>

        {/* MIDI 다운로드 버튼 (항상 보임) */}
        <a
          href={midiDownloadUrl}
          className="button-primary download-link"
          download
        >
          MIDI 악보(.mid) 다운로드
        </a>

        {/* PDF 다운로드 버튼 (데스크톱에서만 보임) */}
        <a
          href={pdfDownloadUrl}
          className="button-primary download-link pdf-viewer-desktop-only"
          download
          style={{ flexGrow: 1, maxWidth: '240px', backgroundColor: '#3b82f6' }}
        >
          PDF 악보(.pdf) 다운로드
        </a>
      </div>

      {/* '처음으로 돌아가기' 버튼 */}
      <button
        id="resetButton"
        onClick={onReset}
        style={{ display: 'block', maxWidth: '490px', margin: '10px auto 0 auto' }}
      >
        처음으로 돌아가기
      </button>
    </div>
  );
}