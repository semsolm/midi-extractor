// src/components/ResultDisplay.jsx
import * as api from '../services/api';
import React from 'react';

export function ResultDisplay({ results, onReset }) {
  // results 객체: { midiUrl, pdfUrl }
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

  return (
    <div className="status-container">
      <div id="statusMessageElement" className="status-success">
        ✅ 변환이 완료되었습니다!
      </div>
      
      {/* --- Start: PDF 뷰어 (index_test.html에서 가져옴) --- */}
      <div id="pdfViewerContainer" style={{ display: 'block' }}>
        <iframe
          id="pdfViewer"
          title="PDF Viewer"
          src={pdfDownloadUrl} // 백엔드에서 받은 PDF 경로를 src로 지정
        ></iframe>
      </div>
      {/* --- End: PDF 뷰어 --- */}

      <div className="controls" style={{ marginTop: '20px' }} >
        <a 
          href={midiDownloadUrl} 
          className="button-primary download-link"
          download // 'download' 속성으로 파일 다운로드
        >
          MIDI 악보(.mid) 다운로드
        </a>
      </div>
      
      {/* '처음으로 돌아가기' 버튼 (index_test.html 스타일 적용) */}
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