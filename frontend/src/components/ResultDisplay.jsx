// src/components/ResultDisplay.jsx
import * as api from '../services/api';

export function ResultDisplay({ results, onReset }) {
  // results 객체: { midiUrl, pdfUrl }
  const midiDownloadUrl = api.getFullDownloadUrl(results.midiUrl);
  const pdfDownloadUrl = api.getFullDownloadUrl(results.pdfUrl);

  return (
    <div className="status-container">
      <div id="statusMessageElement" className="status-success">
        ✅ 변환이 완료되었습니다!
      </div>
      
      <div className="controls">
        <a 
          href={midiDownloadUrl} 
          className="button-primary download-link"
          download // 'download' 속성으로 파일 다운로드
        >
          MIDI 악보(.mid) 다운로드
        </a>
        <a 
          href={pdfDownloadUrl} 
          className="button-secondary download-link"
          target="_blank" // PDF는 새 탭에서 열기 (구현 시)
          rel="noopener noreferrer"
        >
          PDF 악보(.pdf) 다운로드
        </a>
      </div>
      
      <button onClick={onReset} className="button-secondary">
        다른 파일 변환하기
      </button>
    </div>
  );
}