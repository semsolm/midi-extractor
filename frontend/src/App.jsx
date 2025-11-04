import React, { useState } from 'react';
import './App.css'; // 스타일 파일 import
import * as api from './services/api'; // Api 내에 UI 코드 없음
import { UploadForm } from './components/UploadForm.jsx';
import { StatusTracker } from './components/StatusTracker.jsx';
import { ResultDisplay } from './components/ResultDisplay.jsx';
import drummerImg from './assets/drummer2.png'; // 이미지 파일 import

function App() {
  // 'idle', 'uploading', 'processing', 'completed', 'error'
  const [uiState, setUiState] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [jobResult, setJobResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // 1. 업로드 폼에서 '변환 시작' 버튼 클릭 시
  const handleUpload = async (file) => {
    setUiState('uploading');
    setErrorMessage('');
    try {
      const { jobId } = await api.uploadAudioFile(file);
      setJobId(jobId);
      setUiState('processing'); // 업로드 성공 -> '처리 중' 상태로 변경
    } catch (error) {
      setErrorMessage(error.message);
      setUiState('error');
    }
  };

  // 2. StatusTracker가 'completed' 상태를 감지했을 때
  const handleProcessingComplete = (results) => {
    setJobResult(results); // { midiUrl, pdfUrl }
    setUiState('completed');
  };

  // 3. StatusTracker가 'error' 상태를 감지했을 때
  const handleProcessingError = (message) => {
    setErrorMessage(message);
    setUiState('error');
  };

  // 4. '다시하기' 버튼 클릭 시
  const handleReset = () => {
    setUiState('idle');
    setJobId(null);
    setJobResult(null);
    setErrorMessage('');
  };

  // UI 상태에 따라 다른 컴포넌트를 렌더링
  const renderContent = () => {
    switch (uiState) {
      case 'idle':
      case 'uploading':
        return (
          <UploadForm
            onUpload={handleUpload}
            isLoading={uiState === 'uploading'}
          />
        );

      case 'processing':
        return (
          <StatusTracker
            jobId={jobId}
            onComplete={handleProcessingComplete}
            onError={handleProcessingError}
          />
        );

      case 'completed':
        return (
          <ResultDisplay
            results={jobResult}
            onReset={handleReset}
          />
        );

      case 'error':
        return (
          <div className="status-container">
            <div id="statusMessageElement" className="status-error">
              {errorMessage}
            </div>
            <button onClick={handleReset} className="button-primary">
              다시 시도
            </button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <>
      <div className="container">
        {/* 'logo-above-title' 이미지를 제거하고, <h1> 안에 이미지 재삽입 */}
        <h1>
          <img src={drummerImg} alt="드럼 아이콘" className="title-icon" />
          드럼 사운드 자동 분류 및 악보 생성 시스템
        </h1>
        <p className="subtitle">파일 전송 후 변환 완료 시 다운로드가 가능합니다.</p>

        {renderContent()}
      </div>

      {/* --- [신규] 하단 푸터 추가 --- */}
      <footer className="app-footer">
        <p>
          <a href="https://github.com/semsolm/midi-extractor/blob/main/readme.md" target="_blank" rel="noopener noreferrer">개인정보처리방침 </a> |
          <a href="https://github.com/semsolm/midi-extractor/issues" target="_blank" rel="noopener noreferrer">오류/건의</a>
        </p>

        <p>Copyright © 2025. Team 경로당. All Rights Reserved.</p>
        <p>
          본 시스템은 [안양대학교 캡스톤 디자인 수업] 의 팀 프로젝트로 제작되었습니다.
        </p>
        <p>
          본 시스템은 학습 및 비영리 목적으로만 무료로 사용할 수 있습니다.<br />
          생성된 악보의 정확성을 보장하지 않으며, 사용으로 인한 법적 책임을 지지 않습니다.
        </p>
      </footer>
    </>
  );
}

export default App;
