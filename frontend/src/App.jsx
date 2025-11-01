// src/App.js
import React, { useState } from 'react';
import './App.css'; // 스타일 파일 import
import * as api from './services/api'; // Api 내에 UI 코드 없음
import { UploadForm } from './components/UploadForm.jsx';
import { StatusTracker } from './components/StatusTracker.jsx';
import { ResultDisplay } from './components/ResultDisplay.jsx';
import drummerImg from './assets/drummer.png'; // 이미지 파일 import

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
    <div className="container">
      {/* drummer.png는 frontend/assets/ 폴더 내에 있어야 합니다. */}
      <img src={drummerImg} alt="드럼 치는 사람" /> 
      <h1>드럼 사운드 자동 분류 및 악보 생성 시스템</h1>
      <p className="subtitle">파일 전송 후 변환 완료 시 다운로드가 가능합니다.</p>
      
      {renderContent()}
    </div>
  );
}

export default App;