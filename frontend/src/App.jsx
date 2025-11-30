import React, { useState } from 'react';
import './App.css';
import * as api from './services/api';
import { UploadForm } from './components/UploadForm.jsx';
import { StatusTracker } from './components/StatusTracker.jsx';
import { ResultDisplay } from './components/ResultDisplay.jsx';

// [컴포넌트] MIDI to PDF 뷰 (준비중)
const MidiToPdfView = () => (
  <div className="menu-view">
    <h3>🎼 MIDI to PDF</h3>
    <p>
      MIDI 파일을 업로드하면 고품질 PDF 악보로 변환해 드립니다.<br />
      현재 기능 준비 중입니다. 조금만 기다려주세요!
    </p>
  </div>
);

// [컴포넌트] 도움말 뷰
const HelpView = () => (
  <div className="menu-view">
    <h3>도움말 및 정보</h3>
    <p>
      본 시스템은 <strong>Deep Learning</strong> 기술을 활용하여<br/>
      WAV 오디오를 MIDI와 악보로 정밀하게 변환합니다.
    </p>
    <p style={{ marginTop: '20px', fontSize: '0.9rem', color: '#64748B' }}>
      자세한 기술 스택과 코드는 <br/>
      <a href="https://github.com/semsolm/midi-extractor" target="_blank" rel="noopener noreferrer">GitHub 프로젝트 페이지</a>에서 확인하실 수 있습니다.
    </p>
  </div>
);

// [상수] 푸터 콘텐츠
const APP_FOOTER_CONTENT = (
    <>
        <div className="footer-links">
            <a href="https://github.com/semsolm/midi-extractor/blob/main/readme.md" target="_blank" rel="noopener noreferrer">Privacy Policy</a>
            <span style={{color: '#cbd5e1'}}>|</span>
            <a href="https://github.com/semsolm/midi-extractor/issues" target="_blank" rel="noopener noreferrer">Report Issue</a>
        </div>

        <p style={{ marginTop: '20px', fontWeight: 600 }}>© 2025 Team 경로당. All Rights Reserved.</p>

        <p className="footer-disclaimer">
            본 시스템은 [안양대학교 캡스톤 디자인] 프로젝트의 일환으로 제작되었습니다.<br />
            학습 및 비영리 목적으로만 사용 가능하며, 생성된 데이터의 정확성을 보장하지 않습니다.
        </p>
    </>
);

function App() {
  // UI 상태: 'idle', 'uploading', 'processing', 'completed', 'error'
  const [uiState, setUiState] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [jobResult, setJobResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // 메뉴 상태 (기본값을 wav로 변경)
  const [currentMenu, setCurrentMenu] = useState('wav to midi');

  // 메뉴 리스트 정의 (MP3 -> WAV 수정)
  const MENU_ITEMS = [
    { id: 'wav to midi', label: 'WAV to MIDI' },
    { id: 'midi to pdf', label: 'MIDI to PDF' },
    { id: 'help', label: 'Help' },
  ];

  // 1. 업로드 핸들러
  const handleUpload = async (file) => {
    setUiState('uploading');
    setErrorMessage('');
    try {
      const { jobId } = await api.uploadAudioFile(file);
      setJobId(jobId);
      setUiState('processing');
    } catch (error) {
      setErrorMessage(error.message || '파일 업로드 중 오류가 발생했습니다.');
      setUiState('error');
    }
  };

  // 2. 처리 완료 핸들러
  const handleProcessingComplete = (results) => {
    setJobResult(results);
    setUiState('completed');
  };

  // 3. 에러 핸들러
  const handleProcessingError = (message) => {
    setErrorMessage(message);
    setUiState('error');
  };

  // 4. 초기화 핸들러
  const handleReset = () => {
    setUiState('idle');
    setJobId(null);
    setJobResult(null);
    setErrorMessage('');
  };

  // 5. 메뉴 클릭 핸들러
  const handleMenuClick = (menuName) => {
    setCurrentMenu(menuName);
    // WAV 메뉴를 클릭하면 메인 기능 초기화
    if (menuName === 'wav to midi') {
      handleReset();
    }
  };

  // 메인 컨텐츠 렌더링
  const renderMainContent = () => {
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
            <div className="status-error">
              {errorMessage}
            </div>
            <div style={{ marginTop: '20px' }}>
                <button onClick={handleReset} className="button-primary">
                다시 시도
                </button>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // 메뉴별 컨텐츠 렌더링
  const renderContent = () => {
    switch (currentMenu) {
      case 'wav to midi': // id 변경됨
        return (
          <>
            <h2 className="main-title">
              Music, <br/>
              <span>Transformed by AI.</span>
            </h2>

            <p className="subtitle">
              음악(WAV)을 MIDI와 악보로 변환하세요.<br/>
              AI 기술이 당신의 음악 작업을 돕습니다.
            </p>

            {renderMainContent()}
          </>
        );
      case 'midi to pdf':
        return <MidiToPdfView />;
      case 'help':
        return <HelpView />;
      default:
        return <p>페이지를 찾을 수 없습니다.</p>;
    }
  };

  return (
    <>
      <header className="app-header">
        <div className="header-content">
          <div
            className="logo-section"
            onClick={() => handleMenuClick('wav to midi')}
            title="홈으로 이동"
          >
            <span className="app-logo">🎵</span>
            <span className="app-title">Midi-Extractor</span>
          </div>

          <nav className="header-nav">
            {MENU_ITEMS.map((item) => (
              <button
                key={item.id}
                className={`nav-button ${currentMenu === item.id ? 'active' : ''}`}
                onClick={() => handleMenuClick(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <div className="container">
        {renderContent()}
      </div>

      <footer className="app-footer">
        {APP_FOOTER_CONTENT}
      </footer>
    </>
  );
}

export default App;