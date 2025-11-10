import React, { useState } from 'react';
import './App.css'; // 스타일 파일 import
import * as api from './services/api'; // Api 내에 UI 코드 없음 (가정)
import { UploadForm } from './components/UploadForm.jsx';
import { StatusTracker } from './components/StatusTracker.jsx';
import { ResultDisplay } from './components/ResultDisplay.jsx';
import imi from "./assets/imi.png";
// [신규] 메뉴별 컴포넌트 더미 (기능 구현 시 실제 컴포넌트로 대체)
const MidiToPdfView = () => (
  <div className="menu-view">
    <h3>MIDI to PDF 변환</h3>
    <p>MIDI 파일을 업로드하면 PDF 악보로 변환되는 기능이 곧 추가됩니다.</p>
    <p>기능 추가를 기다려주세요! 🛠️</p>
  </div>
);

const HelpView = () => (
  <div className="menu-view">
    <h3>도움말 및 정보</h3>

    <p>
      본 시스템은 드럼 오디오를 MIDI와 악보로 자동 변환하는 AI 기반 프로젝트입니다.<br/>
      자세한 내용은 <a href="https://github.com/semsolm/midi-extractor" target="_blank" rel="noopener noreferrer">GitHub 프로젝트 페이지</a>를 확인해주세요.
    </p>
    <p>문의사항은 '오류/건의' 링크를 이용해 주세요. 🤝</p>
  </div>
);

const APP_FOOTER_CONTENT = (
    <>
        <p className="footer-links"> {/* 신규 클래스 추가로 가독성 향상 */}

            <a href="https://github.com/semsolm/midi-extractor/blob/main/readme.md" target="_blank" rel="noopener noreferrer">개인정보처리방침 </a> |
            <a href="https://github.com/semsolm/midi-extractor/issues" target="_blank" rel="noopener noreferrer">오류/건의</a>
        </p>

        <p>Copyright © 2025. Team 경로당. All Rights Reserved.</p>
        <p>
            본 시스템은 [안양대학교 캡스톤 디자인 수업] 의 팀 프로젝트로 제작되었습니다.
        </p>

        <p className="footer-disclaimer"> {/* 신규 클래스 추가로 가독성 향상 */}
            본 시스템은 학습 및 비영리 목적으로만 무료로 사용할 수 있습니다.<br />
            생성된 악보의 정확성을 보장하지 않으며, 사용으로 인한 법적 책임을 지지 않습니다.
        </p>
    </>
);


function App() {
  // 메인 기능 UI 상태: 'idle', 'uploading', 'processing', 'completed', 'error'
  const [uiState, setUiState] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [jobResult, setJobResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // 🌟 [신규] 상단바 메뉴 상태: 'mp3 to midi', 'midi to pdf', 'help'
  const [currentMenu, setCurrentMenu] = useState('mp3 to midi');




  // 1. 업로드 폼에서 '변환 시작' 버튼 클릭 시
  const handleUpload = async (file) => {
    setUiState('uploading');
    setErrorMessage('');
    try {
      // API 호출은 가정된 코드입니다.
      const { jobId } = await api.uploadAudioFile(file);
      setJobId(jobId);
      setUiState('processing'); // 업로드 성공 -> '처리 중' 상태로 변경
    } catch (error) {
      setErrorMessage(error.message || '파일 업로드 중 알 수 없는 오류가 발생했습니다.');
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

  // 4. '다시하기' 버튼 클릭 시 (메인 기능 상태 초기화)
  const handleReset = () => {
    setUiState('idle');
    setJobId(null);
    setJobResult(null);
    setErrorMessage('');
  };

  // 5. [신규] 메뉴 클릭 핸들러
  const handleMenuClick = (menuName) => {
    setCurrentMenu(menuName);

    // 메인 기능(MP3 to MIDI)으로 돌아가면, 상태도 초기화
    if (menuName === 'mp3 to midi') {
      handleReset();
    }
  };

  // UI 상태에 따라 다른 컴포넌트를 렌더링 (메인 기능)
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

  // [신규] 현재 선택된 메뉴에 따라 렌더링할 콘텐츠 선택
  const renderContent = () => {
    switch (currentMenu) {
      case 'mp3 to midi':
        return (
          <>
            <h2 className="main-title">Mp3 to Midi</h2>
            <p className="subtitle">.</p>
            <p className = "subtitle"> .</p>
            {renderMainContent()}
          </>
        );
      case 'midi to pdf':
        return <MidiToPdfView />;
      case 'help':
        return <HelpView />;
      default:
        return <p>오류: 알 수 없는 메뉴입니다.</p>;
    }
  };

  return (

    <>

      {/* 🌟 [신규] 상단 메뉴바 (Header) 🌟 */}
      <header className="app-header">
        <div className="header-content">
          <div
            className="logo-section"
            onClick={() => handleMenuClick('mp3 to midi')}
            title="홈으로 이동"
          >
            <span className="app-logo" role="img" aria-label="drum">🎵</span>
            <span className="app-title">Midi-extractor</span>
          </div>

          <nav className="header-nav">
            {['mp3 to midi', 'midi to pdf', 'help'].map((menu) => (
              <button
                key={menu}
                className={`nav-button ${currentMenu === menu ? 'active' : ''}`}
                onClick={() => handleMenuClick(menu)}
              >
                {menu}
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