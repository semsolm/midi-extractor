import React, { useState, useEffect } from 'react';
import './App.css';
import * as api from './services/api';
import { UploadForm } from './components/UploadForm.jsx';
import { StatusTracker } from './components/StatusTracker.jsx';
import { ResultDisplay } from './components/ResultDisplay.jsx';
import { AboutUsView } from './components/AboutUsView.jsx';
import { HelpView } from './components/HelpView.jsx';

// [컴포넌트] 도움말 뷰
// const HelpView = () => (
//   <div className="menu-view">
//     <h3>도움말 및 정보</h3>
//     <p>
//       본 시스템은 드럼 오디오를 MIDI와 악보로 자동 변환하는 AI 기반 프로젝트입니다.<br />
//       자세한 내용은 <a href="https://github.com/semsolm/midi-extractor" target="_blank" rel="noopener noreferrer">GitHub 프로젝트 페이지</a>를 확인해주세요.
//     </p>
//     <p>문의사항은 '오류/건의' 링크를 이용해 주세요. 🤝</p>
//   </div>
// );

// [상수] 푸터 콘텐츠
const APP_FOOTER_CONTENT = (
  <>
    <div className="footer-links">
      <a href="https://github.com/semsolm/midi-extractor/blob/main/readme.md" target="_blank" rel="noopener noreferrer">개인정보처리방침</a>
      <span>|</span>
      <a href="https://github.com/semsolm/midi-extractor/issues" target="_blank" rel="noopener noreferrer">오류/건의</a>
    </div>

    <p>Copyright © 2025. Team 경로당. All Rights Reserved.</p>
    <p>본 시스템은 [안양대학교 캡스톤 디자인 수업] 의 팀 프로젝트로 제작되었습니다.</p>

    <p className="footer-disclaimer">
      본 시스템은 학습 및 비영리 목적으로만 무료로 사용할 수 있습니다.<br />
      생성된 악보의 정확성을 보장하지 않으며, 사용으로 인한 법적 책임을 지지 않습니다.
    </p>
  </>
);

function App() {
  // UI 상태: 'idle', 'uploading', 'processing', 'completed', 'error'
  const [uiState, setUiState] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [jobResult, setJobResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // 메뉴 상태 (WAV로 변경)
  const [currentMenu, setCurrentMenu] = useState('wav to midi');

  // 메뉴 리스트 정의
  const MENU_ITEMS = [
    { id: 'wav to midi', label: 'WAV to MIDI' },
    { id: 'About Us', label: 'About Us' },
    { id: 'help', label: 'Help' },
  ];

  // 🌙 다크모드 상태
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  // 다크모드 토글 함수
  const toggleDarkMode = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    localStorage.setItem('darkMode', JSON.stringify(newMode));
  };

  // 다크모드 클래스 적용
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }
  }, [isDarkMode]);

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
      case 'wav to midi':
        return (
          <>
            <h2 className="main-title">
              Music, <br />
              <span>Transformed by AI.</span>
            </h2>

            <p className="subtitle">
              음악(WAV)을 MIDI와 악보로 변환하세요.<br />
              AI 기술이 당신의 음악 작업을 돕습니다.
            </p>

            {renderMainContent()}
          </>
        );
      case 'About Us':
        return <AboutUsView />;
      case 'help':
        return <HelpView />;
      default:
        return <p>페이지를 찾을 수 없습니다.</p>;
    }
  };

  //모바일 사이드바 관련

  //사이드 메뉴 열림/닫힘 상태
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  //메뉴 토글 함수
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  // 모바일 메뉴 클릭 시 닫기 처리
  const handleMobileMenuClick = (menuId) => {
    handleMenuClick(menuId);
    setIsMenuOpen(false);
  };

  return (
    <>
      <div
        className={`menu-overlay ${isMenuOpen ? 'open' : ''}`}
        onClick={toggleMenu}
      />

      {/*모바일 사이드 메뉴 패널*/}
      <aside className={`side-menu ${isMenuOpen ? 'open' : ''}`}>


        {/* 메뉴 리스트 (위쪽 여백 추가) */}
        <nav className="side-nav-list" style={{ marginTop: '20px' }}>
          {MENU_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`side-nav-item ${currentMenu === item.id ? 'active' : ''}`}
              onClick={() => handleMobileMenuClick(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <header className="app-header">
        <div className="header-content">
          <button className="hamburger-btn" onClick={toggleMenu} aria-label="메뉴 열기">
            <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none">
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>
          <div
            className="logo-section"
            onClick={() => handleMenuClick('wav to midi')}
            title="홈으로 이동"
          >
            <span className="app-logo">🎵</span>
            <span className="app-title">만드럼</span>
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

            {/* PC용 다크모드 버튼 */}
            <div className="dark-mode-toggle-wrapper">
              <div className="checkbox model-1">
                <input
                  type="checkbox"
                  id="dark-mode-toggle"
                  checked={isDarkMode}
                  onChange={toggleDarkMode}
                />
                <label htmlFor="dark-mode-toggle"></label>
              </div>
            </div>
          </nav>

          {/* 모바일용 다크모드 버튼*/}
          <div className="mobile-toggle-wrapper">
            <div className="checkbox model-1">
              <input
                type="checkbox"
                id="dark-mode-toggle-mobile"
                checked={isDarkMode}
                onChange={toggleDarkMode}
              />
              <label htmlFor="dark-mode-toggle-mobile"></label>
            </div>
          </div>

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