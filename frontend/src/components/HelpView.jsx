import React from 'react';

export function HelpView() {
  return (
    <div className="about-container">
      {/* 1. 헤더 영역 */}
      <div className="about-header">
        <h2>도움말 & 가이드</h2>
        <p>Midi-Extractor 사용 방법 및 문제 해결</p>
      </div>

      {/* 2. 사용 방법 섹션 */}
      <section className="about-section">
        <h3>사용 방법</h3>
        <div className="feature-list">
          <div className="feature-item">

            <div>
              <strong>1. 파일 업로드</strong>
              <p>파일 선택 버튼을 누르거나 오디오 파일(WAV, MP3)을 드래그하여 업로드하세요.</p>
            </div>
          </div>
          <div className="feature-item">
            <div>
              <strong>2. AI 변환 시작</strong>
              <p>'변환 시작' 버튼을 누르면 AI가 드럼 소리를 분리하고 MIDI와 악보를 자동 생성합니다.</p>
            </div>
          </div>
          <div className="feature-item">
            <div>
              <strong>3. 결과 다운로드</strong>
              <p>변환이 완료되면 드럼 오디오, MIDI 파일, PDF 악보를 다운로드할 수 있습니다.</p>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 문제 해결 가이드 섹션 */}
      <section className="about-section">
        <h3>자주 묻는 질문 (Troubleshooting)</h3>
        <div className="feature-list">
          <div className="feature-item">
            <div>
              <strong>파일 업로드가 안 될 때</strong>
              <p>파일 형식이 WAV/MP3인지 확인해 주세요.</p>
            </div>
          </div>
          <div className="feature-item">
            <div>
              <strong>변환이 멈춘 것 같을 때</strong>
              <p>페이지를 새로고침 후 다시 시도하거나, 다른 브라우저(Chrome 권장)를 이용해 보세요.</p>
            </div>
          </div>
          <div className="feature-item">
            <div>
              <strong>다운로드가 되지 않을 때</strong>
              <p>브라우저의 팝업 차단 또는 다운로드 차단 설정이 되어있는지 확인해 주세요.</p>
            </div>
          </div>
          <div className="feature-item">
            <span className="feature-icon"></span>{/*아이콘 태그*/}
            <div>
              <strong>변환 시간이 너무 오래 걸려요</strong>
              <p>곡이 길거나 사용자가 많으면 지연될 수 있습니다. 10분 이상 반응이 없으면 새로고침 해주세요.</p>
            </div>
          </div>
        </div>
      </section>
      
       {/* 4. 문의하기 링크 (선택 사항) */}
       <div style={{ marginTop: '40px', textAlign: 'center' }}>
        <p className="subtitle" style={{ fontSize: '0.9rem' }}>
          해결되지 않는 문제가 있나요? <br/>
          <a href="https://github.com/semsolm/midi-extractor/issues" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>
            Github 이슈
          </a>
          에 제보해 주세요.
        </p>
      </div>
    </div>
  );
}