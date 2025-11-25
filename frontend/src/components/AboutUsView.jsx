import React from 'react';

export function AboutUsView() {
  return (
    // 기존 구조 유지: menu-view + 추가 스타일 클래스 about-view
    <div className="about-view">
      
      {/* 0. 헤더 */}
      <div className="about-header">
        <h2>About Us</h2>
        <p>Team 경로당 (Capstone Design Project)</p>
      </div>

      {/* 1. 프로젝트 개요 & 목적 */}
      <section className="about-section">
        <h3 className="section-title">
          📘 프로젝트 개요
        </h3>
        {/* 기존 구조 유지: 회색 박스 div */}
        <div className="info-box">
          <p>
            본 프로젝트는 사용자가 업로드한 오디오 파일에서 <strong>드럼 사운드를 자동으로 인식하는 AI 시스템</strong>입니다.
          </p>
          <ul>
            <li>🎵 <strong>음원 분리:</strong> 오디오 파일에서 드럼 트랙만 정교하게 추출</li>
            <li>🥁 <strong>AI 분류:</strong> 3가지 클래스 (<strong>Kick, Snare, Hi-hat</strong>) 자동 인식</li>
            <li>🎼 <strong>악보 생성:</strong> 분석된 데이터를 기반으로 <strong>MIDI 파일</strong> 및 <strong>PDF 악보</strong> 자동 생성</li>
          </ul>
        </div>
      </section>

      {/* 2. 팀원 정보 */}
      <section className="about-section">
        <h3 className="section-title">
          👥 Team 경로당
        </h3>
        {/* 기존 구조 유지: 가로 스크롤 컨테이너 div */}
        <div className="table-container">
          <table className="team-table">
            <thead>
              <tr>
                <th style={{ width: '15%' }}>이름</th>
                <th style={{ width: '20%' }}>학번</th>
                <th style={{ width: '25%' }}>역할</th>
                <th>상세 업무</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>윤상일</td>
                <td>2020E7424</td>
                <td>AI/ML</td>
                <td>모델 설계 및 학습</td>
              </tr>
              <tr>
                <td>양태양</td>
                <td>2021E7411</td>
                <td>Frontend</td>
                <td>UI 개발</td>
              </tr>
              <tr>
                <td>최유진</td>
                <td>2023E7518</td>
                <td>Frontend</td>
                <td>UI 디자인</td>
              </tr>
              <tr>
                <td>이준행</td>
                <td>2020E7427</td>
                <td>Backend</td>
                <td>AI, 풀스택 개발</td>
              </tr>
              <tr>
                <td>정서영</td>
                <td>2020U2329</td>
                <td>Backend</td>
                <td>백엔드, 프론트엔드 지원</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      
    </div>
  );
}