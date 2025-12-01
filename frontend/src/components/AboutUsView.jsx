import React from 'react';

export function AboutUsView() {
  const teamMembers = [
    { name: '윤상일', id: '2020E7424', role: 'AI/ML', task: '모델 설계 및 학습' },
    { name: '양태양', id: '2021E7411', role: 'Frontend', task: 'UI 개발' },
    { name: '최유진', id: '2023E7518', role: 'Frontend', task: 'UI 디자인' },
    { name: '이준행', id: '2020E7427', role: 'Backend', task: 'AI, 풀스택 개발' },
    { name: '정서영', id: '2020U2329', role: 'Backend', task: '백엔드, 프론트엔드 지원' },
  ];

  return (
    <div className="about-container">
      {/* 헤더 */}
      <div className="about-header">
        <h2>About Us</h2>
        <p>Team 경로당 · Capstone Design Project</p>
      </div>

      {/* 프로젝트 소개 */}
      <section className="about-section">
        <h3>프로젝트 소개</h3>
        <p className="about-description">
          드럼 오디오를 AI가 분석하여 MIDI와 악보로 자동 변환하는 시스템입니다.
        </p>

        <div className="feature-list">
          <div className="feature-item">
            <span className="feature-icon">🎵</span>
            <div>
              <strong>음원 분리</strong>
              <p>오디오에서 드럼 트랙만 추출</p>
            </div>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🥁</span>
            <div>
              <strong>AI 인식</strong>
              <p>Kick, Snare, Hi-hat 자동 분류</p>
            </div>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🎼</span>
            <div>
              <strong>악보 생성</strong>
              <p>MIDI 및 PDF 악보 자동 생성</p>
            </div>
          </div>
        </div>
      </section>

      {/* 팀원 소개 */}
      <section className="about-section">
        <h3>팀원 소개</h3>
        <div className="team-grid">
          {teamMembers.map((member, index) => (
            <div className="team-card" key={index}>
              <div className="team-card-header">
                <span className="team-name">{member.name}</span>
                <span className="team-role">{member.role}</span>
              </div>
              <p className="team-task">{member.task}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}