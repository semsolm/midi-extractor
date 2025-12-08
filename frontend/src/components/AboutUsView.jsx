import React from 'react';

export function AboutUsView() {
  const teamMembers = [
    {
      name: '윤상일',
      role: 'AI/ML',
      task: '모델 설계 및 학습',
      github: 'https://github.com/semsolm'
    },
    {
      name: '양태양',
      role: 'Frontend',
      task: '프론트엔드 개발',
      github: 'https://github.com/sunning838'
    },
    {
      name: '최유진',
      role: 'Frontend',
      task: 'UI 디자인',
      github: 'https://github.com/cyj4795'
    },
    {
      name: '이준행',
      role: 'Backend',
      task: 'PM / 백엔드 개발',
      github: 'https://github.com/LeopoldBloom2K'
    },
    {
      name: '정서영',
      role: 'Backend',
      task: '백엔드, 프론트엔드 지원',
      github: 'https://github.com/seoyzz'
    },
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
        <p style={{ fontSize: '0.85rem', color: 'var(--text-sub)', marginBottom: '10px' }}>
          * 카드를 클릭하면 Github 페이지로 이동합니다.
        </p>
        <div className="team-grid">
          {teamMembers.map((member, index) => (
            <a
              className="team-card"
              key={index}
              href={member.github}
              target="_blank"
              rel="noopener noreferrer"
              style={{ textDecoration: 'none', display: 'block', cursor: 'pointer' }}
            >
              <div className="team-card-header">
                <span className="team-name">{member.name}</span>
                <span className="team-role">{member.role}</span>
              </div>
              <p className="team-task">{member.task}</p>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}