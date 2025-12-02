// src/components/StatusTracker.jsx
import * as api from '../services/api';
import React, { useState, useEffect, useRef } from 'react';

export function StatusTracker({ jobId, onComplete, onError }) {
  // 백엔드 tasks.py의 'message' 필드를 표시
  const [statusMessage, setStatusMessage] = useState('서버에 작업을 요청하는 중...');
  // 진행률 상태 추가 (0~100)
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    const checkStatus = async () => {
      try {
        const data = await api.getJobStatus(jobId);

        // audio_processor.py의 tqdm 메시지 등을 그대로 표시
        setStatusMessage(data.message || '상태 확인 중...');

        // 진행률 업데이트 (백엔드에서 progress 필드를 보내는 경우)
        if (data.progress !== undefined) {
          setProgress(data.progress);
        }

        if (data.status === 'completed') {
          clearInterval(intervalRef.current);
          setProgress(100); // 완료 시 100%
          onComplete(data.results); // App.js에 완료 알림
        } else if (data.status === 'error') {
          clearInterval(intervalRef.current);
          onError(data.message || '알 수 없는 오류 발생'); // App.js에 에러 알림
        }
        // 'pending' 또는 'processing'이면 계속 폴링
      } catch (error) {
        clearInterval(intervalRef.current);
        onError(error.message);
      }
    };

    // 1초마다 상태 확인
    intervalRef.current = setInterval(checkStatus, 1000);

    // 컴포넌트 unmount 시 interval 정리
    return () => clearInterval(intervalRef.current);
  }, [jobId, onComplete, onError]);

  return (
    <div className="status-container">
      {/* 스피너 */}
      <div id="spinnerContainer" style={{ display: 'flex', justifyContent: 'center' }}>
        <div className="loader"></div>
      </div>

      {/* 상태 메시지 */}
      <div id="statusMessageElement" className="status-info">
        {statusMessage}
      </div>

      {/* 진행률 바 */}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${progress}%` }}
        ></div>
      </div>

      {/* 진행률 텍스트 */}
      <div className="progress-text">
        작업 진척도: {progress}%
      </div>
    </div>
  );
}