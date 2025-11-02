// src/components/StatusTracker.jsx
import * as api from '../services/api';
import React, { useState, useEffect, useRef } from 'react';

export function StatusTracker({ jobId, onComplete, onError }) {
  // 백엔드 tasks.py의 'message' 필드를 표시
  const [statusMessage, setStatusMessage] = useState('서버에 작업을 요청하는 중...');
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    const checkStatus = async () => {
      try {
        const data = await api.getJobStatus(jobId);
        
        // audio_processor.py의 tqdm 메시지 등을 그대로 표시
        setStatusMessage(data.message || '상태 확인 중...');

        if (data.status === 'completed') {
          clearInterval(intervalRef.current);
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

    // 1.5초마다 상태 확인
    intervalRef.current = setInterval(checkStatus, 1500);

    // 컴포넌트 unmount 시 interval 정리
    return () => clearInterval(intervalRef.current);
  }, [jobId, onComplete, onError]);

  return (
    <div className="status-container">
      {/* index_test.html의 스피너 스타일 */}
      <div id="spinnerContainer" style={{ display: 'flex' }}>
        <div className="loader"></div>
      </div>
      <div id="statusMessageElement" className="status-info">
        {statusMessage}
      </div>
    </div>
  );
}