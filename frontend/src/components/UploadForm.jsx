// src/components/UploadForm.jsx
import React, { useState, useEffect } from 'react';

// 백엔드 API 엔드포인트
// (backend/run.py, backend/app/routes.py 참고)
const API_PROCESS_URL = 'http://127.0.0.1:5000/api/process';

export function UploadForm({ onUpload, isLoading }) {
  const [file, setFile] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);

  // 파일이 변경될 때마다 미리듣기 URL 생성
  useEffect(() => {
    if (!file) {
      setAudioPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setAudioPreview(objectUrl);

    // 컴포넌트 unmount 시 object URL 정리
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) {
      onUpload(file); // 부모(App.js)의 handleUpload 호출
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="input-group">
        <label htmlFor="fileInput">변환할 파일 선택 :</label>
        <input
          type="file"
          id="fileInput"
          name="file"
          accept="audio/*" //
          onChange={handleFileChange}
          disabled={isLoading}
        />
      </div>

      {/* index_test.html의 오디오 플레이어 로직 */}
      {audioPreview && (
        <div id="playerContainer" style={{ display: 'block' }}>
          <p id="listenText">들어보기 :</p>
          <audio id="audioPlayer" controls src={audioPreview}></audio>
        </div>
      )}

      <div className="controls">
        <button
          id="startButton"
          type="submit"
          disabled={!file || isLoading}
        >
          {isLoading ? '업로드 중...' : '변환 시작'}
        </button>
      </div>

      <div id="statusMessageElement" className="status-info">
        파일을 선택하고 '변환 시작'을 눌러주세요.
      </div>
    </form>
  );
}