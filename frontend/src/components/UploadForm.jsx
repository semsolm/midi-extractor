import React, { useState, useEffect } from 'react';

const API_PROCESS_URL = 'http://127.0.0.1:5000/api/process';

export function UploadForm({ onUpload, isLoading }) {
  const [file, setFile] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // 파일 미리듣기 URL 생성
  useEffect(() => {
    if (!file) {
      setAudioPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setAudioPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  // 드래그 이벤트 핸들러
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // 드롭 이벤트
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  // 파일 선택
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  // 업로드 버튼 클릭 시
  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) onUpload(file);
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* === 드래그 앤 드롭 영역 === */}
      <div
        className={`drop-zone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="fileInput"
          name="file"
          accept="audio/*"
          capture="microphone"
          onChange={handleFileChange}
          disabled={isLoading}
          hidden
        />

        <label htmlFor="fileInput" className="drop-label">
          {file ? (
            <>
              <strong>{file.name}</strong>
              <p>다른 파일을 드래그하거나 클릭해 변경하세요.</p>
            </>
          ) : (
            <>
              <p>🎵 파일을 이곳에 드래그하세요! </p>
              <button
                type="button"
                className="drop-button"
                onClick={() => document.getElementById('fileInput').click()}
                disabled={isLoading}
              >
                파일 선택
              </button>
            </>
          )}
        </label>
      </div>

      {/* === 오디오 미리듣기 === */}
      {audioPreview && (
        <div id="playerContainer" style={{ display: 'block' }}>
          <p id="listenText">들어보기 :</p>
          <audio id="audioPlayer" controls src={audioPreview}></audio>
        </div>
      )}

      {/* === 변환 시작 버튼 === */}
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
        {file ? `'${file.name}' 파일이 선택되었습니다.` : "파일을 선택하고 '변환 시작'을 눌러주세요."}
      </div>
    </form>
  );
}
