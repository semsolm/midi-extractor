// src/services/api.js
import axios from 'axios';

// 백엔드 Flask 서버 주소 (run.py에서 5000번 포트 사용)
const API_BASE_URL = 'http://127.0.0.1:5000';

/**
 * 1. 오디오 파일을 서버에 업로드하고 Job ID를 받습니다.
 * (POST /api/process)
 */

export const uploadAudioFile = async (file) => {
    const formData = new FormData();
    // 'audio_file' 키는 routes.py의 request.files['audio_file']와 일치해야 함
    formData.append('audio_file', file);

    try {
        const response = await axios.post(`${API_BASE_URL}/api/process`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        // { "jobId": "..." } 반환
        return response.data; 
    } catch (error) {
        console.error("File upload error:", error);
        throw new Error(error.response?.data?.error || '파일 업로드에 실패했습니다.');
    }
};

/**
 * 2. Job ID로 서버의 작업 상태를 폴링(Polling)합니다.
 * (GET /api/result/<job_id>)
 */
export const getJobStatus = async (jobId) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/result/${jobId}`);
        // { status, message, results } 반환
        return response.data;
    } catch (error) {
        console.error("Get status error:", error);
        throw new Error(error.response?.data?.error || '상태 조회에 실패했습니다.');
    }
};

/**
 * 3. 완료된 작업의 MIDI 다운로드 URL을 반환합니다.
 * (GET /download/midi/<job_id>)
 *
 * @param {string} relativeUrl - 서버가 반환한 상대 경로 (e.g., /download/midi/job-id)
 * @returns {string} - 전체 다운로드 URL
 */
export const getFullDownloadUrl = (relativeUrl) => {
    return `${API_BASE_URL}${relativeUrl}`;
};