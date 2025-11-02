
|파일 업로드 (UploadForm.jsx)|처리 중 (StatusTracker.jsx)|결과 확인 (ResultDisplay.jsx)|
|------|---|---|
|사용자가 오디오 파일을 선택하고 '변환 시작' 버튼을 누릅니다.|App.jsx의 상태가 'processing'으로 변경되고, 서버의 작업 상태를 폴링(polling)하며 로딩 스피너와 메시지를 표시합니다.|App.jsx의 상태가 'completed'로 변경됩니다. MIDI 다운로드 버튼과 index_test.html에서 기획한 **PDF 악보 뷰어**가 나타납니다.|


```bash
├──frontend/           # 🚀 2. 프론트엔드 (React + Vite)
│   ├── src/            #   - React 소스 코드
│   │   ├── assets/     #     - 이미지 (drummer.png)
│   │   ├── components/ #     - 재사용 가능한 UI 컴포넌트
│   │   ├── services/   #     - API 통신 로직 (axios)
│   │   ├── App.jsx     #     - 메인 앱 컴포넌트 (상태 관리)
│   │   ├── App.css     #     - 전역 스타일시트
│   │   └── index.jsx   #     - React 진입점
│   │
│   ├── index.html      #   - Vite 뼈대 HTML (루트에 위치)
│   ├── package.json    #   - Node.js 의존성 (React, Vite, axios)
│   └── vite.config.js  #   - Vite 설정 파일 (JSX 처리)
```