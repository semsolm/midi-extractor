// 파일 경로: frontend/src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
// App.css는 App.js 파일 내부에서 import 할 것입니다.

// 1. "root"라는 id를 가진 DOM 요소를 찾습니다.
// (이 "root"는 frontend/index.html 파일에 있습니다.)
const rootElement = document.getElementById('root');

// 2. React 18 스타일로 root를 생성합니다.
const root = ReactDOM.createRoot(rootElement);

// 3. App 컴포넌트를 root에 렌더링(표시)합니다.
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);