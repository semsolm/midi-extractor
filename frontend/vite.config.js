import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  // React 플러그인을 사용하도록 설정합니다.
  // 이 플러그인이 .js 파일에서도 JSX 문법을
  // 인식할 수 있게 해줍니다.
  plugins: [react()], 
});