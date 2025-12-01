export function pop(e, type = "circle") {
  // 이벤트 버블링 등으로 인한 좌표 오차 방지
  const x = e.clientX;
  const y = e.clientY;

  for (let i = 0; i < 30; i++) {
    createParticle(x, y, type);
  }
}

function createParticle(x, y, type) {
  const particle = document.createElement("particle");
  document.body.appendChild(particle);

  const size = Math.floor(Math.random() * 20 + 5);

  // 스타일 적용
  particle.style.width = `${size}px`;
  particle.style.height = `${size}px`;
  particle.style.position = "fixed"; // 화면 스크롤과 무관하게 클릭 위치 고정
  particle.style.top = "0";
  particle.style.left = "0";
  particle.style.pointerEvents = "none"; // 클릭 방해 금지
  particle.style.zIndex = "9999"; // 다른 요소 위에 표시

  const destinationX = x + (Math.random() - 0.5) * 2 * 75;
  const destinationY = y + (Math.random() - 0.5) * 2 * 75;

  switch (type) {
    case "square":
      particle.style.background = `hsl(${Math.random() * 90 + 270}, 70%, 60%)`;
      particle.style.border = "1px solid white";
      particle.style.borderRadius = "0";
      break;
    case "circle":
      particle.style.background = `hsl(${Math.random() * 90 + 180}, 70%, 60%)`;
      particle.style.borderRadius = "50%";
      break;
    default:
      particle.style.background = `hsl(${Math.random() * 90 + 180}, 70%, 60%)`;
      particle.style.borderRadius = "50%";
  }

  // Web Animations API 사용
  const animation = particle.animate(
    [
      {
        transform: `translate(${x - size / 2}px, ${y - size / 2}px)`,
        opacity: 1,
      },
      {
        transform: `translate(${destinationX}px, ${destinationY}px)`,
        opacity: 0,
      },
    ],
    {
      duration: 500 + Math.random() * 1000,
      easing: "cubic-bezier(0, .9, .57, 1)",
      delay: Math.random() * 200,
    }
  );

  // 애니메이션 종료 후 요소 삭제 (메모리 누수 방지)
  animation.onfinish = () => {
    particle.remove();
  };
}