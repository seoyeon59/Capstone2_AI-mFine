window.addEventListener("DOMContentLoaded", () => {
  // ==========================
  // 1카메라 상태 표시
  // ==========================
  const statusText = document.getElementById("status");
  const cameraFeed = document.getElementById("cameraFeed");

  // 카메라 연결 성공 시
  cameraFeed.onload = () => {
    statusText.textContent = "✅ 카메라 연결 성공";
  };

  // 카메라 연결 실패 시
  cameraFeed.onerror = () => {
    statusText.textContent = "❌ 카메라 연결 실패 — URL 또는 네트워크 확인";
  };

  // ==========================
  // 위험 점수 표시
  // ==========================
  const riskScore = document.getElementById("riskScore"); // 점수 숫자
  const arrow = document.getElementById("arrow");        // 화살표

  async function fetchRiskScore() {
    try {
      // 실제 모델에서 점수를 가져올 경우 아래 코드 사용
       const response = await fetch("/get_score");
       const data = await response.json();
       const score = Math.round((data.risk_score || 0) * 100);

      // 숫자 표시
      riskScore.textContent = `${score}%`;

      // 화살표 위치 표시
      arrow.style.left = `${score}%`;

      // 점수 색상 변경
      if (score > 70) riskScore.style.color = "#e74c3c"; // 빨강
      else if (score > 40) riskScore.style.color = "#f1c40f"; // 노랑
      else riskScore.style.color = "#27ae60"; // 초록
    } catch (e) {
      console.error("점수 불러오기 실패:", e);
    }
  }

  // 1초마다 위험 점수 업데이트
  setInterval(fetchRiskScore, 1000);
  fetchRiskScore(); // 초기 실행

  // 119 신고 팝업
  const btn119 = document.getElementById("btn119");
  const popup = document.getElementById("popup");
  const confirmBtn = document.getElementById("confirmBtn");
  const cancelBtn = document.getElementById("cancelBtn");

  // 신고 버튼 클릭 → 팝업 표시
  btn119.addEventListener("click", () => popup.classList.add("show"));

  // 확인 버튼 클릭 → 모바일에서는 전화 앱 열기
  confirmBtn.addEventListener("click", () => {
    popup.classList.remove("show");
    window.location.href = "tel:119";
  });

  // 취소 버튼 클릭 → 팝업 닫기
  cancelBtn.addEventListener("click", () => popup.classList.remove("show"));
});
