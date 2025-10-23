window.addEventListener("DOMContentLoaded", () => {
  const statusText = document.getElementById("status");
  const cameraFeed = document.getElementById("cameraFeed");

// 카메라 연결 상태 표시
  cameraFeed.onload = () => {
    statusText.textContent = "✅ 카메라 연결 성공";
  };

  cameraFeed.onerror = () => {
    statusText.textContent = "❌ 카메라 연결 실패 — URL 또는 네트워크 확인";
  };
});

// 119 신고 팝업
document.addEventListener("DOMContentLoaded", () => {
    const btn119 = document.getElementById("btn119");
    const popup = document.getElementById("popup");
    const confirmBtn = document.getElementById("confirmBtn");
    const cancelBtn = document.getElementById("cancelBtn");

    // 신고 버튼 클릭 → 팝업 표시
    btn119.addEventListener("click", () => {
        popup.classList.add("show"); // 이 부분이 팝업 보이게 함
    });

    // 확인 버튼 클릭
    confirmBtn.addEventListener("click", () => {
        popup.classList.remove("show");
        window.location.href = "tel:119"; // 모바일에서는 전화 앱 열림
    });

    // 취소 버튼 클릭
    cancelBtn.addEventListener("click", () => {
        popup.classList.remove("show");
    });
});
