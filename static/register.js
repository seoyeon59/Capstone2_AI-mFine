document.addEventListener('DOMContentLoaded', function() {
  const userIdInput = document.getElementById('userId');
  const userIdError = document.getElementById('userIdError');
  const form = document.querySelector('form');
  const password = document.getElementById('password');
  const passwordConfirm = document.getElementById('passwordConfirm');
  const passwordError = document.getElementById('passwordError');

  let isIdTaken = false; // 현재 입력한 아이디가 이미 있는지 상태 저장

  // 아이디 blur 시 서버 체크
  userIdInput.addEventListener('blur', async function() {
    const userId = userIdInput.value.trim();
    if (!userId) {
      userIdError.style.display = 'none';
      isIdTaken = false;
      return;
    }

    try {
      const response = await fetch(`/check_id?id=${encodeURIComponent(userId)}`);
      const data = await response.json();

      if (data.exists) {
        userIdError.textContent = "이미 존재하는 아이디입니다.";
        userIdError.style.display = 'block';
        userIdInput.classList.add('error-input');
      } else {
        userIdError.style.display = 'none';
        userIdInput.classList.remove('error-input');
      }
    } catch (err) {
      console.error(err);
      userIdError.textContent = "아이디 확인 중 오류 발생";
      userIdError.style.display = 'block';
      userIdInput.classList.add('error-input');
    }
  });

  // 폼 제출 시 체크
  form.addEventListener('submit', async function(event) {
    event.preventDefault(); // 우선 폼 제출 막기

    let preventSubmit = false;
    // 아이디 중복 체크 (blur에서 체크했더라도 재확인)
    const userId = userIdInput.value.trim();
    if (userId) {
      try {
        const response = await fetch(`/check_id?id=${encodeURIComponent(userId)}`);
        const data = await response.json();
        if (data.exists) {
          userIdError.style.display = 'block';
          isIdTaken = true;
          preventSubmit = true;
        } else {
          userIdError.style.display = 'none';
          isIdTaken = false;
        }
      } catch (err) {
        console.error(err);
        userIdError.style.display = 'block';
        userIdError.textContent = '아이디 확인 중 오류 발생';
        preventSubmit = true;
        isIdTaken = true;
      }
    }

    // 비밀번호 불일치 체크
    if (password.value !== passwordConfirm.value) {
      passwordError.style.display = 'block';
      preventSubmit = true;
    } else {
      passwordError.style.display = 'none';
    }



    // 모든 체크 통과 시 폼 제출
    if (!preventSubmit) {
      form.submit();
    }
  });
});
