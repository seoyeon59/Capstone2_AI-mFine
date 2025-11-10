document.addEventListener('DOMContentLoaded', function() {
  const privacySection = document.getElementById('privacySection');
  const agreePrivacy = document.getElementById('agreePrivacy');
  const privacyError = document.getElementById('privacyError');
  const registerFormDiv = document.getElementById('registerForm');
  const form = registerFormDiv.querySelector('form');
  const userIdInput = document.getElementById('userId');
  const userIdError = document.getElementById('userIdError');
  const password = document.getElementById('password');
  const passwordConfirm = document.getElementById('passwordConfirm');
  const passwordError = document.getElementById('passwordError');

  let isIdTaken = false;

  // 개인정보 동의 체크 시 폼 표시 & 안내 숨김
  agreePrivacy.addEventListener('change', function() {
    if (agreePrivacy.checked) {
      registerFormDiv.style.display = 'block';
      privacySection.style.display = 'none';
      privacyError.style.display = 'none';
    } else {
      registerFormDiv.style.display = 'none';
    }
  });

  // 아이디 중복 체크
  userIdInput.addEventListener('blur', async function() {
    const userId = userIdInput.value.trim();
    if (!userId) {
      userIdError.style.display = 'none';
      userIdInput.classList.remove('error-input');
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
        isIdTaken = true;
      } else {
        userIdError.style.display = 'none';
        userIdInput.classList.remove('error-input');
        isIdTaken = false;
      }
    } catch (err) {
      console.error(err);
      userIdError.textContent = "아이디 확인 중 오류 발생";
      userIdError.style.display = 'block';
      userIdInput.classList.add('error-input');
      isIdTaken = true;
    }
  });

  // 비밀번호 일치 실시간 체크
  function checkPasswordMatch() {
    if (passwordConfirm.value === "") {
      passwordError.style.display = 'none';
      return;
    }

    if (password.value !== passwordConfirm.value) {
      passwordError.textContent = "비밀번호가 일치하지 않습니다.";
      passwordError.style.display = 'block';
    } else {
      passwordError.style.display = 'none';
    }
  }

  password.addEventListener('input', checkPasswordMatch);
  passwordConfirm.addEventListener('input', checkPasswordMatch);

  // 폼 제출 시 체크
  form.addEventListener('submit', async function(event) {
    event.preventDefault();

    let preventSubmit = false;

    if (!agreePrivacy.checked) {
      privacyError.style.display = 'block';
      preventSubmit = true;
    } else {
      privacyError.style.display = 'none';
    }

    if (userIdInput.value.trim() && isIdTaken) {
      userIdError.style.display = 'block';
      preventSubmit = true;
    }

    if (password.value !== passwordConfirm.value) {
      passwordError.style.display = 'block';
      preventSubmit = true;
    }

    if (!preventSubmit) {
      form.submit();
    }
  });
});