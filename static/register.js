document.addEventListener('DOMContentLoaded', function() {
  const registerFormDiv = document.getElementById('registerForm');
  const privacySection = document.getElementById('privacySection');
  const agreeAll = document.getElementById('agreeAll');
  const agreeError = document.getElementById('agreeError');

  const form = registerFormDiv.querySelector('form');
  const userIdInput = document.getElementById('userId');
  const userIdError = document.getElementById('userIdError');
  const password = document.getElementById('password');
  const passwordConfirm = document.getElementById('passwordConfirm');
  const passwordFormatError = document.getElementById('passwordFormatError');
  const passwordError = document.getElementById('passwordError');

  let isIdTaken = false;

  // -------------------- 전체 동의 체크 시 폼 표시 --------------------
  agreeAll.addEventListener('change', () => {
    if (agreeAll.checked) {
      registerFormDiv.style.display = 'block';
      privacySection.style.display = 'none';
      agreeError.style.display = 'none';
    } else {
      registerFormDiv.style.display = 'none';
    }
  });

  // -------------------- 아이디 중복 체크 --------------------
  userIdInput.addEventListener('blur', async function() {
    const userId = userIdInput.value.trim();
    if (!userId) {
      hideUserIdError();
      return;
    }

    try {
      const response = await fetch(`/check_id?id=${encodeURIComponent(userId)}`);

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();

      if (data.exists) showUserIdError("이미 존재하는 아이디입니다.");
      else hideUserIdError();
    } catch (err) {
      console.error(err);
      showUserIdError("아이디 확인 중 오류 발생");
    }
  });

  function showUserIdError(msg) {
    userIdError.textContent = msg;
    userIdError.style.display = 'block';
    userIdInput.classList.add('error-input');
    isIdTaken = true;
  }

  function hideUserIdError() {
    userIdError.style.display = 'none';
    userIdInput.classList.remove('error-input');
    isIdTaken = false;
  }

  // -------------------- 비밀번호 체크 --------------------
  function checkPasswordFormat() {
    const value = password.value;
    const isValid = /^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]).{8,}$/.test(value);
    passwordFormatError.style.display = (value && !isValid) ? 'block' : 'none';
    return isValid;
  }

  function checkPasswordMatch() {
    const match = password.value === passwordConfirm.value;
    passwordError.style.display = (!match && passwordConfirm.value) ? 'block' : 'none';
    return match;
  }

  password.addEventListener('input', () => {
    checkPasswordFormat();
    checkPasswordMatch();
  });

  passwordConfirm.addEventListener('input', checkPasswordMatch);

  // -------------------- 폼 제출 --------------------
  form.addEventListener('submit', function(event) {
    let preventSubmit = false;

    // 전체 동의 체크 확인
    if (!agreeAll.checked) {
      agreeError.style.display = 'block';
      preventSubmit = true;
    } else {
      agreeError.style.display = 'none';
    }

    // 아이디 중복 체크
    if (userIdInput.value.trim() && isIdTaken) {
      userIdError.style.display = 'block';
      preventSubmit = true;
    }

    // 비밀번호 체크
    if (!checkPasswordFormat() || !checkPasswordMatch()) {
      preventSubmit = true;
    }

    if (preventSubmit) event.preventDefault();
  });
});