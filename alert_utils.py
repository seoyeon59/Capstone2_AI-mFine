# 문자 전송 기능 담당
from twilio.rest import Client

# Twilio 계정 정보 (환경변수나 .env에서 관리 권장) --> 수정해야됨
TWILIO_SID = "AC19f7db9b30ffaf0620d8c0d33477d0cd"
TWILIO_TOKEN = "your_auth_token"
TWILIO_FROM = "+8201 2390 2894"

client = Client(TWILIO_SID, TWILIO_TOKEN)

def send_sms(to_number, message):
    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=to_number
        )
        print(f"📩 SMS sent successfully: {msg.sid}")
    except Exception as e:
        print(f"❌ SMS Error: {e}")
