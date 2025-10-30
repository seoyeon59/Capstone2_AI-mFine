# 문자 전송 기능 담당
from twilio.rest import Client


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
