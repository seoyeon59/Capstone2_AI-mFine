import pymysql
import os
import json
import boto3
from datetime import datetime, timedelta

# RDS 연결 정보 (환경 변수를 통해 안전하게 로드)
DB_HOST = os.environ.get('DB_HOST', 'swu-sw-02-db.cfoqwsiqgd5l.ap-northeast-2.rds.amazonaws.com')  # RDS 엔드포인트
DB_USER = os.environ.get('DB_USER', 'admin')  # RDS 마스터 사용자 이름
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'aimfine2!')  # RDS 마스터 암호
DB_NAME = os.environ.get('DB_NAME', 'capstone2')
DB_PORT = 3306

# SNS 클라이언트는 SMS 발송에 사용됩니다.
sns_client = boto3.client('sns')


# DB 연결 함수 (이전 코드와 동일)
def get_db_connection():
    # ... (DB 연결 로직)
    try:
        conn = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10
        )
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None


# 전화번호와 알람 기록을 가져오는 헬퍼 함수
def get_user_info_and_alert_time(cursor, user_id, alert_level):
    # 1. 사용자 전화번호 조회
    cursor.execute("SELECT phone_number FROM users WHERE id = %s", (user_id,))
    user_info = cursor.fetchone()
    phone_number = user_info['phone_number'] if user_info else None

    # 2. 마지막 알람 발송 시간 조회
    cursor.execute(
        "SELECT last_sent_timestamp FROM alert_history WHERE user_id = %s AND alert_level = %s",
        (user_id, alert_level)
    )
    last_alert = cursor.fetchone()
    last_sent_time = last_alert['last_sent_timestamp'] if last_alert else None

    return phone_number, last_sent_time


# 알람 기록을 업데이트하는 헬퍼 함수
def update_alert_history(cursor, user_id, alert_level):
    # MySQL의 ON DUPLICATE KEY UPDATE 문법을 사용하여 UPSERT (INSERT OR UPDATE) 수행
    sql = """
        INSERT INTO alert_history (user_id, alert_level, last_sent_timestamp)
        VALUES (%s, %s, NOW())
        ON DUPLICATE KEY UPDATE last_sent_timestamp = NOW()
    """
    cursor.execute(sql, (user_id, alert_level))


# SMS 발송 로직
def send_sms_alert(phone_number, subject, message):
    if not phone_number:
        print("전화번호 없음, SMS 발송 실패")
        return

    # SNS를 통해 SMS를 직접 발송 (Topic ARN 대신 PhoneNumber 사용)
    try:
        response = sns_client.publish(
            PhoneNumber=phone_number,
            Message=message,
            Subject=subject  # Subject는 일부 통신사에서만 표시될 수 있습니다.
        )
        print(f"✅ SMS 발송 성공: {phone_number} - Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"❌ SMS 발송 실패: {e}")


def lambda_handler(event, context):
    conn = get_db_connection()
    if conn is None:
        return {"statusCode": 500, "body": "DB Connection Failed"}

    current_time = datetime.now()

    try:
        with conn.cursor() as cursor:
            # 1. 낙상 점수(risk_score)가 가장 높은 레코드를 조회 (예: 최근 2분 이내)
            cutoff_time = current_time - timedelta(minutes=5)
            sql = f"""
                SELECT user_id, risk_score, timestamp 
                FROM realtime_screen 
                WHERE timestamp >= '{cutoff_time.strftime("%Y-%m-%d %H:%M:%S")}'
                ORDER BY risk_score DESC
                LIMIT 1
            """
            cursor.execute(sql)
            latest_data = cursor.fetchone()

            if not latest_data:
                print("최근 5분 이내 낙상 데이터 없음.")
                return {"statusCode": 200, "body": "No recent data to analyze"}

            score = latest_data['risk_score']
            user_id = latest_data['user_id']

            # 2. 알람 레벨 결정
            if score >= 70:
                alert_level = "WARNING"
                subject = "🚨 긴급 낙상 경고"
                message = f"[낙상 경고] {user_id}님의 위험 점수 {score:.2f}% (70% 이상). 즉시 확인 필요."
                required_interval_sec = 99999999  # 경고: 1회만 발송
            elif score >= 60:
                alert_level = "CAUTION"
                subject = "⚠️ 낙상 주의 알람"
                message = f"[낙상 주의] {user_id}님의 위험 점수 {score:.2f}% (60% 이상). 관찰 요망."
                required_interval_sec = 5 * 60  # 주의: 5분 간격 (300초)
            else:
                print(f"점수 {score:.2f}% (60% 미만), 알람 미발송.")
                return {"statusCode": 200, "body": "Score below alert threshold"}

            # 3. 사용자 정보 및 마지막 발송 시간 확인
            phone_number, last_sent_time = get_user_info_and_alert_time(cursor, user_id, alert_level)

            # 4. 발송 조건 확인 및 SMS 발송

            should_send = False

            if alert_level == "WARNING":
                # 경고 (WARNING): 기록이 없어야 발송 (최초 1회)
                if last_sent_time is None:
                    should_send = True

            elif alert_level == "CAUTION":
                # 주의 (CAUTION): 기록이 없거나, 마지막 발송 후 5분(300초)이 경과해야 발송
                if last_sent_time is None:
                    should_send = True
                else:
                    time_diff = (current_time - last_sent_time).total_seconds()
                    if time_diff >= required_interval_sec:
                        should_send = True

            if should_send:
                # 5. SMS 발송
                send_sms_alert(phone_number, subject, message)

                # 6. 알람 기록 업데이트 (발송 후 기록/갱신)
                update_alert_history(cursor, user_id, alert_level)

            else:
                print(f"알람 발송 주기에 도달하지 않았거나 이미 발송된 경고입니다. (레벨: {alert_level})")

        conn.commit()
        return {"statusCode": 200, "body": "Alert check and send completed."}

    except Exception as e:
        print(f"Lambda Execution Error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
    finally:
        if conn:
            conn.close()