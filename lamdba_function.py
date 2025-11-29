import pymysql
import boto3
import os
import json
from datetime import datetime, timedelta, timezone

# ==========================
# 1. 환경 변수 설정 및 상수 정의
# ==========================
# RDS 연결 정보
DB_HOST = os.environ.get('DB_HOST', 'swu-sw-02-db.cfoqwsiqgd5l.ap-northeast-2.rds.amazonaws.com')
DB_USER = os.environ.get('DB_USER', 'admin')
# 🚨 보안 경고: 실제 운영 환경에서는 Secrets Manager를 사용하세요.
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'aimfine2!')
DB_NAME = os.environ.get('DB_NAME', 'capstone2')
DB_PORT = int(os.environ.get('DB_PORT', 3306))

# 알림 기준 점수 설정
ALERT_SEND_SCORE_CAUTION = 60.0  # 주의 알림 시작 기준
ALERT_SEND_SCORE_CRITICAL = 70.0  # 경고 알림 시작 기준
CAUTION_COOLDOWN_MINUTES = 5  # 주의 알림 쿨다운 시간 (5분)

# SNS 토픽 ARN
SNS_TOPIC_ARN = "arn:aws:sns:ap-northeast-2:760392872177:swu-sw-02-Topic"

# AWS SNS 클라이언트 초기화 (리전: 서울)
SNS_CLIENT = boto3.client("sns", region_name="ap-northeast-2")

# DB 연결을 Global 변수로 설정하여 재사용합니다.
GLOBAL_DB_CONNECTION = None


# ==========================
# 2. DB 연결 함수
# ==========================
def get_db_connection():
    """DB 연결을 생성하고 Global 변수에 저장하여 재사용합니다."""
    global GLOBAL_DB_CONNECTION

    if GLOBAL_DB_CONNECTION is None or not GLOBAL_DB_CONNECTION.open:
        try:
            print(f"INFO: DB 연결 시도 (Host: {DB_HOST}, Port: {DB_PORT})")
            GLOBAL_DB_CONNECTION = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                db=DB_NAME,
                port=DB_PORT,
                # MySQL의 DATETIME 객체를 Python의 datetime 객체로 가져올 때 사용
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=10
            )
            print("✅ DB 연결 성공.")

        except pymysql.MySQLError as e:
            print(f"❌ DB 연결 오류: {e}")
            GLOBAL_DB_CONNECTION = None
            raise  # 연결 실패 시 예외 발생

    return GLOBAL_DB_CONNECTION


# ==========================
# 3. 알림 상태 관리 함수
# ==========================

def get_alert_status(connection, user_id):
    """DB에서 해당 사용자의 마지막 알림 시각을 조회합니다."""
    sql = "SELECT last_caution_alert, last_critical_alert FROM alert_status WHERE user_id = %s;"

    with connection.cursor() as cursor:
        cursor.execute(sql, (user_id,))
        status = cursor.fetchone()

        if status:
            return status
        else:
            # 상태가 없으면 초기 레코드 삽입 후 None 반환
            print(f"INFO: {user_id}의 초기 알림 상태 레코드 생성.")
            insert_sql = "INSERT INTO alert_status (user_id, last_caution_alert, last_critical_alert) VALUES (%s, NULL, NULL);"
            cursor.execute(insert_sql, (user_id,))
            connection.commit()
            return {'last_caution_alert': None, 'last_critical_alert': None}


def update_alert_status(connection, user_id, alert_type):
    """알림 발송 성공 시 DB에 현재 시각을 UTC 기준으로 업데이트합니다."""
    # alert_type: 'caution' 또는 'critical'
    field = f"last_{alert_type}_alert"

    # 🚨 수정됨: NOW() 대신 UTC_TIMESTAMP()를 사용하여 DB에 항상 UTC 기준으로 시간을 저장
    sql = f"UPDATE alert_status SET {field} = UTC_TIMESTAMP() WHERE user_id = %s;"

    with connection.cursor() as cursor:
        cursor.execute(sql, (user_id,))
    connection.commit()
    print(f"✅ DB 알림 상태 업데이트 성공: {alert_type} (UTC)")


# ==========================
# 4. 메인 핸들러 함수
# ==========================
def lambda_handler(event, context):
    """
    Flask 서버로부터 호출되어 위험 점수를 확인하고, SNS 토픽으로 알림을 발행합니다.
    (낙상 경고: 하루 최초 1회, 낙상 주의: 5분 쿨다운)
    """
    connection = None
    try:
        # 1. API Gateway 데이터 처리
        data_dict = json.loads(event['body']) if 'body' in event else event
        target_user_id = data_dict.get('user_id')
        target_risk_score = round(data_dict.get('risk_score', data_dict.get('avg_score', 0.0)), 2)

        print(f"DEBUG: 처리할 데이터: User ID={target_user_id}, Score={target_risk_score:.2f}")

        # 2. 필수 데이터 검사 및 DB 연결
        if not target_user_id:
            print("INFO: 유효한 user_id가 없습니다.")
            return {'statusCode': 200, 'body': 'Missing user_id.'}

        connection = get_db_connection()

        # 3. 사용자 정보 및 알림 상태 조회
        sql_user = "SELECT non_guardian_name, mail, phone_number FROM users WHERE id = %s LIMIT 1;"
        user_info = None
        with connection.cursor() as cursor:
            cursor.execute(sql_user, (target_user_id,))
            user_info = cursor.fetchone()

        monitored_name = user_info.get('non_guardian_name', '모니터링 대상자') if user_info else '모니터링 대상자'

        # 알림 발송 기록 조회
        alert_status = get_alert_status(connection, target_user_id)

        # 현재 시각을 명시적으로 UTC aware 객체로 설정
        now_utc = datetime.now(timezone.utc)

        # 4. 낙상 경고 (70.0 초과) 처리 - 하루 최초 1회만 발송
        if target_risk_score > ALERT_SEND_SCORE_CRITICAL:
            alert_type = 'critical'
            fall_status = "낙상 경고 단계"
            alert_level_message = f"🚨 {monitored_name}님이 {fall_status}입니다 (점수: {target_risk_score:.2f}). 즉시 확인하세요."

            last_alert_time = alert_status['last_critical_alert']
            should_send = True

            if last_alert_time is not None:
                # 🚨 수정됨: DB에서 가져온 시간을 명시적으로 UTC로 간주하여 Python의 now_utc와 날짜 비교
                # DB에 UTC_TIMESTAMP()로 저장했으므로, DB 시간은 이제 UTC입니다.
                if last_alert_time.tzinfo is None:
                    last_alert_time = last_alert_time.replace(tzinfo=timezone.utc)

                # 마지막 알림 시간이 오늘(UTC 기준)과 같은 날짜인지 비교
                if last_alert_time.date() == now_utc.date():
                    should_send = False
                    print(f"INFO: 경고 알림 (Score {target_risk_score:.2f}) - 오늘(UTC 기준) 이미 발송되었으므로 건너뜁니다.")

            if not should_send:
                return {'statusCode': 200, 'body': 'Critical alert skipped due to daily limit.'}

        # 5. 낙상 주의 (60.0 초과 ~ 70.0 이하) 처리 - 5분 쿨다운 적용
        elif target_risk_score > ALERT_SEND_SCORE_CAUTION:
            alert_type = 'caution'
            fall_status = "낙상 주의단계"
            alert_level_message = f"[⚠️ 주의 단계] {monitored_name}님이 {fall_status}입니다 (점수: {target_risk_score:.2f}). 확인해주세요."

            last_alert_time = alert_status['last_caution_alert']
            should_send = True

            if last_alert_time is not None:
                # 🚨 수정됨: DB에서 가져온 시간을 명시적으로 UTC로 간주하여 Python의 now_utc와 시간 비교
                # DB에 UTC_TIMESTAMP()로 저장했으므로, DB 시간은 이제 UTC입니다.
                if last_alert_time.tzinfo is None:
                    last_alert_time = last_alert_time.replace(tzinfo=timezone.utc)

                # 마지막 알림 시간 + 쿨다운 시간
                cooldown_expiry = last_alert_time + timedelta(minutes=CAUTION_COOLDOWN_MINUTES)

                if now_utc < cooldown_expiry:
                    should_send = False
                    time_to_wait = cooldown_expiry - now_utc
                    # 초 단위까지 계산하여 얼마나 기다려야 하는지 명시적으로 출력
                    wait_seconds = int(time_to_wait.total_seconds())
                    print(f"INFO: 주의 알림 (Score {target_risk_score:.2f}) - 쿨다운 중. {wait_seconds}초 후 재발송 가능.")

            if not should_send:
                return {'statusCode': 200, 'body': 'Caution alert skipped due to 5-minute cooldown.'}

        else:
            # 알림 기준 점수 미만 (60.0 이하)
            print(f"INFO: 점수({target_risk_score:.2f})가 최종 알림 기준 미만입니다.")
            return {'statusCode': 200, 'body': 'Skipped due to low score.'}

        # 6. 알림 발송 및 DB 상태 업데이트 (should_send가 True일 경우)

        # 6-1. 전화번호 형식 변환 (이전 코드와 동일)
        raw_phone = user_info.get('phone_number', 'N/A')
        if raw_phone != 'N/A' and raw_phone is not None:
            raw_phone = str(raw_phone)

        monitored_phone = 'N/A'
        if isinstance(raw_phone, str) and raw_phone.isdigit():
            if raw_phone.startswith('0'):
                monitored_phone = '+82' + raw_phone[1:]
            else:
                monitored_phone = '+82' + raw_phone

        # 6-2. SNS 메시지 구성
        message = alert_level_message
        subject = f"[캡스톤 알림] {fall_status}"

        print(f"[INFO] SNS 토픽 발송 시도: {message}")

        # 6-3. SNS 토픽 발행
        response = SNS_CLIENT.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=message,
            Subject=subject,
        )
        print(f"✅ SNS 토픽 발행 성공: MessageId={response.get('MessageId')}")

        # 6-4. DB 상태 업데이트 (UTC_TIMESTAMP() 사용)
        update_alert_status(connection, target_user_id, alert_type)

        return {
            'statusCode': 200,
            'body': f'{alert_type.capitalize()} alert sent successfully.'
        }

    except pymysql.MySQLError as e:
        print(f"❌ DB 연결/쿼리 오류: {e}")
        # DB 연결 오류 발생 시 연결을 닫고 None으로 초기화 (연결 재시도 대비)
        if connection:
            connection.close()
            global GLOBAL_DB_CONNECTION
            GLOBAL_DB_CONNECTION = None
        return {
            'statusCode': 500,
            'body': f"DB Error: {e}"
        }
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return {
            'statusCode': 500,
            'body': f"Unexpected Error: {e}"
        }
    finally:
        # 함수 실행이 끝날 때마다 연결을 닫지 않고, 에러 시에만 닫도록 변경 (재사용 최적화)
        pass