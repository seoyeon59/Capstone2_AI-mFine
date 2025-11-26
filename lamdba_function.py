import pymysql
import boto3
import os
import json
from datetime import datetime, timedelta

# ==========================
# 1. 환경 변수 설정 및 상수 정의
# ==========================
# RDS 연결 정보 (보안을 위해 환경 변수 대신 Secrets Manager 사용 권장)
DB_HOST = os.environ.get('DB_HOST', 'swu-sw-02-db.cfoqwsiqgd5l.ap-northeast-2.rds.amazonaws.com')
DB_USER = os.environ.get('DB_USER', 'admin')
# 🚨 보안 경고: 실제 운영 환경에서는 DB_PASSWORD를 Secrets Manager에서 가져오세요.
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'aimfine2!')
DB_NAME = os.environ.get('DB_NAME', 'capstone2')
DB_PORT = int(os.environ.get('DB_PORT', 3306))

# Flask 서버에서 호출하는 최소 알림 기준 (40점 초과)
ALERT_MIN_SCORE = 40.0

# SNS 클라이언트 초기화 (Lambda 실행 환경에서 재사용)
SNS_CLIENT = boto3.client("sns", region_name="ap-northeast-2")

# DB 연결을 Global 변수로 설정하여 Lambda 'Warm Start' 시 재사용하도록 합니다.
GLOBAL_DB_CONNECTION = None


# ==========================
# 2. DB 연결 함수
# ==========================
def get_db_connection():
    """
    DB 연결을 생성하고 Global 변수에 저장하여 재사용합니다.
    """
    global GLOBAL_DB_CONNECTION

    # 기존 연결이 유효한지 확인하고, 유효하지 않으면 새로 연결합니다.
    if GLOBAL_DB_CONNECTION is None or not GLOBAL_DB_CONNECTION.open:
        try:
            print(f"INFO: DB 연결 시도 (Host: {DB_HOST}, Port: {DB_PORT})")
            GLOBAL_DB_CONNECTION = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                db=DB_NAME,
                port=DB_PORT,
                cursorclass=pymysql.cursors.DictCursor,  # 결과를 딕셔너리로 받도록 설정
                connect_timeout=10  # 연결 타임아웃 설정
            )
            print("✅ DB 연결 성공.")

        except pymysql.MySQLError as e:
            print(f"❌ DB 연결 오류: {e}")
            GLOBAL_DB_CONNECTION = None
            raise  # 연결 실패 시 예외 발생

    return GLOBAL_DB_CONNECTION


# ==========================
# 3. 메인 핸들러 함수
# ==========================
def lambda_handler(event, context):
    """
    Flask 서버로부터 호출되어 위험 점수를 확인하고, 알림을 전송합니다.
    """

    try:
        # 1. API Gateway 데이터 처리 (Flask가 보낸 user_id와 risk_score 추출)
        data_dict = {}
        if 'body' in event:
            # API Gateway Proxy 통합 시 JSON 문자열을 파싱
            data_dict = json.loads(event['body'])
        else:
            data_dict = event

        target_user_id = data_dict.get('user_id')
        target_risk_score = data_dict.get('risk_score', 0.0)

        print(f"DEBUG: 처리할 데이터: User ID={target_user_id}, Score={target_risk_score:.2f}")

        # 2. 필수 데이터 및 알림 조건 검사
        if not target_user_id or target_risk_score <= ALERT_MIN_SCORE:
            print(f"INFO: 유효한 user_id가 없거나, 점수({target_risk_score:.2f})가 최소 알림 기준({ALERT_MIN_SCORE}) 미만입니다.")
            return {'statusCode': 200, 'body': 'Skipped due to low score or missing user_id.'}

        # 3. DB에서 사용자 정보(전화번호 및 모니터링 대상자 이름) 조회
        connection = get_db_connection()

        # 쿼리: phone_number와 non_guardian_name 컬럼을 조회합니다.
        sql_user = "SELECT phone_number, non_guardian_name FROM users WHERE id = %s LIMIT 1;"

        user_info = None
        with connection.cursor() as cursor:
            cursor.execute(sql_user, (target_user_id,))
            user_info = cursor.fetchone()

        # 4. 조회된 전화번호로 알림 전송
        if user_info and 'phone_number' in user_info:
            user_id = target_user_id
            risk_score = target_risk_score

            # 모니터링 대상자 이름을 가져옵니다. (없으면 '모니터링 대상자'로 기본값 설정)
            monitored_name = user_info.get('non_guardian_name', '모니터링 대상자')
            phone_number = str(user_info['phone_number'])

            # --- 전화번호 형식 변환 ---
            if not phone_number.startswith('+'):
                phone_number = '+82' + phone_number.lstrip('0')

            print(f"DEBUG: SMS 발송 대상 전화번호 (SNS 형식): {phone_number}")

            # 🚩 위험 점수 레벨에 따른 메시지 구성
            if risk_score > 70.0:
                # 70% 초과: 경고 단계 (빨강)
                fall_status = "낙상 경고 단계"
                # 메시지에 이름과 점수, 경고 수준 반영
                alert_level_message = f"🚨 {monitored_name}님이 {fall_status}입니다 (점수: {risk_score:.2f}). 즉시 확인하세요."
            elif risk_score > 60.0:
                # 60% 초과: 주의 단계 (노랑)
                fall_status = "낙상 주의단계"
                # 메시지에 이름과 점수, 주의 수준 반영
                alert_level_message = f"[⚠️ 주의 단계] {monitored_name}님이 {fall_status}입니다 (점수: {risk_score:.2f}). 확인해주세요."
            else:
                # 40% < score <= 60% 구간은 알림을 건너뜕니다.
                print(f"INFO: 점수 {risk_score:.2f}는 60점 이하입니다. 알림을 건너뜁니다.")
                return {'statusCode': 200, 'body': 'Score below 60.0 threshold.'}

            # SNS 메시지 최종 구성
            message = f"[캡스톤] {alert_level_message}"

            print(f"[INFO] SMS 발송 시도: {message}")

            # 5. SNS SMS 전송
            try:
                response = SNS_CLIENT.publish(
                    PhoneNumber=phone_number,
                    Message=message
                )
                print(f"✅ SNS SMS 전송 성공: MessageId={response.get('MessageId')}")

            except Exception as e:
                print(f"❌ SMS 발송 실패 (SNS Publish): {e}")

        else:
            # DB 조회 실패 시 진단 로그
            if user_info is None:
                print(f"WARNING: DB에 사용자 ID ({target_user_id})의 정보가 없습니다. (ID 미존재 확인 필요)")
            else:
                # user_info는 있으나 'phone_number' 키가 없는 경우 (컬럼명 오류 가능성)
                print(f"WARNING: 사용자 ID ({target_user_id})는 존재하지만, 'phone_number' 컬럼이 누락되었습니다. (컬럼 이름 확인 필요)")

        return {
            'statusCode': 200,
            'body': 'Alert check and send completed.'
        }

    except pymysql.MySQLError as e:
        print(f"❌ DB 연결/쿼리 오류: {e}")
        global GLOBAL_DB_CONNECTION
        if GLOBAL_DB_CONNECTION:
            GLOBAL_DB_CONNECTION.close()
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
        # Global Connection 패턴을 사용하므로 연결을 닫지 않습니다.
        pass