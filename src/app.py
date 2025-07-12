import os
from flask import Flask, request, jsonify
from kafka import KafkaProducer, KafkaConsumer
import json
import threading
import time

# --- 설정 ---
KAFKA_BROKER = 'localhost:9092'
PRODUCER_TOPIC = 'flask-topic'
CONSUMER_TOPIC = 'flask-topic'
MODE = os.environ.get("MODE", "development")

if MODE == "docker":
    KAFKA_BROKER = 'kafka:9092'  # Docker Compose 에서 Kafka 서비스 이름 사용
elif MODE == "kubernetes":
    KAFKA_BROKER = 'kafka-svc:9092'


# --- Flask 앱 생성 ---
app = Flask(__name__)

# --- Kafka Producer 설정 ---
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
except Exception as e:
    print(f"Error creating producer: {e}")
    producer = None

# --- Kafka Consumer를 별도 스레드에서 실행 ---
def consume_messages():
    try:
        consumer = KafkaConsumer(
            CONSUMER_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='latest',
            group_id='flask-consumer-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        print(f"Consumer started on topic '{CONSUMER_TOPIC}'...")
        for message in consumer:
            print(f"Received message: {message.value}")
    except Exception as e:
        print(f"Error creating or running consumer: {e}")
        time.sleep(5) # 재시도 전 대기
        consume_messages() # 에러 발생 시 재귀적으로 재시도

# --- API 엔드포인트 ---
@app.route('/python', methods=['GET'])
def index():
    return "python API is running!"

@app.route('/python/message', methods=['POST'])
def send():
    if not producer:
        return jsonify({"error": "Producer is not available"}), 500
    
    message_data = request.get_json()
    if not message_data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    try:
        producer.send(PRODUCER_TOPIC, value=message_data)
        producer.flush()
        print(f"Sent message: {message_data}")
        return jsonify({"status": "Message sent successfully", "data": message_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 메인 실행 ---
if __name__ == '__main__':
    # Consumer를 백그라운드 스레드로 실행
    consumer_thread = threading.Thread(target=consume_messages, daemon=True)
    consumer_thread.start()
    
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5001, debug=(MODE != "kubernetes"))
