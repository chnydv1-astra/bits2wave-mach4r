
import paho.mqtt.client as mqtt
import json
import csv
import os
from datetime import datetime

# ===================================
# MQTT CONFIGURATION
# ===================================

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC = "coverage-ai/test-7f3a/telemetry/rssi"

CSV_FILE = "rssi_measurements.csv"

# ===================================
# CSV INITIALIZATION
# ===================================

file_exists = os.path.exists(CSV_FILE)

csv_file = open(CSV_FILE, "a", newline="", buffering=1)

writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow([
        "timestamp",
        "device_id",
        "ssid",
        "rssi",
        "sequence",
        "esp_timestamp"
    ])

received_count = 0

# ===================================
# MQTT CALLBACKS
# ===================================

def on_connect(client, userdata, flags, reason_code, properties):

    if reason_code == 0:

        print("\nConnected to MQTT broker")

        client.subscribe(TOPIC, qos=0)

        print("Subscribed:", TOPIC)
        print("Collecting RSSI measurements...")
        print("Press CTRL+C to finish collection.")

    else:
        print("Connection failed:", reason_code)


def on_message(client, userdata, message):

    global received_count

    try:

        data = json.loads(message.payload.decode())

        timestamp = datetime.now().isoformat(
            timespec="milliseconds"
        )

        writer.writerow([
            timestamp,
            data["device_id"],
            data["ssid"],
            data["rssi"],
            data["sequence"],
            data["timestamp"]
        ])

        received_count += 1

        print(
            f"Sample {received_count} | "
            f"RSSI: {data['rssi']} dBm | "
            f"Sequence: {data['sequence']}"
        )

    except Exception as error:

        print("Invalid measurement:", error)


# ===================================
# MQTT CLIENT
# ===================================

client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="COVERAGE_AI_COLLECTOR"
)

client.on_connect = on_connect
client.on_message = on_message

# ===================================
# START COLLECTION
# ===================================

try:

    print("Connecting to HiveMQ...")

    client.connect(BROKER, PORT, 60)

    client.loop_forever()

except KeyboardInterrupt:

    print("\nStopping measurement collection...")

finally:

    client.disconnect()
    csv_file.close()

    print("Total measurements:", received_count)
    print("Saved to:", CSV_FILE)
