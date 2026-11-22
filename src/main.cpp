
#include <Arduino.h>
#include <WiFi.h>

// Target WiFi access point
const char* TARGET_SSID = "Chandas";
const char* TARGET_BSSID = "8a:57:c4:77:52:91"; 

unsigned long measurementID = 0;

void setup() {

    Serial.begin(115200);
    delay(1500);

    // Initialize WiFi in station mode
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    delay(500);

    Serial.println();
    Serial.println("================================");
    Serial.println(" INTELLIGENT NETWORK COVERAGE");
    Serial.println(" ESP32-S3 RSSI MEASUREMENT");
    Serial.println("================================");

    Serial.print("Target SSID: ");
    Serial.println(TARGET_SSID);

    Serial.print("Target BSSID: ");
    Serial.println(TARGET_BSSID);

    Serial.println("System initialized.");
}

void loop() {

    Serial.println("\nScanning WiFi networks...");

    // Scan nearby WiFi networks
    int networkCount = WiFi.scanNetworks(false, true);

    bool targetFound = false;

    for (int i = 0; i < networkCount; i++) {

        String scannedBSSID = WiFi.BSSIDstr(i);

        if (scannedBSSID.equalsIgnoreCase(TARGET_BSSID)) {

            measurementID++;

            int rssi = WiFi.RSSI(i);
            int channel = WiFi.channel(i);

            Serial.println("--------------------------------");

            Serial.print("Measurement ID: ");
            Serial.println(measurementID);

            Serial.print("SSID: ");
            Serial.println(WiFi.SSID(i));

            Serial.print("BSSID: ");
            Serial.println(scannedBSSID);

            Serial.print("RSSI: ");
            Serial.print(rssi);
            Serial.println(" dBm");

            Serial.print("Channel: ");
            Serial.println(channel);

            Serial.println("Status: TARGET FOUND");

            Serial.println("--------------------------------");

            targetFound = true;
            break;
        }
    }

    if (!targetFound) {

        Serial.println("TARGET NOT FOUND");

    }

    // Clear scan results
    WiFi.scanDelete();

    // Wait before next scan
    delay(2000);
}
