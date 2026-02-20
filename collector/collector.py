import subprocess
import time
import sqlite3
from datetime import datetime
import os
import numpy as np
DB_PATH = "data/network_logs.db"

os.makedirs("data", exist_ok=True)


WINDOW_SIZE = 24

latency_window = []

TARGET = "8.8.8.8"
INTERVAL = 5 # seconds


# ---------------------- DB INIT ----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            time TEXT,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()
# ---------------------- HELPER FUNCTIONS ----------------------
def ping_target():
    try:
        result = subprocess.run(
            ["ping", "-c", "1", TARGET],
            capture_output=True,
            text=True,
            timeout=3
        )
        for line in result.stdout.split("\n"):
            if "time=" in line:
                return float(line.split("time=")[1].split(" ")[0])
    except Exception as e:
        print(f"Ping error: {e}")
    return None

def update_window(latency):
    latency_window.append(latency)
    if len(latency_window) > WINDOW_SIZE:
        latency_window.pop(0)


def calculate_jitter():
    if len(latency_window) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(latency_window))))

# ---------------------- MAIN LOOP ----------------------


if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        while True:
            latency = ping_target()
            if latency is not None:
                c.execute(
                    "INSERT INTO metrics VALUES (?, ?)",
                    (datetime.now().isoformat(), latency)
                )
                conn.commit()
                update_window(latency)
                print(f"Logged latency: {latency} ms")

            jitter = calculate_jitter()
            print(f"Jitter: {jitter:.2f} ms")

            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Stopping collector...")
    finally:
        conn.close()
