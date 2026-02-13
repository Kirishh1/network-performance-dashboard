import subprocess
import time
import sqlite3
from datetime import datetime
import os

DB_PATH = "data/network_logs.db"

os.makedirs("data", exist_ok=True)



latency_window = []
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

def ping_target():
    result = subprocess.run(
        ["ping", "-c", "1", "8.8.8.8"],
        capture_output=True,
        text=True
    )
    for line in result.stdout.split("\n"):
        if "time=" in line:
            return float(line.split("time=")[1].split(" ")[0])
    return None

def calculate_jitter():
    if len(latency_window) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(latency_window)):
        diffs.append(abs(latency_window[i] - latency_window[i - 1]))

    return sum(diffs) / len(diffs)


def log_latency(latency):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO metrics VALUES (?, ?)",
        (datetime.now().isoformat(), latency)
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    while True:
        latency = ping_target()
        if latency is not None:
            log_latency(latency)
            print(f"Logged latency: {latency} ms")
        time.sleep(5)
        jitter = calculate_jitter()
        print(f"Jitter: {jitter:.2f} ms")

