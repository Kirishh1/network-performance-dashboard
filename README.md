# Network Performance Dashboard

Real-time network monitoring tool that tracks latency, jitter, and packet loss. Built with Python and Streamlit to monitor internet connection quality.

## Features

- Interactive charts with real-time data visualization
- Network quality scoring (0-100)
- Customizable alert thresholds
- Packet loss detection
- Historical data analysis with P95/P99 percentiles
- Auto-refresh every 5 seconds

## Installation

```bash
git clone https://github.com/Kirishh1/network-performance-dashboard.git
cd network-performance-dashboard
pip install -r requirements.txt
```

## Usage

Start the data collector:
```bash
python collector.py
```

Launch the dashboard:
```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

## How It Works

The collector pings a target server (default: 8.8.8.8) every 5 seconds and stores latency data in SQLite. The dashboard visualizes this data with interactive charts and calculates network quality metrics.

## Configuration

Change ping target in `collector.py`:
```python
TARGET = "8.8.8.8"  # Your preferred server
INTERVAL = 5  # Seconds between pings
```

## Tech Stack

- Python
- Streamlit
- Plotly
- Pandas
- SQLite
