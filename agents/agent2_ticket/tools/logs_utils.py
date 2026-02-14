import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

def get_log_file_path(logs_dir: Path) -> Path:
    """
    Returns the path to the log file for the current day.
    """
    date = datetime.now().strftime("%Y-%m-%d")
    return logs_dir / f"chat_log_{date}.json"

def log_event(logs_dir: Path, event_type: str, data: Dict[str, Any]):
    """
    Logs a key event to the daily log file.
    
    Each log entry contains a timestamp, the type of event, and minimal
    relevant data. Events could include:
        - "user_message"       : a message sent by the user
        - "assistant_message"  : a message returned by the assistant
        - "ticket_created"     : a new ticket or request created
        - "error"              : an error or exception
        - "system"             : any important system events
    """
    log_file = get_log_file_path(logs_dir)
    entry = {
        "timestamp": datetime.now().isoformat(),  # ISO format timestamp
        "event_type": event_type,                  # Event category
        "data": data                               # Event-specific data
    }

    # Load existing logs from today's file if it exists
    logs: List[Dict[str, Any]] = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    # Append the new event and save the file
    logs.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def load_logs(logs_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads all log entries from today's log file.
    """
    log_file = get_log_file_path(logs_dir)
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
