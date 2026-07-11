from datetime import date, timedelta

# ── Severity lookup table ──────────────────────────────────
# YOLO Johana cuma detect KEBERADAAN objek (helmet, safety_vest, dst),
# bukan KETIADAANNYA. Jadi hazard "no_helmet"/"no_safety_vest" itu
# hasil INFERENSI di ai_pipeline.py (person terdeteksi TAPI helmet
# tidak ada di daftar deteksi) — bukan label mentah dari YOLO.
SEVERITY_TABLE = {
    "chemical_spill":  {"risk_level": "critical", "priority": "high",   "due_days": 1},
    "exposed_cable":   {"risk_level": "critical", "priority": "high",   "due_days": 1},
    "wet_floor":       {"risk_level": "high",     "priority": "high",   "due_days": 3},
    "blocked_walkway": {"risk_level": "high",     "priority": "high",   "due_days": 3},
    "no_helmet":       {"risk_level": "medium",   "priority": "medium", "due_days": 7},
    "no_safety_vest":  {"risk_level": "medium",   "priority": "medium", "due_days": 7},
}

# Fallback kalau label tidak dikenali
DEFAULT_SEVERITY = {"risk_level": "medium", "priority": "medium", "due_days": 7}


def get_severity(yolo_label: str, confidence_score: float = 1.0) -> dict:
    """
    Ambil risk_level, priority, dan due_date berdasarkan YOLO label.
    Kalau confidence_score < 0.5, naikkan satu level kehati-hatian.
    """
    rule = SEVERITY_TABLE.get(yolo_label.lower(), DEFAULT_SEVERITY)

    risk_level = rule["risk_level"]
    priority   = rule["priority"]
    due_days   = rule["due_days"]

    # Kalau confidence rendah, treat lebih serius
    if confidence_score < 0.5 and yolo_label.lower() in SEVERITY_TABLE:
        if priority == "low":
            priority = "medium"
        elif priority == "medium":
            priority = "high"
        due_days = max(1, due_days - 2)

    due_date = date.today() + timedelta(days=due_days)

    return {
        "risk_level": risk_level,
        "priority":   priority,
        "due_date":   due_date,
    }
