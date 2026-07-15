from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
import uuid
import os
from supabase import create_client
from app.database import get_db
from app.middleware.auth import get_current_user, inspector_only, manager_or_admin
from app.models.user import User
from app.models.inspection import Inspection
from app.models.hazard import Hazard
from app.models.corrective_action import CorrectiveAction
from app.services.ai_pipeline import call_yolo, call_rag
from app.services.severity_rules import get_severity
from app.services import email_service


# ── Geometry & PPE inference helpers ───────────────────────────────
def compute_iou(box_a, box_b):
    """
    Hitung Intersection-over-Union dua bbox berbentuk list [x1, y1, x2, y2].
    Defensif: menerima list kosong / kurang dari 4 elemen / box zero-area,
    dan TIDAK PERNAH raise IndexError, ZeroDivisionError, atau TypeError.
    Kembalikan 0.0 untuk semua kasus yang tidak valid.
    """
    # Harus list/tuple dengan minimal 4 elemen (pakai len(), bukan .get())
    if not isinstance(box_a, (list, tuple)) or not isinstance(box_b, (list, tuple)):
        return 0.0
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    try:
        ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
        bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    except (TypeError, ValueError):
        return 0.0

    # Normalisasi supaya (x1,y1) pojok kiri-atas, (x2,y2) pojok kanan-bawah
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    if area_a <= 0 or area_b <= 0:  # box zero-area → IoU tidak bermakna
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    union = area_a + area_b - inter
    if union <= 0:  # guard ZeroDivisionError
        return 0.0

    return inter / union


def infer_ppe_violations(detections):
    """
    Inferensi pelanggaran PPE per-orang dari deteksi mentah YOLO.

    Return: list baru berisi HANYA (1) pelanggaran PPE hasil inferensi
    ("no_helmet"/"no_safety_vest") + (2) hazard non-PPE (dengan risk_level
    dilekatkan). Deteksi mentah person/helmet/hard_hat/safety_vest/vest
    TIDAK ikut muncul di output.

    Semua akses key dict pakai .get() dengan default; aman terhadap input
    kosong, dict tanpa key, tidak ada person, atau person tanpa helmet/vest.
    """
    HELMET_LABELS = {"helmet", "hard_hat"}
    VEST_LABELS = {"safety_vest", "vest"}
    PPE_LABELS = {"person", "helmet", "hard_hat", "safety_vest", "vest"}
    RISK_MAP = {
        "blocked_walkway":   "high",
        "wet_floor":         "medium",
        "exposed_cable":     "high",
        "fire_hazard":       "critical",
        "spill":             "medium",
        "missing_guardrail": "critical",
    }
    HELMET_IOU_THRESHOLD = 0.05
    VEST_IOU_THRESHOLD = 0.10

    if not isinstance(detections, (list, tuple)) or not detections:
        return []

    persons, helmets, vests, output = [], [], [], []

    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label", "")).lower()

        if label == "person":
            persons.append(det)
        elif label in HELMET_LABELS:
            helmets.append(det)
        elif label in VEST_LABELS:
            vests.append(det)
        elif label in PPE_LABELS:
            # PPE lain yang harus difilter keluar — jangan diteruskan
            continue
        else:
            # Hazard non-PPE → teruskan dengan risk_level dari mapping
            hazard = dict(det)
            hazard["risk_level"] = RISK_MAP.get(label, "medium")
            output.append(hazard)

    # Cek PPE per orang lewat hubungan spasial (IoU)
    for person in persons:
        person_bbox = person.get("bbox", [])
        if not isinstance(person_bbox, (list, tuple)) or len(person_bbox) < 4:
            # Tanpa bbox valid → tidak bisa inferensi spasial, lewati orang ini
            continue

        try:
            px1, py1, px2, py2 = (
                float(person_bbox[0]), float(person_bbox[1]),
                float(person_bbox[2]), float(person_bbox[3]),
            )
        except (TypeError, ValueError):
            continue

        y_top, y_bot = min(py1, py2), max(py1, py2)
        x_left, x_right = min(px1, px2), max(px1, px2)
        # Region kepala = separuh ATAS bbox person (untuk cek helmet)
        top_half = [x_left, y_top, x_right, y_top + (y_bot - y_top) / 2.0]

        wearing_helmet = any(
            compute_iou(h.get("bbox", []), top_half) >= HELMET_IOU_THRESHOLD
            for h in helmets
        )
        # Vest dicek terhadap SELURUH bbox person
        wearing_vest = any(
            compute_iou(v.get("bbox", []), person_bbox) >= VEST_IOU_THRESHOLD
            for v in vests
        )

        if not wearing_helmet:
            output.append({
                "label":      "no_helmet",
                "yolo_label": "no_helmet",
                "confidence": 0.90,
                "bbox":       person_bbox,
                "risk_level": "high",
                "inferred":   True,
            })
        if not wearing_vest:
            output.append({
                "label":      "no_safety_vest",
                "yolo_label": "no_safety_vest",
                "confidence": 0.90,
                "bbox":       person_bbox,
                "risk_level": "high",
                "inferred":   True,
            })

    return output


router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ── POST /inspections ──────────────────────────────────────
@router.post("/", status_code=201)
async def create_inspection(
    location: str = Form(...),
    area: Optional[str] = Form(None),
    image: UploadFile = File(...),
    current_user: User = Depends(inspector_only),
    db: Session = Depends(get_db)
):
    # Upload image ke Supabase Storage pakai supabase-py client
    # (bukan httpx manual — key format baru Supabase tidak selalu
    # bisa dipakai langsung di header Authorization: Bearer)
    image_bytes = await image.read()
    filename = f"{uuid.uuid4()}_{image.filename}"

    supabase = get_supabase()
    try:
        supabase.storage.from_("inspections").upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": image.content_type or "image/jpeg", "upsert": "true"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

    image_url = f"{SUPABASE_URL}/storage/v1/object/public/inspections/{filename}"

    # Simpan inspection ke DB
    inspection = Inspection(
        user_id=current_user.id,
        location=location,
        area=area,
        image_url=image_url,
        status="pending"
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    return {
        "inspection_id": str(inspection.id),
        "image_url": image_url,
        "status": inspection.status
    }


# ── POST /inspections/{id}/analyze ────────────────────────
@router.post("/{inspection_id}/analyze")
async def analyze_inspection(
    inspection_id: str,
    current_user: User = Depends(inspector_only),
    db: Session = Depends(get_db)
):
    # Cek inspection ada
    inspection = db.query(Inspection).filter(
        Inspection.id == inspection_id,
        Inspection.user_id == current_user.id
    ).first()

    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    if not inspection.image_url:
        raise HTTPException(status_code=400, detail="No image found for this inspection")


    # 1. YOLO detection — raw detections diterima di sini.
    try:
        detections = await call_yolo(inspection.image_url)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"AI analysis failed (YOLO/RAG service error): {str(e)}"
        )

    # 2. Inferensi pelanggaran PPE per-orang + teruskan hazard non-PPE.
    #    Deteksi mentah person/helmet/safety_vest difilter keluar di dalam
    #    infer_ppe_violations; hasilnya dipakai untuk RAG dan simpan ke DB.
    enriched_hazards = infer_ppe_violations(detections)

    ocr_text = ""

    # Helper baca label & confidence lintas format dict (mentah vs sintetis).
    def _label(d):
        return d.get("label") or d.get("yolo_label") or ""

    def _confidence(d):
        c = d.get("confidence")
        if c is None:
            c = d.get("confidence_score", 1.0)
        return c

    # 3. RAG — kirim enriched_hazards sekaligus (batch). Gagal = non-fatal.
    hazard_inputs = [
        {
            "label":            _label(d),
            "confidence_score": _confidence(d),
            "ocr_text":         ocr_text,
        }
        for d in enriched_hazards
    ]
    try:
        rag_results = await call_rag(hazard_inputs)
    except Exception:
        rag_results = []
    rag_map = {r["label"]: r for r in rag_results if isinstance(r, dict) and "label" in r}

    # 4. Simpan hazards + corrective actions dari enriched_hazards
    hazard_list = []
    for h in enriched_hazards:
        label      = _label(h)
        confidence = _confidence(h)
        severity   = get_severity(label, confidence)
        rag        = rag_map.get(label, {})
        # Utamakan risk_level dari infer_ppe_violations bila ada, else severity.
        risk_level = h.get("risk_level") or severity["risk_level"]
        action_description = rag.get("action_description", "Refer to EHSS guidelines")

        hazard = Hazard(
            inspection_id=inspection.id,
            category=label.replace("_", " ").title(),
            risk_level=risk_level,
            confidence_score=confidence,
            yolo_label=label,
            ocr_text=ocr_text,
            description=action_description
        )
        db.add(hazard)
        db.flush()

        action = CorrectiveAction(
            hazard_id=hazard.id,
            action_description=action_description,
            priority=severity["priority"],
            due_date=severity["due_date"],
            action_status="open"
        )
        db.add(action)

        hazard_list.append({
            "hazard_id": str(hazard.id),
            "category": hazard.category,
            "risk_level": hazard.risk_level,
            "confidence_score": hazard.confidence_score,
            "yolo_label": hazard.yolo_label,
            "corrective_action": {
                "action_description": action.action_description,
                "priority": action.priority,
                "due_date": str(action.due_date),
            }
        })

    # Update inspection status
    inspection.status = "analyzed"
    db.commit()

    # Notifikasi email ke semua manager/admin kalau ada hazard critical.
    # Dibungkus try/except supaya gagal kirim email tidak menggagalkan
    # response analisa yang sudah berhasil.
    critical_labels = [h["category"] for h in hazard_list if h["risk_level"] == "critical"]
    if critical_labels:
        try:
            recipients = db.query(User).filter(
                User.role.in_(["manager", "admin"]),
                User.status == "active"
            ).all()
            for recipient in recipients:
                email_service.send_critical_hazard(
                    recipient.email,
                    current_user.name,
                    inspection.location,
                    critical_labels,
                    str(inspection.id),
                )
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send critical hazard email: {e}")

    return {
        "inspection_id": str(inspection.id),
        "status": "analyzed",
        "hazards": hazard_list
    }

import httpx
YOLO_SERVICE_URL = os.getenv("YOLO_SERVICE_URL", "http://localhost:8000")

@router.post("/live-preview")
async def live_preview(
    image: UploadFile = File(...),
    current_user: User = Depends(inspector_only),
):
    image_bytes = await image.read()
    filename = f"live-preview/{current_user.id}.jpg"

    supabase = get_supabase()
    try:
        supabase.storage.from_("inspections").upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
    except Exception:
        return {"detections": []}

    image_url = f"{SUPABASE_URL}/storage/v1/object/public/inspections/{filename}"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.post(f"{YOLO_SERVICE_URL}/detect", json={"image_url": image_url})
            res.raise_for_status()
            detections = res.json().get("detections", [])
    except Exception:
        detections = []

    return {"detections": detections}
@router.get("/")
def list_inspections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Allow all authenticated users to view their own inspections"""
    # Inspector sees only their own, manager/admin see all
    if current_user.role == "inspector":
        inspections = db.query(Inspection).filter(
            Inspection.user_id == current_user.id
        ).order_by(Inspection.created_at.desc()).all()
    else:
        inspections = db.query(Inspection).order_by(Inspection.created_at.desc()).all()

    return [
        {
            "id": str(i.id),
            "location": i.location,
            "area": i.area,
            "image_url": i.image_url,
            "status": i.status,
            "inspected_at": str(i.inspected_at),
        }
        for i in inspections
    ]


# ── GET /inspections/{id} ──────────────────────────────────
@router.get("/{inspection_id}")
def get_inspection(
    inspection_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    inspection = db.query(Inspection).filter(
        Inspection.id == inspection_id
    ).first()

    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    # Inspector hanya bisa lihat milik sendiri
    if current_user.role == "inspector" and str(inspection.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    hazards = db.query(Hazard).filter(Hazard.inspection_id == inspection.id).all()
    hazard_list = []

    for h in hazards:
        actions = db.query(CorrectiveAction).filter(
            CorrectiveAction.hazard_id == h.id
        ).all()
        hazard_list.append({
            "id": str(h.id),
            "category": h.category,
            "risk_level": h.risk_level,
            "confidence_score": h.confidence_score,
            "yolo_label": h.yolo_label,
            "ocr_text": h.ocr_text,
            "corrective_actions": [
                {
                    "id": str(a.id),
                    "action_description": a.action_description,
                    "priority": a.priority,
                    "due_date": str(a.due_date),
                    "action_status": a.action_status,
                }
                for a in actions
            ]
        })

    return {
        "id": str(inspection.id),
        "location": inspection.location,
        "area": inspection.area,
        "image_url": inspection.image_url,
        "status": inspection.status,
        "inspected_at": str(inspection.inspected_at),
        "hazards": hazard_list
    }
