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
from app.services.ai_pipeline import run_full_pipeline
from app.services import email_service

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

    
    try:
        hazard_results = await run_full_pipeline(inspection.image_url)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"AI analysis failed (YOLO/RAG service error): {str(e)}"
        )

    # Simpan hazards + corrective actions
    hazard_list = []
    for h in hazard_results:
        hazard = Hazard(
            inspection_id=inspection.id,
            category=h["category"],
            risk_level=h["risk_level"],
            confidence_score=h["confidence_score"],
            yolo_label=h["yolo_label"],
            ocr_text=h["ocr_text"],
            description=h["corrective_action"]["action_description"]
        )
        db.add(hazard)
        db.flush()

        action = CorrectiveAction(
            hazard_id=hazard.id,
            action_description=h["corrective_action"]["action_description"],
            priority=h["corrective_action"]["priority"],
            due_date=h["corrective_action"]["due_date"],
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
