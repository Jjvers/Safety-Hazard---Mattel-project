import os
import uuid
import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.middleware.auth import admin_only
from app.models.user import User
from app.models.ehss_document import EhssDocument

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

class ApproveRequest(BaseModel):
    status: str  # active / inactive

@router.get("/users")
def list_users(current_user = Depends(admin_only), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "id": str(u.id),
            "name": u.name,
            "email": u.email,
            "role": u.role,
            "status": u.status,
            "created_at": str(u.created_at),
        }
        for u in users
    ]

@router.patch("/users/{user_id}/approve")
def approve_user(
    user_id: str,
    body: ApproveRequest,
    current_user = Depends(admin_only),
    db: Session = Depends(get_db)
):
    if body.status not in ["active", "inactive"]:
        raise HTTPException(status_code=400, detail="Status must be active or inactive")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.status = body.status
    db.commit()

    return {"message": f"User {body.status} successfully", "user_id": user_id}

@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    current_user = Depends(admin_only),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

# ── POST /admin/ehss-docs ───────────────────────────────────
@router.post("/ehss-docs", status_code=201)
async def upload_ehss_doc(
    title: str = Form(...),
    category: str = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db)
):
    file_bytes = await file.read()
    filename = f"{uuid.uuid4()}_{file.filename}"

    async with httpx.AsyncClient() as client:
        upload_res = await client.post(
            f"{SUPABASE_URL}/storage/v1/object/ehss-docs/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": file.content_type,
            },
            content=file_bytes
        )

    if upload_res.status_code not in [200, 201]:
        raise HTTPException(status_code=500, detail="Failed to upload document")

    file_url = f"{SUPABASE_URL}/storage/v1/object/public/ehss-docs/{filename}"

    doc = EhssDocument(
        title=title,
        file_url=file_url,
        category=category,
        uploaded_by=current_user.id
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "id": str(doc.id),
        "title": doc.title,
        "file_url": doc.file_url,
        "category": doc.category,
        "uploaded_at": str(doc.uploaded_at),
    }


@router.get("/ehss-docs")
def list_ehss_docs(
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db)
):
    docs = db.query(EhssDocument).order_by(EhssDocument.uploaded_at.desc()).all()
    return [
        {
            "id": str(d.id),
            "title": d.title,
            "file_url": d.file_url,
            "category": d.category,
            "uploaded_at": str(d.uploaded_at),
        }
        for d in docs
    ]
