import os
import uuid
import httpx
from supabase import create_client
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from pydantic import EmailStr
from app.middleware.auth import admin_only, hash_password
from app.models.user import User
from app.models.ehss_document import EhssDocument
from app.services import email_service

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
class ApproveRequest(BaseModel):
    status: str  # active / inactive

class UpdateUserRequest(BaseModel):
    employee_id: str | None = None
    department: str | None = None

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str  # inspector / manager / admin
    employee_id: str | None = None
    department: str | None = None

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
            "employee_id": u.employee_id,
            "department": u.department,
            "created_at": str(u.created_at),
        }
        for u in users
    ]

@router.post("/users", status_code=201)
def create_user(
    body: CreateUserRequest,
    current_user=Depends(admin_only),
    db: Session = Depends(get_db),
):
    """
    Admin membuat user baru langsung dalam status 'active' (tanpa perlu
    alur approval seperti registrasi publik).
    """
    if body.role not in ["inspector", "manager", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
        status="active",  # dibuat admin → langsung aktif
        employee_id=body.employee_id,
        department=body.department,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": str(user.id),
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "employee_id": user.employee_id,
        "department": user.department,
    }

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

    try:
        if body.status == "active":
            email_service.send_approved(user.email, user.name)
        elif body.status == "inactive":
            email_service.send_rejected(user.email, user.name)
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send approval status email: {e}")

    return {"message": f"User {body.status} successfully", "user_id": user_id}

@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    body: UpdateUserRequest,
    current_user = Depends(admin_only),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.employee_id is not None:
        user.employee_id = body.employee_id
    if body.department is not None:
        user.department = body.department
    db.commit()

    return {"message": "User updated successfully", "user_id": user_id}

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
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_bytes = await file.read()
        filename = f"{uuid.uuid4()}_{file.filename}"

        # Upload pakai Supabase Python client (bukan raw HTTP)
        supabase = get_supabase()
        res = supabase.storage.from_("ehss-docs").upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": "application/pdf"}
        )

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

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error uploading EHSS document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.get("/ehss-docs")
def list_ehss_docs(
    db: Session = Depends(get_db)
):
    """Allow all authenticated users to view EHSS docs, not just admin"""
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

# ── DELETE /admin/ehss-docs/{doc_id} ────────────────────────
@router.delete("/ehss-docs/{doc_id}")
def delete_ehss_doc(
    doc_id: str,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
):
    doc = db.query(EhssDocument).filter(EhssDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Hapus file dari Supabase Storage (best-effort — kegagalan hapus file
    # tidak boleh menggagalkan penghapusan record DB).
    try:
        filename = doc.file_url.split("/ehss-docs/")[-1]
        if filename:
            get_supabase().storage.from_("ehss-docs").remove([filename])
    except Exception as e:
        print(f"[STORAGE WARN] Failed to delete EHSS doc file: {e}")

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted successfully", "id": doc_id}