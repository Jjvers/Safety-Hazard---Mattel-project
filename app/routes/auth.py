from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from app.database import get_db
from app.models.user import User
from app.middleware.auth import hash_password, verify_password, create_access_token
from app.services.email_service import (
    send_register_user, send_register_admin,
    send_reset_password, verify_reset_token, invalidate_reset_token
)

router = APIRouter()

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/register", status_code=201)
def register(body: RegisterRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    if body.role not in ["inspector", "manager", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    user = User(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
        status="pending",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Kirim email ke user baru
    background_tasks.add_task(send_register_user, body.email, body.name, body.role)

    # Kirim notifikasi ke semua Admin
    admins = db.query(User).filter(User.role == "admin", User.status == "active").all()
    for admin in admins:
        background_tasks.add_task(send_register_admin, admin.email, body.name, body.email, body.role)

    return {"user_id": str(user.id), "message": "Account created. Waiting for Admin approval."}


@router.post("/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Account pending approval")
    if user.status == "inactive":
        raise HTTPException(status_code=403, detail="Account is inactive")

    token = create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "name": user.name}


@router.post("/forgot-password")
def forgot_password(body: ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    # Selalu return 200 supaya tidak bocorkan info email terdaftar atau tidak
    if user and user.status == "active":
        background_tasks.add_task(send_reset_password, user.email, user.name)
    return {"message": "If that email is registered and active, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = verify_reset_token(body.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password_hash = hash_password(body.new_password)
    db.commit()
    invalidate_reset_token(body.token)

    return {"message": "Password reset successfully. You can now log in with your new password."}
