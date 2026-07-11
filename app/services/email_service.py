import os
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_FROM     = os.getenv("MAIL_FROM", MAIL_USERNAME)
MAIL_SERVER   = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT     = int(os.getenv("MAIL_PORT", 587))
APP_URL       = os.getenv("APP_URL", "https://safetyvision-backend-production.up.railway.app")

reset_tokens: dict = {}

def send_email(to_email: str, subject: str, html_body: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"SafetyVision EHSS <{MAIL_FROM}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False

def base_template(content: str) -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{{margin:0;padding:0;background:#F8F9FA;font-family:'Segoe UI',Arial,sans-serif;}}
.wrapper{{max-width:560px;margin:40px auto;background:#fff;border-radius:12px;border:1px solid #E5E7EB;overflow:hidden;}}
.header{{background:#E3000F;padding:28px 32px;text-align:center;}}
.header-title{{color:#fff;font-size:20px;font-weight:700;margin:0;}}
.header-sub{{color:rgba(255,255,255,0.85);font-size:13px;margin:4px 0 0;}}
.body{{padding:32px;}}
.greeting{{font-size:16px;font-weight:600;color:#111827;margin-bottom:16px;}}
.text{{font-size:14px;color:#6B7280;line-height:1.7;margin-bottom:16px;}}
.btn{{display:inline-block;background:#E3000F;color:#fff!important;text-decoration:none;padding:12px 28px;border-radius:8px;font-weight:600;font-size:14px;margin:8px 0 16px;}}
.divider{{border:none;border-top:1px solid #E5E7EB;margin:24px 0;}}
.badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}}
.badge-pending{{background:#FEF3C7;color:#92400E;}}
.badge-active{{background:#D1FAE5;color:#065F46;}}
.badge-inactive{{background:#FEE2E2;color:#991B1B;}}
.footer{{background:#F8F9FA;padding:20px 32px;text-align:center;border-top:1px solid #E5E7EB;}}
.footer-text{{font-size:12px;color:#9CA3AF;}}
</style></head><body>
<div class="wrapper">
  <div class="header">
    <div class="header-title">🛡️ SafetyVision</div>
    <div class="header-sub">Mattel EHSS · AI-powered Workplace Hazard Detection</div>
  </div>
  <div class="body">{content}</div>
  <div class="footer">
    <div class="footer-text">© 2026 Mattel, Inc. · EHSS SafetyVision<br>This is an automated message. Please do not reply.</div>
  </div>
</div></body></html>"""

def send_register_user(to_email: str, name: str, role: str) -> bool:
    role_label = {"inspector": "Safety Inspector", "manager": "EHSS Manager"}.get(role, role.title())
    content = f"""
    <div class="greeting">Hi {name},</div>
    <div class="text">Thank you for requesting access to <strong>Mattel EHSS SafetyVision</strong>. Your account is currently <strong>pending Administrator approval</strong>.</div>
    <div class="text"><strong>Email:</strong> {to_email}<br><strong>Role:</strong> <span class="badge badge-pending">{role_label.upper()}</span><br><strong>Status:</strong> <span class="badge badge-pending">PENDING APPROVAL</span></div>
    <hr class="divider">
    <div class="text">You will receive another email once your account has been reviewed. This typically takes less than 24 hours.</div>"""
    return send_email(to_email, "SafetyVision — Account Request Received", base_template(content))

def send_register_admin(admin_email: str, new_name: str, new_email: str, role: str) -> bool:
    role_label = {"inspector": "Safety Inspector", "manager": "EHSS Manager"}.get(role, role.title())
    content = f"""
    <div class="greeting">New Access Request</div>
    <div class="text">A new user has requested access to <strong>SafetyVision</strong> and is awaiting your approval.</div>
    <div class="text"><strong>Name:</strong> {new_name}<br><strong>Email:</strong> {new_email}<br><strong>Role:</strong> <span class="badge badge-pending">{role_label.upper()}</span></div>
    <hr class="divider">
    <a href="{APP_URL}/docs" class="btn">Open Admin Panel →</a>"""
    return send_email(admin_email, f"SafetyVision — New Access Request: {new_name}", base_template(content))

def send_approved(to_email: str, name: str) -> bool:
    content = f"""
    <div class="greeting">Great news, {name}! 🎉</div>
    <div class="text">Your account has been <strong>approved</strong>. Status: <span class="badge badge-active">ACTIVE</span></div>
    <a href="{APP_URL}" class="btn">Sign In to SafetyVision →</a>"""
    return send_email(to_email, "SafetyVision — Account Approved ✅", base_template(content))

def send_rejected(to_email: str, name: str) -> bool:
    content = f"""
    <div class="greeting">Hi {name},</div>
    <div class="text">Your access request has been <strong>declined</strong>. Status: <span class="badge badge-inactive">REJECTED</span></div>
    <div class="text">If you believe this is a mistake, please contact your EHSS Manager or Administrator.</div>"""
    return send_email(to_email, "SafetyVision — Access Request Update", base_template(content))

def generate_reset_token(email: str) -> str:
    token = secrets.token_urlsafe(32)
    reset_tokens[token] = {"email": email, "expires": datetime.utcnow() + timedelta(hours=1)}
    return token

def verify_reset_token(token: str) -> Optional[str]:
    data = reset_tokens.get(token)
    if not data: return None
    if datetime.utcnow() > data["expires"]:
        del reset_tokens[token]
        return None
    return data["email"]

def invalidate_reset_token(token: str):
    reset_tokens.pop(token, None)

def send_reset_password(to_email: str, name: str) -> bool:
    token = generate_reset_token(to_email)
    reset_url = f"{APP_URL}/auth/reset-password?token={token}"
    content = f"""
    <div class="greeting">Hi {name},</div>
    <div class="text">We received a request to reset your password for <strong>SafetyVision</strong>.</div>
    <a href="{reset_url}" class="btn">Reset Password →</a>
    <hr class="divider">
    <div class="text" style="font-size:13px;color:#9CA3AF;">⏱️ This link expires in <strong>1 hour</strong>. If you didn't request this, you can safely ignore this email.</div>"""
    return send_email(to_email, "SafetyVision — Password Reset Request", base_template(content))
