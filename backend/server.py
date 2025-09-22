from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import pandas as pd
import io
from enum import Enum
import jwt
import hashlib
from passlib.context import CryptContext

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Create the main app without a prefix
app = FastAPI(title="Student Management System API", version="2.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Risk Categories Enum
class RiskCategory(str, Enum):
    HIGH_RISK = "high_risk"
    MID_RISK = "mid_risk" 
    LOW_RISK = "low_risk"
    SAFE = "safe"

class UserRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"

class FeeStatus(str, Enum):
    PAID = "paid"
    PENDING = "pending"
    OVERDUE = "overdue"

# Authentication Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    role: UserRole
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: UserRole = UserRole.TEACHER

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

# Student Models
class AttendanceRecord(BaseModel):
    date: datetime
    present: bool
    remarks: Optional[str] = None

class AssessmentRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assessment_name: str
    marks_obtained: float
    total_marks: float
    percentage: float
    date: datetime
    subject: Optional[str] = None

class FeeRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fee_type: str  # tuition, library, lab, etc.
    amount: float
    due_date: datetime
    paid_date: Optional[datetime] = None
    status: FeeStatus
    payment_method: Optional[str] = None

class Student(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    class_grade: Optional[str] = None
    roll_number: Optional[str] = None
    attendance_percentage: float = Field(ge=0, le=100)
    assessment_marks: float = Field(ge=0, le=100)
    risk_category: RiskCategory
    suggestions: List[str] = []
    attendance_records: List[AttendanceRecord] = []
    assessment_records: List[AssessmentRecord] = []
    fee_records: List[FeeRecord] = []
    total_fees: float = 0.0
    paid_fees: float = 0.0
    pending_fees: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class StudentCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    class_grade: Optional[str] = None
    roll_number: Optional[str] = None
    attendance_percentage: float = Field(ge=0, le=100)
    assessment_marks: float = Field(ge=0, le=100)

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    class_grade: Optional[str] = None
    roll_number: Optional[str] = None
    attendance_percentage: Optional[float] = None
    assessment_marks: Optional[float] = None

# Other Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class RiskSummary(BaseModel):
    total_students: int
    high_risk_count: int
    mid_risk_count: int
    low_risk_count: int
    safe_count: int
    high_risk_percentage: float
    mid_risk_percentage: float
    low_risk_percentage: float
    safe_percentage: float

class UploadResponse(BaseModel):
    message: str
    students_processed: int
    students_added: int
    errors: List[str] = []

class AttendanceSummary(BaseModel):
    student_id: str
    student_name: str
    total_days: int
    present_days: int
    absent_days: int
    attendance_percentage: float
    recent_attendance: List[AttendanceRecord]

class FeeSummary(BaseModel):
    student_id: str
    student_name: str
    total_fees: float
    paid_fees: float
    pending_fees: float
    overdue_amount: float
    fee_records: List[FeeRecord]

class AssessmentSummary(BaseModel):
    student_id: str
    student_name: str
    total_assessments: int
    average_marks: float
    highest_marks: float
    lowest_marks: float
    assessment_records: List[AssessmentRecord]

# Authentication Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return User(**user)

# Helper Functions
def calculate_risk_category(attendance: float, marks: float) -> RiskCategory:
    """Calculate student risk category based on attendance and assessment marks"""
    if attendance < 75 and marks < 50:
        return RiskCategory.HIGH_RISK
    elif 75 <= attendance <= 80 and 50 <= marks <= 65:
        return RiskCategory.MID_RISK
    elif 80 <= attendance <= 90 and marks >= 66:
        return RiskCategory.LOW_RISK
    elif attendance > 90 and marks > 80:
        return RiskCategory.SAFE
    else:
        # Handle edge cases - prioritize attendance for borderline cases
        if attendance < 75:
            return RiskCategory.HIGH_RISK
        elif attendance <= 80:
            return RiskCategory.MID_RISK
        elif attendance <= 90:
            return RiskCategory.LOW_RISK
        else:
            return RiskCategory.SAFE

def generate_suggestions(risk_category: RiskCategory, attendance: float, marks: float) -> List[str]:
    """Generate improvement suggestions based on risk category"""
    suggestions = []
    
    if risk_category == RiskCategory.HIGH_RISK:
        suggestions.extend([
            "Immediate intervention required - schedule one-on-one counseling session",
            f"Attendance is critically low at {attendance:.1f}% - implement daily check-ins",
            f"Assessment performance needs significant improvement from {marks:.1f}%",
            "Consider additional tutoring support and remedial classes",
            "Develop a structured study plan with weekly milestones",
            "Engage parents/guardians for support system strengthening"
        ])
    elif risk_category == RiskCategory.MID_RISK:
        suggestions.extend([
            f"Monitor attendance closely - currently at {attendance:.1f}%",
            f"Assessment scores at {marks:.1f}% need improvement to reach safe levels",
            "Provide targeted support in weak subject areas",
            "Encourage peer study groups and collaborative learning",
            "Set up bi-weekly progress reviews",
            "Consider study skills workshops"
        ])
    elif risk_category == RiskCategory.LOW_RISK:
        suggestions.extend([
            f"Good progress! Maintain current attendance level of {attendance:.1f}%",
            f"Strong assessment performance at {marks:.1f}% - aim for excellence",
            "Focus on consistent performance across all subjects",
            "Consider advanced learning opportunities",
            "Maintain regular study routine",
            "Prepare for leadership roles and peer mentoring"
        ])
    else:  # SAFE
        suggestions.extend([
            f"Excellent performance! Attendance at {attendance:.1f}% and marks at {marks:.1f}%",
            "Continue maintaining high standards",
            "Consider advanced placement or enrichment programs", 
            "Explore leadership and mentoring opportunities",
            "Set goals for academic excellence and career preparation",
            "Share success strategies with peers"
        ])
    
    return suggestions

def generate_sample_data(student: Student) -> Student:
    """Generate sample attendance, assessment, and fee data for a student"""
    import random
    from datetime import timedelta
    
    # Generate attendance records for last 30 days
    attendance_records = []
    present_days = 0
    total_days = 30
    
    for i in range(total_days):
        date = datetime.utcnow() - timedelta(days=i)
        # Skip weekends
        if date.weekday() < 5:  # Monday to Friday
            present = random.random() < (student.attendance_percentage / 100)
            attendance_records.append(AttendanceRecord(
                date=date,
                present=present,
                remarks="Present" if present else "Absent"
            ))
            if present:
                present_days += 1
    
    # Generate assessment records
    assessment_records = []
    subjects = ["Mathematics", "Science", "English", "History", "Geography"]
    for i, subject in enumerate(subjects):
        marks = student.assessment_marks + random.uniform(-10, 10)
        marks = max(0, min(100, marks))  # Ensure between 0-100
        assessment_records.append(AssessmentRecord(
            assessment_name=f"{subject} Test {i+1}",
            marks_obtained=marks,
            total_marks=100,
            percentage=marks,
            date=datetime.utcnow() - timedelta(days=random.randint(1, 60)),
            subject=subject
        ))
    
    # Generate fee records
    fee_records = []
    fee_types = [
        ("Tuition Fee", 5000),
        ("Library Fee", 500),
        ("Lab Fee", 1000),
        ("Sports Fee", 300),
        ("Annual Fee", 2000)
    ]
    
    total_fees = 0
    paid_fees = 0
    
    for fee_type, amount in fee_types:
        total_fees += amount
        
        # 70% chance of being paid
        if random.random() < 0.7:
            status = FeeStatus.PAID
            paid_date = datetime.utcnow() - timedelta(days=random.randint(1, 30))
            paid_fees += amount
        else:
            due_date = datetime.utcnow() + timedelta(days=random.randint(-10, 30))
            if due_date < datetime.utcnow():
                status = FeeStatus.OVERDUE
            else:
                status = FeeStatus.PENDING
            paid_date = None
        
        fee_records.append(FeeRecord(
            fee_type=fee_type,
            amount=amount,
            due_date=datetime.utcnow() + timedelta(days=30),
            paid_date=paid_date,
            status=status,
            payment_method="Online" if paid_date else None
        ))
    
    student.attendance_records = attendance_records
    student.assessment_records = assessment_records
    student.fee_records = fee_records
    student.total_fees = total_fees
    student.paid_fees = paid_fees
    student.pending_fees = total_fees - paid_fees
    
    return student

# Authentication Routes
@api_router.post("/auth/register", response_model=User)
async def register(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    existing_user = await db.users.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    existing_email = await db.users.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password and create user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        role=user_data.role
    )
    
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    
    await db.users.insert_one(user_dict)
    return user

@api_router.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login user and return access token"""
    user = await db.users.find_one({"username": user_credentials.username})
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    user_obj = User(**user)
    return Token(access_token=access_token, token_type="bearer", user=user_obj)

@api_router.get("/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user

# Initialize default admin user
@api_router.post("/auth/init-admin")
async def init_admin():
    """Initialize default admin user"""
    admin_exists = await db.users.find_one({"role": "admin"})
    if admin_exists:
        return {"message": "Admin user already exists"}
    
    admin_user = User(
        username="admin",
        email="admin@school.com",
        role=UserRole.ADMIN
    )
    
    admin_dict = admin_user.dict()
    admin_dict["hashed_password"] = get_password_hash("admin123")
    
    await db.users.insert_one(admin_dict)
    return {"message": "Admin user created", "username": "admin", "password": "admin123"}

# Original routes (now protected)
@api_router.get("/")
async def root(current_user: User = Depends(get_current_user)):
    return {"message": "Student Management System API", "user": current_user.username}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate, current_user: User = Depends(get_current_user)):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks(current_user: User = Depends(get_current_user)):
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Student Management Routes (Protected)
@api_router.post("/students", response_model=Student)
async def create_student(student_data: StudentCreate, current_user: User = Depends(get_current_user)):
    """Create a single student record"""
    risk_category = calculate_risk_category(student_data.attendance_percentage, student_data.assessment_marks)
    suggestions = generate_suggestions(risk_category, student_data.attendance_percentage, student_data.assessment_marks)
    
    student = Student(
        name=student_data.name,
        email=student_data.email,
        phone=student_data.phone,
        class_grade=student_data.class_grade,
        roll_number=student_data.roll_number,
        attendance_percentage=student_data.attendance_percentage,
        assessment_marks=student_data.assessment_marks,
        risk_category=risk_category,
        suggestions=suggestions
    )
    
    # Generate sample data
    student = generate_sample_data(student)
    
    await db.students.insert_one(student.dict())
    return student

@api_router.get("/students", response_model=List[Student])
async def get_students(risk_category: Optional[RiskCategory] = None, current_user: User = Depends(get_current_user)):
    """Get all students, optionally filtered by risk category"""
    query = {}
    if risk_category:
        query["risk_category"] = risk_category.value
    
    students = await db.students.find(query).to_list(1000)
    return [Student(**student) for student in students]

@api_router.get("/students/{student_id}", response_model=Student)
async def get_student(student_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific student by ID"""
    student = await db.students.find_one({"id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return Student(**student)

@api_router.put("/students/{student_id}", response_model=Student)
async def update_student(student_id: str, student_update: StudentUpdate, current_user: User = Depends(get_current_user)):
    """Update student information"""
    student = await db.students.find_one({"id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    update_data = student_update.dict(exclude_unset=True)
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        
        # Recalculate risk if attendance or marks changed
        if "attendance_percentage" in update_data or "assessment_marks" in update_data:
            attendance = update_data.get("attendance_percentage", student["attendance_percentage"])
            marks = update_data.get("assessment_marks", student["assessment_marks"])
            update_data["risk_category"] = calculate_risk_category(attendance, marks)
            update_data["suggestions"] = generate_suggestions(update_data["risk_category"], attendance, marks)
        
        await db.students.update_one({"id": student_id}, {"$set": update_data})
        
        updated_student = await db.students.find_one({"id": student_id})
        return Student(**updated_student)
    
    return Student(**student)

@api_router.delete("/students/{student_id}")
async def delete_student(student_id: str, current_user: User = Depends(get_current_user)):
    """Delete a student record"""
    result = await db.students.delete_one({"id": student_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student deleted successfully"}

@api_router.delete("/students")
async def delete_all_students(current_user: User = Depends(get_current_user)):
    """Delete all student records"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only admin can delete all students")
    result = await db.students.delete_many({})
    return {"message": f"Deleted {result.deleted_count} students"}

# Analytics Routes
@api_router.get("/risk-summary", response_model=RiskSummary)
async def get_risk_summary(current_user: User = Depends(get_current_user)):
    """Get risk category distribution summary"""
    total_students = await db.students.count_documents({})
    
    if total_students == 0:
        return RiskSummary(
            total_students=0,
            high_risk_count=0, mid_risk_count=0, low_risk_count=0, safe_count=0,
            high_risk_percentage=0, mid_risk_percentage=0, low_risk_percentage=0, safe_percentage=0
        )
    
    pipeline = [
        {"$group": {"_id": "$risk_category", "count": {"$sum": 1}}}
    ]
    
    risk_counts = await db.students.aggregate(pipeline).to_list(10)
    counts = {risk["_id"]: risk["count"] for risk in risk_counts}
    
    high_risk_count = counts.get("high_risk", 0)
    mid_risk_count = counts.get("mid_risk", 0)
    low_risk_count = counts.get("low_risk", 0)
    safe_count = counts.get("safe", 0)
    
    return RiskSummary(
        total_students=total_students,
        high_risk_count=high_risk_count,
        mid_risk_count=mid_risk_count,
        low_risk_count=low_risk_count,
        safe_count=safe_count,
        high_risk_percentage=round((high_risk_count / total_students) * 100, 2),
        mid_risk_percentage=round((mid_risk_count / total_students) * 100, 2),
        low_risk_percentage=round((low_risk_count / total_students) * 100, 2),
        safe_percentage=round((safe_count / total_students) * 100, 2)
    )

@api_router.get("/suggestions")
async def get_overall_suggestions(current_user: User = Depends(get_current_user)):
    """Get overall improvement suggestions based on risk distribution"""
    summary = await get_risk_summary(current_user)
    suggestions = []
    
    if summary.total_students == 0:
        return {"suggestions": ["No students data available. Please upload student data to get suggestions."]}
    
    # High risk suggestions
    if summary.high_risk_count > 0:
        suggestions.append(f"🚨 URGENT: {summary.high_risk_count} students ({summary.high_risk_percentage}%) are at high risk and need immediate intervention.")
        if summary.high_risk_percentage > 20:
            suggestions.append("Consider implementing a comprehensive student support program.")
            suggestions.append("Schedule emergency parent-teacher conferences for high-risk students.")
    
    # Mid risk suggestions  
    if summary.mid_risk_count > 0:
        suggestions.append(f"⚠️ {summary.mid_risk_count} students ({summary.mid_risk_percentage}%) are at moderate risk - implement preventive measures.")
        suggestions.append("Set up peer tutoring programs to help mid-risk students.")
    
    # Low risk suggestions
    if summary.low_risk_count > 0:
        suggestions.append(f"📈 {summary.low_risk_count} students ({summary.low_risk_percentage}%) are progressing well - maintain current support.")
    
    # Safe students
    if summary.safe_count > 0:
        suggestions.append(f"✅ {summary.safe_count} students ({summary.safe_percentage}%) are performing excellently!")
        if summary.safe_percentage > 50:
            suggestions.append("Consider advanced programs and leadership opportunities for high-performing students.")
    
    # Overall suggestions based on distribution
    if summary.high_risk_percentage + summary.mid_risk_percentage > 50:
        suggestions.append("🎯 Focus on improving overall attendance and assessment strategies.")
        suggestions.append("Review curriculum difficulty and teaching methods.")
        suggestions.append("Implement early warning systems to identify at-risk students sooner.")
    
    return {"suggestions": suggestions}

# Attendance Routes
@api_router.get("/attendance", response_model=List[AttendanceSummary])
async def get_attendance_summary(current_user: User = Depends(get_current_user)):
    """Get attendance summary for all students"""
    students = await db.students.find({}).to_list(1000)
    attendance_summaries = []
    
    for student_data in students:
        student = Student(**student_data)
        total_days = len(student.attendance_records)
        present_days = sum(1 for record in student.attendance_records if record.present)
        absent_days = total_days - present_days
        
        attendance_summaries.append(AttendanceSummary(
            student_id=student.id,
            student_name=student.name,
            total_days=total_days,
            present_days=present_days,
            absent_days=absent_days,
            attendance_percentage=student.attendance_percentage,
            recent_attendance=student.attendance_records[:10]  # Last 10 records
        ))
    
    return attendance_summaries

@api_router.get("/attendance/{student_id}", response_model=AttendanceSummary)
async def get_student_attendance(student_id: str, current_user: User = Depends(get_current_user)):
    """Get detailed attendance for a specific student"""
    student = await db.students.find_one({"id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student_obj = Student(**student)
    total_days = len(student_obj.attendance_records)
    present_days = sum(1 for record in student_obj.attendance_records if record.present)
    absent_days = total_days - present_days
    
    return AttendanceSummary(
        student_id=student_obj.id,
        student_name=student_obj.name,
        total_days=total_days,
        present_days=present_days,
        absent_days=absent_days,
        attendance_percentage=student_obj.attendance_percentage,
        recent_attendance=student_obj.attendance_records
    )

# Fee Management Routes
@api_router.get("/fees", response_model=List[FeeSummary])
async def get_fee_summary(current_user: User = Depends(get_current_user)):
    """Get fee summary for all students"""
    students = await db.students.find({}).to_list(1000)
    fee_summaries = []
    
    for student_data in students:
        student = Student(**student_data)
        overdue_amount = sum(
            record.amount for record in student.fee_records 
            if record.status == FeeStatus.OVERDUE
        )
        
        fee_summaries.append(FeeSummary(
            student_id=student.id,
            student_name=student.name,
            total_fees=student.total_fees,
            paid_fees=student.paid_fees,
            pending_fees=student.pending_fees,
            overdue_amount=overdue_amount,
            fee_records=student.fee_records
        ))
    
    return fee_summaries

@api_router.get("/fees/{student_id}", response_model=FeeSummary)
async def get_student_fees(student_id: str, current_user: User = Depends(get_current_user)):
    """Get detailed fee information for a specific student"""
    student = await db.students.find_one({"id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student_obj = Student(**student)
    overdue_amount = sum(
        record.amount for record in student_obj.fee_records 
        if record.status == FeeStatus.OVERDUE
    )
    
    return FeeSummary(
        student_id=student_obj.id,
        student_name=student_obj.name,
        total_fees=student_obj.total_fees,
        paid_fees=student_obj.paid_fees,
        pending_fees=student_obj.pending_fees,
        overdue_amount=overdue_amount,
        fee_records=student_obj.fee_records
    )

# Assessment Routes
@api_router.get("/assessments", response_model=List[AssessmentSummary])
async def get_assessment_summary(current_user: User = Depends(get_current_user)):
    """Get assessment summary for all students"""
    students = await db.students.find({}).to_list(1000)
    assessment_summaries = []
    
    for student_data in students:
        student = Student(**student_data)
        if student.assessment_records:
            marks = [record.percentage for record in student.assessment_records]
            average_marks = sum(marks) / len(marks)
            highest_marks = max(marks)
            lowest_marks = min(marks)
        else:
            average_marks = student.assessment_marks
            highest_marks = student.assessment_marks
            lowest_marks = student.assessment_marks
        
        assessment_summaries.append(AssessmentSummary(
            student_id=student.id,
            student_name=student.name,
            total_assessments=len(student.assessment_records),
            average_marks=round(average_marks, 2),
            highest_marks=highest_marks,
            lowest_marks=lowest_marks,
            assessment_records=student.assessment_records
        ))
    
    return assessment_summaries

@api_router.get("/assessments/{student_id}", response_model=AssessmentSummary)
async def get_student_assessments(student_id: str, current_user: User = Depends(get_current_user)):
    """Get detailed assessment information for a specific student"""
    student = await db.students.find_one({"id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student_obj = Student(**student)
    if student_obj.assessment_records:
        marks = [record.percentage for record in student_obj.assessment_records]
        average_marks = sum(marks) / len(marks)
        highest_marks = max(marks)
        lowest_marks = min(marks)
    else:
        average_marks = student_obj.assessment_marks
        highest_marks = student_obj.assessment_marks
        lowest_marks = student_obj.assessment_marks
    
    return AssessmentSummary(
        student_id=student_obj.id,
        student_name=student_obj.name,
        total_assessments=len(student_obj.assessment_records),
        average_marks=round(average_marks, 2),
        highest_marks=highest_marks,
        lowest_marks=lowest_marks,
        assessment_records=student_obj.assessment_records
    )

# Excel Upload Route (Protected)
@api_router.post("/upload-excel", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Upload and process Excel file with student data"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    try:
        # Read Excel file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Validate required columns
        required_columns = ['Student Name', 'Attendance %', 'Assessment Marks']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}. Required columns are: {', '.join(required_columns)}"
            )
        
        students_processed = 0
        students_added = 0
        errors = []
        
        for index, row in df.iterrows():
            try:
                name = str(row['Student Name']).strip()
                attendance = float(row['Attendance %'])
                marks = float(row['Assessment Marks'])
                
                # Optional columns
                email = str(row.get('Email', '')).strip() if 'Email' in df.columns else None
                phone = str(row.get('Phone', '')).strip() if 'Phone' in df.columns else None
                class_grade = str(row.get('Class', '')).strip() if 'Class' in df.columns else None
                roll_number = str(row.get('Roll Number', '')).strip() if 'Roll Number' in df.columns else None
                
                # Validate data ranges
                if not (0 <= attendance <= 100):
                    errors.append(f"Row {index + 2}: Attendance must be between 0-100, got {attendance}")
                    continue
                
                if not (0 <= marks <= 100):
                    errors.append(f"Row {index + 2}: Assessment marks must be between 0-100, got {marks}")
                    continue
                
                if not name or name.lower() in ['nan', 'none', '']:
                    errors.append(f"Row {index + 2}: Student name is required")
                    continue
                
                # Check if student already exists
                existing_student = await db.students.find_one({"name": name})
                if existing_student:
                    errors.append(f"Row {index + 2}: Student '{name}' already exists")
                    continue
                
                # Calculate risk and create student
                risk_category = calculate_risk_category(attendance, marks)
                suggestions = generate_suggestions(risk_category, attendance, marks)
                
                student = Student(
                    name=name,
                    email=email if email and email != 'nan' else None,
                    phone=phone if phone and phone != 'nan' else None,
                    class_grade=class_grade if class_grade and class_grade != 'nan' else None,
                    roll_number=roll_number if roll_number and roll_number != 'nan' else None,
                    attendance_percentage=attendance,
                    assessment_marks=marks,
                    risk_category=risk_category,
                    suggestions=suggestions
                )
                
                # Generate sample data
                student = generate_sample_data(student)
                
                await db.students.insert_one(student.dict())
                students_added += 1
                
            except Exception as e:
                errors.append(f"Row {index + 2}: Error processing data - {str(e)}")
            
            students_processed += 1
        
        return UploadResponse(
            message=f"Processed {students_processed} rows, added {students_added} students",
            students_processed=students_processed,
            students_added=students_added,
            errors=errors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()