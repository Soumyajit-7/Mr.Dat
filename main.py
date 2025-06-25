#!/usr/bin/env python3
"""
FastAPI Backend for Mr.Dat - Data Cleaning Service
Integrates with Apache Airflow for data cleaning pipeline orchestration
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import requests
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
AIRFLOW_API_BASE = "http://localhost:8080/api/v1"
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/airflow")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class DataCleaningJob(Base):
    __tablename__ = "data_cleaning_jobs"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, running, success, failed
    dag_run_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Pipeline configuration
    max_iterations = Column(Integer, default=3)
    confidence_threshold = Column(Float, default=0.8)
    enable_auto_correction = Column(Boolean, default=True)
    schema_validation_enabled = Column(Boolean, default=True)
    outlier_detection_enabled = Column(Boolean, default=True)
    
    # Results
    original_rows = Column(Integer, nullable=True)
    original_cols = Column(Integer, nullable=True)
    final_rows = Column(Integer, nullable=True)
    final_cols = Column(Integer, nullable=True)
    data_completeness = Column(Float, nullable=True)
    issues_resolved = Column(Integer, nullable=True)
    remaining_issues = Column(Integer, nullable=True)
    output_path = Column(String, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class PipelineConfig(BaseModel):
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    enable_auto_correction: bool = True
    schema_validation_enabled: bool = True
    outlier_detection_enabled: bool = True
    
    @validator('max_iterations')
    def validate_max_iterations(cls, v):
        if v < 1 or v > 10:
            raise ValueError('max_iterations must be between 1 and 10')
        return v
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('confidence_threshold must be between 0.0 and 1.0')
        return v

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    filename: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    dag_run_id: Optional[str] = None
    
    # Configuration
    max_iterations: int
    confidence_threshold: float
    enable_auto_correction: bool
    schema_validation_enabled: bool
    outlier_detection_enabled: bool
    
    # Results
    original_rows: Optional[int] = None
    original_cols: Optional[int] = None
    final_rows: Optional[int] = None
    final_cols: Optional[int] = None
    data_completeness: Optional[float] = None
    issues_resolved: Optional[int] = None
    remaining_issues: Optional[int] = None
    output_path: Optional[str] = None

class SystemStats(BaseModel):
    total_jobs: int
    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    airflow_status: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Airflow API Client
class AirflowClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.auth = (username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
    
    def trigger_dag(self, dag_id: str, dag_run_id: str, conf: Dict = None) -> Dict:
        """Trigger a DAG run"""
        url = f"{self.base_url}/dags/{dag_id}/dagRuns"
        payload = {
            "dag_run_id": dag_run_id,
            "conf": conf or {}
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger DAG: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to trigger pipeline: {str(e)}")
    
    def get_dag_run_status(self, dag_id: str, dag_run_id: str) -> Dict:
        """Get DAG run status"""
        url = f"{self.base_url}/dags/{dag_id}/dagRuns/{dag_run_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get DAG run status: {e}")
            return {"state": "unknown"}
    
    def get_dag_status(self, dag_id: str) -> Dict:
        """Get DAG status"""
        url = f"{self.base_url}/dags/{dag_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get DAG status: {e}")
            return {"is_active": False}

# Initialize Airflow client
airflow_client = AirflowClient(AIRFLOW_API_BASE, AIRFLOW_USERNAME, AIRFLOW_PASSWORD)

# Background task to monitor jobs
async def monitor_job_status(job_id: str, dag_run_id: str, db: Session):
    """Monitor job status and update database"""
    max_attempts = 120  # 10 minutes with 5-second intervals
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Get job from database
            job = db.query(DataCleaningJob).filter(DataCleaningJob.id == job_id).first()
            if not job:
                logger.error(f"Job {job_id} not found in database")
                break
            
            # Get DAG run status from Airflow
            dag_status = airflow_client.get_dag_run_status("data_cleaning_pipeline", dag_run_id)
            airflow_state = dag_status.get("state", "unknown")
            
            # Update job status based on Airflow state
            if airflow_state == "running":
                if job.status != "running":
                    job.status = "running"
                    job.started_at = datetime.utcnow()
            elif airflow_state == "success":
                job.status = "success"
                job.completed_at = datetime.utcnow()
                # Try to get results from XCom or pipeline output
                await update_job_results(job_id, db)
                break
            elif airflow_state == "failed":
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                job.error_message = "Pipeline execution failed"
                break
            
            db.commit()
            await asyncio.sleep(5)  # Check every 5 seconds
            attempt += 1
            
        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
            break
    
    # If we've exhausted attempts and job is still running, mark as timeout
    if attempt >= max_attempts:
        job = db.query(DataCleaningJob).filter(DataCleaningJob.id == job_id).first()
        if job and job.status == "running":
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error_message = "Job monitoring timeout"
            db.commit()

async def update_job_results(job_id: str, db: Session):
    """Update job results from pipeline output"""
    try:
        job = db.query(DataCleaningJob).filter(DataCleaningJob.id == job_id).first()
        if not job:
            return
        
        # Look for output files in the output directory
        job_output_dir = OUTPUT_DIR / job_id
        if job_output_dir.exists():
            # Find the results JSON file
            result_files = list(job_output_dir.glob("pipeline_results_*.json"))
            if result_files:
                with open(result_files[0], 'r') as f:
                    results = json.load(f)
                
                # Update job with results
                audit_report = results.get('audit_report', {})
                data_transform = audit_report.get('data_transformation', {})
                quality_metrics = audit_report.get('quality_metrics', {})
                
                job.original_rows = data_transform.get('original_shape', [0, 0])[0]
                job.original_cols = data_transform.get('original_shape', [0, 0])[1]
                job.final_rows = data_transform.get('final_shape', [0, 0])[0]
                job.final_cols = data_transform.get('final_shape', [0, 0])[1]
                job.data_completeness = quality_metrics.get('data_completeness', 0.0)
                job.issues_resolved = len(audit_report.get('issues_resolved', []))
                job.remaining_issues = len(audit_report.get('remaining_issues', []))
                
                # Find cleaned data file
                data_files = list(job_output_dir.glob("cleaned_data_*.csv"))
                if data_files:
                    job.output_path = str(data_files[0])
                
                db.commit()
                
    except Exception as e:
        logger.error(f"Error updating job results for {job_id}: {e}")

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Mr.Dat Backend API")
    yield
    # Shutdown
    logger.info("Shutting down Mr.Dat Backend API")

app = FastAPI(
    title="Mr.Dat - Data Cleaning Service",
    description="Backend API for Mr.Dat data cleaning pipeline with Airflow integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Mr.Dat - Your Data Cleaning Service", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check Airflow connection
    try:
        dag_status = airflow_client.get_dag_status("data_cleaning_pipeline")
        airflow_status = "healthy" if dag_status.get("is_active") else "dag_inactive"
    except Exception as e:
        airflow_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "airflow": airflow_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/data-cleaning/upload", response_model=JobResponse)
async def upload_and_clean_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: str = Form("{}"),
    db: Session = Depends(get_db)
):
    """Upload CSV file and start data cleaning pipeline"""
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Parse configuration
    try:
        config_dict = json.loads(config) if config else {}
        pipeline_config = PipelineConfig(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    
    # Generate job ID and create job record
    job_id = str(uuid.uuid4())
    dag_run_id = f"manual__{job_id}"
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate CSV file
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        file_path.unlink(missing_ok=True)  # Clean up file
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
    
    # Create job record
    job = DataCleaningJob(
        id=job_id,
        filename=file.filename,
        status="pending",
        dag_run_id=dag_run_id,
        max_iterations=pipeline_config.max_iterations,
        confidence_threshold=pipeline_config.confidence_threshold,
        enable_auto_correction=pipeline_config.enable_auto_correction,
        schema_validation_enabled=pipeline_config.schema_validation_enabled,
        outlier_detection_enabled=pipeline_config.outlier_detection_enabled
    )
    
    db.add(job)
    db.commit()
    
    # Prepare Airflow DAG configuration
    dag_conf = {
        "job_id": job_id,
        "input_file": str(file_path),
        "output_dir": str(OUTPUT_DIR / job_id),
        "max_iterations": pipeline_config.max_iterations,
        "confidence_threshold": pipeline_config.confidence_threshold,
        "enable_auto_correction": pipeline_config.enable_auto_correction,
        "schema_validation_enabled": pipeline_config.schema_validation_enabled,
        "outlier_detection_enabled": pipeline_config.outlier_detection_enabled
    }
    
    # Create output directory
    (OUTPUT_DIR / job_id).mkdir(exist_ok=True)
    
    try:
        # Trigger Airflow DAG
        airflow_response = airflow_client.trigger_dag(
            dag_id="data_cleaning_pipeline",
            dag_run_id=dag_run_id,
            conf=dag_conf
        )
        
        # Start background monitoring
        background_tasks.add_task(monitor_job_status, job_id, dag_run_id, db)
        
        logger.info(f"Started data cleaning job {job_id} for file {file.filename}")
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Data cleaning job started successfully"
        )
        
    except Exception as e:
        # Clean up on failure
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        file_path.unlink(missing_ok=True)
        
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@app.get("/data-cleaning/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get status of a data cleaning job"""
    job = db.query(DataCleaningJob).filter(DataCleaningJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job.id,
        filename=job.filename,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        dag_run_id=job.dag_run_id,
        max_iterations=job.max_iterations,
        confidence_threshold=job.confidence_threshold,
        enable_auto_correction=job.enable_auto_correction,
        schema_validation_enabled=job.schema_validation_enabled,
        outlier_detection_enabled=job.outlier_detection_enabled,
        original_rows=job.original_rows,
        original_cols=job.original_cols,
        final_rows=job.final_rows,
        final_cols=job.final_cols,
        data_completeness=job.data_completeness,
        issues_resolved=job.issues_resolved,
        remaining_issues=job.remaining_issues,
        output_path=job.output_path
    )

@app.get("/data-cleaning/jobs")
async def list_jobs(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List data cleaning jobs with pagination"""
    query = db.query(DataCleaningJob)
    
    if status:
        query = query.filter(DataCleaningJob.status == status)
    
    jobs = query.offset(offset).limit(limit).all()
    total = query.count()
    
    return {
        "jobs": [
            {
                "job_id": job.id,
                "filename": job.filename,
                "status": job.status,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "data_completeness": job.data_completeness
            }
            for job in jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/data-cleaning/jobs/{job_id}/download")
async def download_cleaned_data(job_id: str, db: Session = Depends(get_db)):
    """Download cleaned data file"""
    job = db.query(DataCleaningJob).filter(DataCleaningJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "success":
        raise HTTPException(status_code=400, detail="Job not completed successfully")
    
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Cleaned data file not found")
    
    return FileResponse(
        job.output_path,
        filename=f"cleaned_{job.filename}",
        media_type="text/csv"
    )

@app.delete("/data-cleaning/jobs/{job_id}")
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a job and its associated files"""
    job = db.query(DataCleaningJob).filter(DataCleaningJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up files
    try:
        # Remove input file
        input_files = list(UPLOAD_DIR.glob(f"{job_id}_*"))
        for file_path in input_files:
            file_path.unlink(missing_ok=True)
        
        # Remove output directory
        output_dir = OUTPUT_DIR / job_id
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
    except Exception as e:
        logger.warning(f"Failed to clean up files for job {job_id}: {e}")
    
    # Remove from database
    db.delete(job)
    db.commit()
    
    return {"message": "Job deleted successfully"}

@app.get("/stats", response_model=SystemStats)
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_jobs = db.query(DataCleaningJob).count()
    pending_jobs = db.query(DataCleaningJob).filter(DataCleaningJob.status == "pending").count()
    running_jobs = db.query(DataCleaningJob).filter(DataCleaningJob.status == "running").count()
    completed_jobs = db.query(DataCleaningJob).filter(DataCleaningJob.status == "success").count()
    failed_jobs = db.query(DataCleaningJob).filter(DataCleaningJob.status == "failed").count()
    
    # Check Airflow status
    try:
        dag_status = airflow_client.get_dag_status("data_cleaning_pipeline")
        airflow_status = "active" if dag_status.get("is_active") else "inactive"
    except:
        airflow_status = "unavailable"
    
    return SystemStats(
        total_jobs=total_jobs,
        pending_jobs=pending_jobs,
        running_jobs=running_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        airflow_status=airflow_status
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )