# 🧼 Mr.Dat — End-to-End CSV Data-Cleaning Service

Mr.Dat is an intelligent, automated data-cleaning platform that allows users to upload raw CSV datasets and receive cleaned, validated versions via a FastAPI interface. It leverages AI-powered agents, machine learning models, and Apache Airflow orchestration to deliver scalable and production-ready data pipelines.

---

## 🚀 Key Features

- **FastAPI Web Interface** for dataset upload & output download  
- **Apache Airflow DAG** to orchestrate multi-stage processing  
- **AI-Enhanced Cleaning Agents** via Groq API (async-enabled)  
- **ML-based Imputation** using RandomForest, KMeans, KNNImputer, and LabelEncoder  
- **Validation & Reporting** of missing values, schema mismatches, and outliers  
- **PostgreSQL-backed Job Tracking** using SQLAlchemy  
- **Slack Notifications** for DAG success/failure alerts  

---

## 🧩 System Architecture

### 1. **Orchestration Layer (Airflow)**
- **Scheduler**: Apache Airflow 3.0+  
- **Operators**: `PythonOperator`, `BashOperator`, `FileSensor`, `EmptyOperator`  
- **Integration**:
  - PostgreSQL via `PostgresHook` & `PostgresOperator`
  - Slack via `SlackWebhookOperator`
- **Configuration**: Airflow Variables & XCom used for passing file paths and schema metadata  

### 2. **Core Cleaning Pipeline (`data_cleaning_pipeline.py`)**
- **Async Framework**: `asyncio`-based loop for concurrent multi-agent processing  
- **AI Integration**:  
  - Groq API over `aiohttp` for:
    - Schema recommendation  
    - Outlier detection  
    - Value standardization  
- **ETL & Transform**:  
  - **Pandas**, **NumPy** for data manipulation  
  - **scikit-learn** for:
    - RandomForest-based imputation  
    - KMeans clustering & KNNImputer  
    - Label encoding for categorical fields  
- **Validation Engine**: `ValidationStatus` enums and dataclasses track each cleaning step  

### 3. **Backend API (`main.py`)**
- **Framework**: FastAPI with Uvicorn ASGI  
- **Data Models**: Pydantic schemas for job configuration and request validation  
- **Persistence**:
  - SQLAlchemy ORM over PostgreSQL  
  - Tables for `DataCleaningJob`, results, validation reports  
- **Airflow Integration**:
  - REST API client to trigger & monitor DAG runs  
  - Background async tasks poll DAG state and update DB  
- **File I/O**:
  - Handles `UploadFile` (CSV)  
  - Saves cleaned output and provides download link via `FileResponse`  

---

## ⚙️ Tech Stack

| Category             | Tools / Libraries                                  |
|----------------------|----------------------------------------------------|
| Backend API          | FastAPI, Uvicorn, Pydantic                         |
| Workflow Orchestration | Apache Airflow 3.0+                             |
| ML & Data Cleaning   | Pandas, NumPy, scikit-learn                        |
| AI Integration       | Groq API, aiohttp, asyncio                         |
| Database             | PostgreSQL, SQLAlchemy ORM                         |
| Notifications        | Slack Webhooks                                     |
| Deployment Tools     | Airflow Variables, REST API, XCom, Alembic (implied) |

---
