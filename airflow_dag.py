#!/usr/bin/env python3
"""
Apache Airflow DAG for Iterative Self-Correcting Data Cleaning Pipeline
Orchestrates the multi-agent data cleaning system with proper error handling and monitoring
Compatible with Airflow 3.0+
"""

import os
import json
import logging
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from data_cleaning_pipeline import DataCleaningOrchestrator, PipelineConfig, ValidationStatus

# Updated imports for Airflow 3.0+
POSTGRES_AVAILABLE = False
SLACK_AVAILABLE = True
import logging, sys
logging.info("PYTHONPATH for DAG parsing: %s", sys.path)
try:
    # Try to import PostgreSQL provider
    from airflow.providers.postgres.operators.postgres import PostgresOperator
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    POSTGRES_AVAILABLE = True
except ImportError:
    logging.warning("PostgreSQL provider not available - database operations will be skipped")
    
try:
    # Try to import Slack provider
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    SLACK_AVAILABLE = True
except ImportError:
    logging.warning("Slack provider not available - notifications will use logging")

# Import your data cleaning pipeline classes
# Note: Make sure the pipeline module is in your Python path

import os, sys

# 1) Compute the absolute path to this DAG file’s directory:
dag_folder = os.path.dirname(os.path.realpath(__file__))

# 2) Insert it at the front of Python’s search path:
if dag_folder not in sys.path:
    sys.path.insert(0, dag_folder)


try:
    from data_cleaning_pipeline import (
        DataCleaningOrchestrator,
        PipelineConfig,
        ValidationStatus
    )
except ImportError:
    logging.error("data_cleaning_pipeline module not found. Please ensure it's in your Python path.")
    # Create dummy classes for testing
    class DataCleaningOrchestrator:
        def __init__(self, config):
            self.config = config
        
        async def clean_data(self, data, schema):
            # Dummy implementation for testing
            return data, {
                'final_status': 'success',
                'iterations': [{'iteration': 1}],
                'audit_report': {
                    'data_transformation': {
                        'original_shape': data.shape,
                        'final_shape': data.shape
                    },
                    'quality_metrics': {
                        'data_completeness': 95.0
                    },
                    'issues_resolved': ['test_issue'],
                    'remaining_issues': []
                }
            }
    
    class PipelineConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class ValidationStatus:
        pass

# DAG Configuration
DAG_ID = "data_cleaning_pipeline"
SCHEDULE = "@daily"  # Run daily, adjust as needed
START_DATE = datetime.now() - timedelta(days=1)
CATCHUP = False

# Default arguments for all tasks
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': START_DATE,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Configuration from Airflow Variables
DATA_INPUT_PATH = Variable.get("data_cleaning_input_path", "/home/tiger/projects/data-cleaning-pipeline/data/input/", deserialize_json=False)
DATA_OUTPUT_PATH = Variable.get("data_cleaning_output_path", "/home/tiger/projects/data-cleaning-pipeline/data/output/", deserialize_json=False)
GROQ_API_KEY = Variable.get("groq_api_key", "")
SLACK_WEBHOOK_URL = Variable.get("SLACK_WEBHOOK_URL", "")
POSTGRES_CONN_ID = "postgres_default"

def get_pipeline_config(**context) -> PipelineConfig:
    """Create pipeline configuration from Airflow Variables"""
    return PipelineConfig(
        groq_api_key=GROQ_API_KEY,
        model_name=Variable.get("groq_model_name", "llama-3.3-70b-versatile", deserialize_json=False),
        max_iterations=int(Variable.get("max_iterations", "3", deserialize_json=False)),
        confidence_threshold=float(Variable.get("confidence_threshold", "0.8", deserialize_json=False)),
        batch_size=int(Variable.get("batch_size", "100", deserialize_json=False)),
        enable_auto_correction=Variable.get("enable_auto_correction", "true", deserialize_json=False).lower() == "true",
        schema_validation_enabled=Variable.get("schema_validation_enabled", "true", deserialize_json=False).lower() == "true",
        outlier_detection_enabled=Variable.get("outlier_detection_enabled", "true", deserialize_json=False).lower() == "true"
    )

def validate_inputs(**context):
    """Validate input files and configuration"""
    logging.info("Validating pipeline inputs...")
    
    # Check if input directory exists and has files
    input_path = Path(DATA_INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in input directory: {input_path}")
    
    # Validate API key
    if not GROQ_API_KEY:
        raise ValueError("GROQ API key not configured. Set 'groq_api_key' Variable in Airflow.")
    
    # Store file list in XCom for downstream tasks
    file_list = [str(f) for f in csv_files]
    context['task_instance'].xcom_push(key='input_files', value=file_list)
    
    logging.info(f"Found {len(csv_files)} CSV files to process: {file_list}")
    return file_list

def load_data(**context):
    """Load and validate input data"""
    logging.info("Loading input data...")
    
    # Get file list from XCom
    file_list = context['task_instance'].xcom_pull(key='input_files', task_ids='data_preparation.validate_inputs')
    
    if not file_list:
        raise ValueError("No input files found from validation step")
    
    # For this example, we'll process the first file
    # In production, you might want to process multiple files or combine them
    input_file = file_list[0]
    
    try:
        data = pd.read_csv(input_file)
        logging.info(f"Loaded data with shape: {data.shape}")
        
        # Basic data validation
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Store data info in XCom
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': dict(data.dtypes.astype(str)),
            'missing_values': data.isnull().sum().sum(),
            'file_path': input_file
        }
        
        context['task_instance'].xcom_push(key='data_info', value=data_info)
        context['task_instance'].xcom_push(key='raw_data_path', value=input_file)
        
        return data_info
        
    except Exception as e:
        logging.error(f"Error loading data from {input_file}: {str(e)}")
        raise

def define_schema(**context):
    """Define or load expected schema for validation"""
    logging.info("Defining data schema...")
    
    # Get data info from previous task
    data_info = context['task_instance'].xcom_pull(key='data_info', task_ids='data_preparation.load_data')
    
    # Try to load schema from file, or create a basic one
    schema_file = Path(DATA_INPUT_PATH) / "schema.json"
    
    if schema_file.exists():
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        logging.info("Loaded schema from file")
    else:
        # Create basic schema from data
        schema = {
            'columns': data_info['columns'],
            'types': {}
        }
        
        # Basic type mapping
        for col, dtype in data_info['dtypes'].items():
            if 'int' in dtype:
                schema['types'][col] = 'int'
            elif 'float' in dtype:
                schema['types'][col] = 'float'
            elif 'object' in dtype:
                schema['types'][col] = 'string'
            elif 'datetime' in dtype:
                schema['types'][col] = 'datetime'
            else:
                schema['types'][col] = 'string'
        
        logging.info("Created basic schema from data types")
    
    context['task_instance'].xcom_push(key='schema', value=schema)
    return schema

async def run_cleaning_pipeline_async(data_path: str, schema: Dict, config: PipelineConfig) -> tuple:
    """Run the async data cleaning pipeline"""
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize orchestrator
    orchestrator = DataCleaningOrchestrator(config)
    
    # Run cleaning pipeline
    cleaned_data, results = await orchestrator.clean_data(data, schema)
    
    return cleaned_data, results

def execute_pipeline(**context):
    """Execute the main data cleaning pipeline"""
    logging.info("Starting data cleaning pipeline execution...")
    
    # Get inputs from XCom
    data_path = context['task_instance'].xcom_pull(key='raw_data_path', task_ids='data_preparation.load_data')
    schema = context['task_instance'].xcom_pull(key='schema', task_ids='data_preparation.define_schema')
    
    # Get configuration
    config = get_pipeline_config(**context)
    
    try:
        # Run the async pipeline
        cleaned_data, results = asyncio.run(
            run_cleaning_pipeline_async(data_path, schema, config)
        )
        
        # Generate output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_date = context['ds']  # Airflow execution date
        
        output_dir = Path(DATA_OUTPUT_PATH) / run_date
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cleaned_data_path = output_dir / f"cleaned_data_{timestamp}.csv"
        results_path = output_dir / f"pipeline_results_{timestamp}.json"
        
        # Save results
        cleaned_data.to_csv(cleaned_data_path, index=False)
        
        # Serialize results with proper handling of complex objects
        serializable_results = json.loads(json.dumps(results, default=str))
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Store results in XCom
        pipeline_summary = {
            'final_status': results['final_status'],
            'iterations_run': len(results['iterations']),
            'cleaned_data_path': str(cleaned_data_path),
            'results_path': str(results_path),
            'original_shape': list(results['audit_report']['data_transformation']['original_shape']),
            'final_shape': list(results['audit_report']['data_transformation']['final_shape']),
            'data_completeness': results['audit_report']['quality_metrics']['data_completeness'],
            'issues_resolved': len(results['audit_report']['issues_resolved']),
            'remaining_issues': len(results['audit_report']['remaining_issues'])
        }
        
        context['task_instance'].xcom_push(key='pipeline_summary', value=pipeline_summary)
        
        logging.info(f"Pipeline completed successfully:")
        logging.info(f"- Status: {results['final_status']}")
        logging.info(f"- Iterations: {len(results['iterations'])}")
        logging.info(f"- Data completeness: {pipeline_summary['data_completeness']:.2f}%")
        logging.info(f"- Output saved to: {cleaned_data_path}")
        
        return pipeline_summary
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        # Store error info for notification
        context['task_instance'].xcom_push(key='error_info', value=str(e))
        raise

def validate_output(**context):
    """Validate pipeline output and results"""
    logging.info("Validating pipeline output...")
    
    summary = context['task_instance'].xcom_pull(key='pipeline_summary', task_ids='execute_pipeline')
    
    if not summary:
        raise ValueError("No pipeline summary found")
    
    # Validate output files exist
    cleaned_data_path = Path(summary['cleaned_data_path'])
    results_path = Path(summary['results_path'])
    
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_data_path}")
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Load and validate cleaned data
    cleaned_data = pd.read_csv(cleaned_data_path)
    
    if cleaned_data.empty:
        raise ValueError("Cleaned data is empty")
    
    # Check if pipeline improved data quality
    original_shape = summary['original_shape']
    final_shape = summary['final_shape']
    data_completeness = summary['data_completeness']
    
    validation_results = {
        'files_exist': True,
        'data_not_empty': not cleaned_data.empty,
        'data_completeness_acceptable': data_completeness >= 80,  # 80% threshold
        'pipeline_successful': summary['final_status'] in ['success', 'max_iterations_reached']
    }
    
    all_valid = all(validation_results.values())
    
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    
    if not all_valid:
        raise ValueError(f"Output validation failed: {validation_results}")
    
    logging.info("Output validation passed")
    return validation_results

def store_metadata(**context):
    """Store pipeline metadata in PostgreSQL"""
    logging.info("Storing pipeline metadata...")

    # Pull XComs
    ti = context['task_instance']
    summary = ti.xcom_pull(key='pipeline_summary', task_ids='execute_pipeline')
    validation_results = ti.xcom_pull(key='validation_results', task_ids='output_handling.validate_output')

    # Use logical_date instead of execution_date
    exec_dt = context.get('execution_date') or context['logical_date']
    run_date = context['ds']  # still available

    metadata = {
        'dag_id':         context['dag'].dag_id,
        'task_id':        context['task'].task_id,
        'execution_date': exec_dt,
        'run_date':       run_date,
        'pipeline_status':    summary['final_status'],
        'iterations_run':     summary['iterations_run'],
        'original_rows':      summary['original_shape'][0],
        'original_cols':      summary['original_shape'][1],
        'final_rows':         summary['final_shape'][0],
        'final_cols':         summary['final_shape'][1],
        'data_completeness':  summary['data_completeness'],
        'issues_resolved':    summary['issues_resolved'],
        'remaining_issues':   summary['remaining_issues'],
        'output_path':        summary['cleaned_data_path'],
        'validation_passed':  all(validation_results.values())
    }

    if POSTGRES_AVAILABLE:
        try:
            hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
            insert_sql = """
            INSERT INTO data_cleaning_pipeline_runs (
              dag_id, execution_date, run_date, pipeline_status, iterations_run,
              original_rows, original_cols, final_rows, final_cols,
              data_completeness, issues_resolved, remaining_issues,
              output_path, validation_passed, created_at
            ) VALUES (
              %(dag_id)s, %(execution_date)s, %(run_date)s, %(pipeline_status)s, %(iterations_run)s,
              %(original_rows)s, %(original_cols)s, %(final_rows)s, %(final_cols)s,
              %(data_completeness)s, %(issues_resolved)s, %(remaining_issues)s,
              %(output_path)s, %(validation_passed)s, NOW()
            )
            """
            hook.run(insert_sql, parameters=metadata)
            logging.info("Metadata stored successfully")
        except Exception as e:
            logging.warning(f"Failed to store metadata: {e}")
    else:
        logging.info(f"PostgreSQL not available—instead logging: {metadata}")

    return metadata


def create_metadata_table(**context):
    """Create PostgreSQL table if PostgreSQL is available"""
    if POSTGRES_AVAILABLE:
        try:
            postgres_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
            
            create_sql = """
            CREATE TABLE IF NOT EXISTS data_cleaning_pipeline_runs (
                id SERIAL PRIMARY KEY,
                dag_id VARCHAR(100),
                execution_date TIMESTAMP,
                run_date DATE,
                pipeline_status VARCHAR(50),
                iterations_run INTEGER,
                original_rows INTEGER,
                original_cols INTEGER,
                final_rows INTEGER,
                final_cols INTEGER,
                data_completeness DECIMAL(5,2),
                issues_resolved INTEGER,
                remaining_issues INTEGER,
                output_path TEXT,
                validation_passed BOOLEAN,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            
            postgres_hook.run(create_sql)
            logging.info("Metadata table created/verified successfully")
            
        except Exception as e:
            logging.warning(f"Failed to create metadata table: {str(e)}")
    else:
        logging.info("PostgreSQL not available - skipping table creation")
    
    return True

def create_success_notification(**context):
    """Create success notification message"""
    summary = context['task_instance'].xcom_pull(key='pipeline_summary', task_ids='execute_pipeline')
    
    message = f"""
    ✅ Data Cleaning Pipeline Completed Successfully
    
    📊 *Pipeline Summary:*
    • Status: {summary['final_status']}
    • Iterations: {summary['iterations_run']}
    • Data Completeness: {summary['data_completeness']:.2f}%
    • Issues Resolved: {summary['issues_resolved']}
    • Remaining Issues: {summary['remaining_issues']}
    
    📁 *Output:*
    • Cleaned Data: {summary['cleaned_data_path']}
    • Shape: {summary['original_shape']} → {summary['final_shape']}
    
    🕐 *Execution Date:* {context['ds']}
    """
    
    logging.info(message)
    return message

def create_failure_notification(**context):
    """Create failure notification message"""
    error_info = context['task_instance'].xcom_pull(key='error_info', task_ids='execute_pipeline')
    
    message = f"""
    ❌ Data Cleaning Pipeline Failed
    
    💥 *Error:* {error_info or 'Unknown error'}
    
    🕐 *Execution Date:* {context['ds']}
    📋 *DAG:* {context['dag'].dag_id}
    🔧 *Task:* {context['task'].task_id}
    
    Please check the logs for more details.
    """
    
    logging.error(message)
    return message

# Create the DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Iterative Self-Correcting Data Cleaning Pipeline',
    schedule=SCHEDULE,
    catchup=CATCHUP,
    max_active_runs=1,
    tags=['data-cleaning', 'ai', 'groq', 'validation']
)

# Define tasks
start_task = EmptyOperator(
    task_id='start',
    dag=dag
)

# Data preparation tasks
with TaskGroup('data_preparation', dag=dag) as data_prep_group:
    validate_inputs_task = PythonOperator(
        task_id='validate_inputs',
        python_callable=validate_inputs,
        dag=dag
    )
    
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        dag=dag
    )
    
    define_schema_task = PythonOperator(
        task_id='define_schema',
        python_callable=define_schema,
        dag=dag
    )
    
    validate_inputs_task >> load_data_task >> define_schema_task

# Create PostgreSQL table if PostgreSQL is available
create_table_task = PythonOperator(
    task_id='create_metadata_table',
    python_callable=create_metadata_table,
    dag=dag
)

# Main pipeline execution
execute_pipeline_task = PythonOperator(
    task_id='execute_pipeline',
    python_callable=execute_pipeline,
    dag=dag
)

# Output validation and storage
with TaskGroup('output_handling', dag=dag) as output_group:
    validate_output_task = PythonOperator(
        task_id='validate_output',
        python_callable=validate_output,
        dag=dag
    )
    
    store_metadata_task = PythonOperator(
        task_id='store_metadata',
        python_callable=store_metadata,
        dag=dag
    )
    
    validate_output_task >> store_metadata_task

# Notification tasks - using PythonOperator instead of SlackWebhookOperator for better compatibility
success_notification_task = PythonOperator(
    task_id='send_success_notification',
    python_callable=create_success_notification,
    dag=dag,
    trigger_rule='all_success'
)

failure_notification_task = PythonOperator(
    task_id='send_failure_notification',
    python_callable=create_failure_notification,
    dag=dag,
    trigger_rule='one_failed'
)

end_task = EmptyOperator(
    task_id='end',
    dag=dag,
    trigger_rule='none_failed_min_one_success'
)

# Define task dependencies
start_task >> create_table_task >> data_prep_group >> execute_pipeline_task >> output_group >> [success_notification_task, end_task]
execute_pipeline_task >> failure_notification_task
failure_notification_task >> end_task