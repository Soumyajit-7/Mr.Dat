#!/usr/bin/env python3
"""
Enhanced Iterative Self-Correcting Data Cleaning Pipeline
Multi-agent system with comprehensive data validation, cleaning, and intelligent imputation
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import aiohttp
from pathlib import Path
import traceback
import re
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning_pipeline.log'),
        logging.StreamHandler()
    ]
)

class ValidationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"

@dataclass
class ValidationResult:
    status: ValidationStatus
    issues: List[str]
    corrections: List[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class PipelineConfig:
    groq_api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    batch_size: int = 100
    enable_auto_correction: bool = True
    schema_validation_enabled: bool = True
    outlier_detection_enabled: bool = True
    categorical_cleaning_enabled: bool = True
    numerical_cleaning_enabled: bool = True
    datetime_cleaning_enabled: bool = True
    advanced_imputation_enabled: bool = True

class GroqClient:
    """Async client for Groq API interactions"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_completion(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Generate completion using Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,  # Reduced for more focused responses
            "response_format": {"type": "text"}  # Ensure text response
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            try:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Groq API error: {response.status} - {error_text}")
                        raise Exception(f"Groq API error: {response.status}")
            except asyncio.TimeoutError:
                self.logger.error("Groq API request timed out")
                raise Exception("API request timed out")
            except Exception as e:
                self.logger.error(f"Error calling Groq API: {str(e)}")
                raise

class CategoricalCleaningAgent:
    """Agent for cleaning categorical data - inconsistencies, outliers, standardization"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def clean_categorical_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Comprehensive categorical data cleaning"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return data, ValidationResult(
                status=ValidationStatus.PASS,
                issues=[],
                corrections=[],
                confidence=1.0,
                metadata={"categorical_columns": 0}
            )
        
        cleaned_data = data.copy()
        issues = []
        corrections = []
        
        for col in categorical_cols:
            self.logger.info(f"Cleaning categorical column: {col}")
            
            # Get unique values analysis
            unique_values = cleaned_data[col].dropna().unique()
            unique_count = len(unique_values)
            
            if unique_count > 0:
                # Detect and clean outlier values
                outlier_values = await self._detect_categorical_outliers(col, unique_values)
                if outlier_values:
                    issues.append(f"Column '{col}': Found outlier values: {outlier_values}")
                    cleaned_data = self._remove_categorical_outliers(cleaned_data, col, outlier_values)
                    corrections.append(f"Removed outlier values from column '{col}': {outlier_values}")
                
                # Standardize similar values (case sensitivity, spelling variations)
                standardization_map = await self._generate_standardization_map(col, unique_values)
                if standardization_map:
                    issues.append(f"Column '{col}': Found inconsistent values requiring standardization")
                    cleaned_data = self._apply_standardization(cleaned_data, col, standardization_map)
                    corrections.append(f"Standardized values in column '{col}': {len(standardization_map)} mappings applied")
        
        status = ValidationStatus.PASS if not issues else ValidationStatus.PARTIAL
        confidence = 0.8 if corrections else 1.0
        
        return cleaned_data, ValidationResult(
            status=status,
            issues=issues,
            corrections=corrections,
            confidence=confidence,
            metadata={"categorical_columns_processed": len(categorical_cols)}
        )
    
    async def _detect_categorical_outliers(self, col_name: str, unique_values: np.ndarray) -> List[str]:
        """Detect outlier values in categorical data using AI"""
        if len(unique_values) == 0:
            return []
        
        # Common outlier patterns
        outlier_patterns = [
            r'(?i)^(nan|null|none|n/a|na|unknown|error|undefined|missing|blank|empty|\s*|\?)$',
            r'(?i)^(not\s+available|not\s+specified|not\s+found|invalid|corrupt)$',
            r'^[\s\-_\.]+$',  # Only special characters
            r'^[\d\.\-\+\s]*$' if not any(str(v).replace('.', '').replace('-', '').replace('+', '').replace(' ', '').isdigit() for v in unique_values[:5]) else r'^$'  # Numbers in non-numeric column
        ]
        
        outliers = []
        for value in unique_values:
            value_str = str(value).strip()
            for pattern in outlier_patterns:
                if re.match(pattern, value_str):
                    outliers.append(value)
                    break
        
        # Use AI for additional detection
        ai_outliers = await self._ai_detect_outliers(col_name, unique_values)
        outliers.extend(ai_outliers)
        
        return list(set(outliers))
    
    async def _ai_detect_outliers(self, col_name: str, unique_values: np.ndarray) -> List[str]:
        """Use AI to detect additional outliers"""
        if len(unique_values) > 50:  # Limit for API call
            unique_values = unique_values[:50]
        
        prompt = f"""
        Analyze the following unique values from column '{col_name}' and identify any outliers or invalid entries:
        
        Unique Values: {list(unique_values)}
        
        Look for:
        1. Invalid/error values (like "Error", "Invalid", "N/A", etc.)
        2. Inconsistent formats
        3. Values that don't belong in this category
        4. Obvious data entry errors
        
        Return ONLY a valid JSON object:
        {{
            "outliers": ["value1", "value2"]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.groq_client.generate_completion(messages)
            
            # More robust JSON extraction
            patterns = [
                r'\{[^{}]*"outliers"[^{}]*\[[^\]]*\][^{}]*\}',  # Find outliers array
                r'\{.*?"outliers".*?\}',  # Broader search
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group())
                        if "outliers" in result and isinstance(result["outliers"], list):
                            return result["outliers"]
                    except json.JSONDecodeError:
                        continue
            
            self.logger.warning(f"Could not parse AI outlier detection response: {response[:200]}...")
            return []
            
        except Exception as e:
            self.logger.error(f"Error in AI outlier detection: {e}")
            return []

    
    async def _generate_standardization_map(self, col_name: str, unique_values: np.ndarray) -> Dict[str, str]:
        """Generate standardization mapping for similar values"""
        if len(unique_values) <= 1:
            return {}
        
        # Basic standardization patterns
        standardization_map = {}
        
        # Group similar values (case-insensitive)
        value_groups = {}
        for value in unique_values:
            if pd.isna(value):
                continue
            key = str(value).lower().strip()
            if key not in value_groups:
                value_groups[key] = []
            value_groups[key].append(str(value))
        
        # Find groups with variations
        for key, variations in value_groups.items():
            if len(variations) > 1:
                # Choose the most common format or the first alphabetically
                standard_value = max(variations, key=lambda x: (variations.count(x), -len(x)))
                for variation in variations:
                    if variation != standard_value:
                        standardization_map[variation] = standard_value
        
        # Use AI for more complex standardization
        ai_mapping = await self._ai_generate_standardization(col_name, unique_values)
        standardization_map.update(ai_mapping)
        
        return standardization_map
    
    async def _ai_generate_standardization(self, col_name: str, unique_values: np.ndarray) -> Dict[str, str]:
        """Use AI to generate standardization mapping"""
        if len(unique_values) > 30:  # Limit for API call
            unique_values = unique_values[:30]
        
        prompt = f"""
        Analyze these values from column '{col_name}' and create a standardization mapping for similar values:
        
        Values: {list(unique_values)}
        
        Look for:
        1. Case variations (India vs india vs INDIA)
        2. Spelling variations (US vs USA vs United States)
        3. Format inconsistencies (New York vs New-York vs NY)
        4. Abbreviations vs full forms
        
        Create a mapping to standardize to the most appropriate canonical form.
        
        Return ONLY a valid JSON object in this exact format:
        {{
            "mappings": {{
                "original_value": "standardized_value"
            }}
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.groq_client.generate_completion(messages)
            
            # More robust JSON extraction
            json_match = None
            
            # Try multiple patterns to find JSON
            patterns = [
                r'\{[^{}]*"mappings"[^{}]*\{[^{}]*\}[^{}]*\}',  # Find mappings object
                r'\{.*?"mappings".*?\}',  # Broader search
                r'\{[^}]*\}',  # Any JSON object
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        result = json.loads(json_str)
                        if "mappings" in result:
                            return result.get("mappings", {})
                        json_match = match
                        break
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON found, return empty dict
            self.logger.warning(f"Could not parse AI response for standardization: {response[:200]}...")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error in AI standardization: {e}")
            return {}
    
    def _remove_categorical_outliers(self, data: pd.DataFrame, col: str, outliers: List[str]) -> pd.DataFrame:
        """Remove outlier values from categorical column"""
        cleaned_data = data.copy()
        outlier_mask = cleaned_data[col].isin(outliers)
        cleaned_data.loc[outlier_mask, col] = np.nan
        return cleaned_data
    
    def _apply_standardization(self, data: pd.DataFrame, col: str, mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply standardization mapping to column"""
        cleaned_data = data.copy()
        cleaned_data[col] = cleaned_data[col].replace(mapping)
        return cleaned_data

class NumericalCleaningAgent:
    """Agent for cleaning numerical data - non-numeric values, format issues"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def clean_numerical_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Clean numerical columns from non-numeric values"""
        cleaned_data = data.copy()
        issues = []
        corrections = []
        
        # Identify potential numerical columns (including mixed-type columns)
        potential_numeric_cols = []
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                potential_numeric_cols.append(col)
            else:
                # Check if object column contains mostly numeric values
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    numeric_count = sum(1 for val in non_null_values if self._is_convertible_to_numeric(str(val)))
                    if numeric_count / len(non_null_values) > 0.7:  # 70% threshold
                        potential_numeric_cols.append(col)
        
        for col in potential_numeric_cols:
            self.logger.info(f"Cleaning numerical column: {col}")
            
            # Detect non-numeric values
            non_numeric_values = await self._detect_non_numeric_values(col, cleaned_data[col])
            
            if non_numeric_values:
                issues.append(f"Column '{col}': Found non-numeric values: {non_numeric_values}")
                
                # Remove non-numeric values
                cleaned_data = self._clean_non_numeric_values(cleaned_data, col, non_numeric_values)
                
                # Convert to appropriate numeric type
                cleaned_data = self._convert_to_numeric(cleaned_data, col)
                
                corrections.append(f"Cleaned and converted column '{col}' to numeric type")
        
        status = ValidationStatus.PASS if not issues else ValidationStatus.PARTIAL
        confidence = 0.8 if corrections else 1.0
        
        return cleaned_data, ValidationResult(
            status=status,
            issues=issues,
            corrections=corrections,
            confidence=confidence,
            metadata={"numeric_columns_processed": len(potential_numeric_cols)}
        )
    
    def _is_convertible_to_numeric(self, value: str) -> bool:
        """Check if a string value can be converted to numeric"""
        try:
            # Remove common non-numeric characters and try conversion
            cleaned_value = re.sub(r'[,$%\s]', '', str(value))
            float(cleaned_value)
            return True
        except (ValueError, TypeError):
            return False
    
    async def _detect_non_numeric_values(self, col_name: str, series: pd.Series) -> List[str]:
        """Detect non-numeric values in a potentially numeric column"""
        non_numeric = []
        
        for value in series.dropna().unique():
            if not self._is_convertible_to_numeric(str(value)):
                non_numeric.append(str(value))
        
        # Common non-numeric patterns in numeric columns
        outlier_patterns = [
            r'(?i)^(nan|null|none|n/a|na|unknown|error|undefined|missing|blank|empty|\s*|\?|-)$',
            r'(?i)^(not\s+available|not\s+specified|not\s+found|invalid|corrupt)$',
            r'^[a-zA-Z]+$',  # Pure text
        ]
        
        for value in series.dropna().unique():
            value_str = str(value).strip()
            for pattern in outlier_patterns:
                if re.match(pattern, value_str) and value_str not in non_numeric:
                    non_numeric.append(value_str)
                    break
        
        return non_numeric
    
    def _clean_non_numeric_values(self, data: pd.DataFrame, col: str, non_numeric_values: List[str]) -> pd.DataFrame:
        """Remove non-numeric values from column"""
        cleaned_data = data.copy()
        
        # Replace non-numeric values with NaN
        mask = cleaned_data[col].astype(str).isin(non_numeric_values)
        cleaned_data.loc[mask, col] = np.nan
        
        return cleaned_data
    
    def _convert_to_numeric(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Convert column to appropriate numeric type"""
        cleaned_data = data.copy()
        
        try:
            # Try to convert to numeric
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Determine if it should be int or float
            non_null_values = cleaned_data[col].dropna()
            if len(non_null_values) > 0:
                if all(val.is_integer() for val in non_null_values):
                    cleaned_data[col] = cleaned_data[col].astype('Int64')  # Nullable integer
                else:
                    cleaned_data[col] = cleaned_data[col].astype('float64')
                    
        except Exception as e:
            self.logger.error(f"Error converting column {col} to numeric: {e}")
        
        return cleaned_data

class DateTimeCleaningAgent:
    """Agent for cleaning datetime data"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def clean_datetime_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Clean and standardize datetime columns"""
        cleaned_data = data.copy()
        issues = []
        corrections = []
        
        # Identify potential datetime columns
        potential_datetime_cols = []
        
        for col in data.columns:
            if data[col].dtype == 'datetime64[ns]':
                potential_datetime_cols.append(col)
            else:
                # Check if column contains datetime-like strings
                sample_values = data[col].dropna().head(10)
                if len(sample_values) > 0:
                    datetime_count = sum(1 for val in sample_values if self._is_datetime_like(str(val)))
                    if datetime_count / len(sample_values) > 0.7:
                        potential_datetime_cols.append(col)
        
        for col in potential_datetime_cols:
            self.logger.info(f"Cleaning datetime column: {col}")
            
            # Detect invalid datetime values
            invalid_values = self._detect_invalid_datetime_values(cleaned_data[col])
            
            if invalid_values:
                issues.append(f"Column '{col}': Found invalid datetime values: {invalid_values}")
                
                # Clean invalid values
                cleaned_data = self._clean_invalid_datetime_values(cleaned_data, col, invalid_values)
                
                # Standardize datetime format
                cleaned_data = self._standardize_datetime_format(cleaned_data, col)
                
                corrections.append(f"Cleaned and standardized datetime column '{col}'")
        
        status = ValidationStatus.PASS if not issues else ValidationStatus.PARTIAL
        confidence = 0.8 if corrections else 1.0
        
        return cleaned_data, ValidationResult(
            status=status,
            issues=issues,
            corrections=corrections,
            confidence=confidence,
            metadata={"datetime_columns_processed": len(potential_datetime_cols)}
        )
    
    def _is_datetime_like(self, value: str) -> bool:
        """Check if a string looks like a datetime"""
        if pd.isna(value) or str(value).strip() == '':
            return False
            
        # Common invalid patterns first
        invalid_patterns = [
            r'(?i)^(nan|null|none|n/a|na|unknown|error|undefined|missing|blank|empty|\s*|\?|-)$',
            r'(?i)^(not\s+available|invalid|corrupt)$',
        ]
        
        value_str = str(value).strip()
        for pattern in invalid_patterns:
            if re.match(pattern, value_str):
                return False
        
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # DD Mon
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',  # Mon DD
        ]
        
        for pattern in datetime_patterns:
            if re.search(pattern, value_str, re.IGNORECASE):
                # Additional validation - try pandas conversion
                try:
                    pd.to_datetime(value_str, errors='raise')
                    return True
                except:
                    continue
        
        return False

    
    def _detect_invalid_datetime_values(self, series: pd.Series) -> List[str]:
        """Detect invalid datetime values"""
        invalid_values = []
        
        for value in series.dropna().unique():
            if not self._is_datetime_like(str(value)):
                # Check for common invalid patterns
                value_str = str(value).strip()
                invalid_patterns = [
                    r'(?i)^(nan|null|none|n/a|na|unknown|error|undefined|missing|blank|empty|\s*|\?)$',
                    r'(?i)^(not\s+available|invalid|corrupt)$',
                ]
                
                for pattern in invalid_patterns:
                    if re.match(pattern, value_str):
                        invalid_values.append(value_str)
                        break
                else:
                    # If it doesn't match invalid patterns but still not datetime-like, it's invalid
                    invalid_values.append(value_str)
        
        return invalid_values
    
    def _clean_invalid_datetime_values(self, data: pd.DataFrame, col: str, invalid_values: List[str]) -> pd.DataFrame:
        """Clean invalid datetime values"""
        cleaned_data = data.copy()
        
        # Replace invalid values with NaN
        mask = cleaned_data[col].astype(str).isin(invalid_values)
        cleaned_data.loc[mask, col] = np.nan
        
        return cleaned_data
    
    def _standardize_datetime_format(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Standardize datetime format"""
        cleaned_data = data.copy()
        
        try:
            # Convert to datetime
            cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce', infer_datetime_format=True)
        except Exception as e:
            self.logger.error(f"Error standardizing datetime column {col}: {e}")
        
        return cleaned_data

class AdvancedImputationAgent:
    """Agent for advanced missing value imputation using ML techniques"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def impute_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Advanced imputation using multiple techniques"""
        if data.isnull().sum().sum() == 0:
            return data, ValidationResult(
                status=ValidationStatus.PASS,
                issues=[],
                corrections=[],
                confidence=1.0,
                metadata={"missing_values": 0}
            )
        
        cleaned_data = data.copy()
        issues = []
        corrections = []
        
        # Separate numerical and categorical columns
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns
        
        # Impute numerical columns
        if len(numerical_cols) > 0:
            cleaned_data, num_corrections = await self._impute_numerical_columns(cleaned_data, numerical_cols)
            corrections.extend(num_corrections)
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            cleaned_data, cat_corrections = await self._impute_categorical_columns(cleaned_data, categorical_cols)
            corrections.extend(cat_corrections)
        
        # Final check for remaining missing values
        remaining_missing = cleaned_data.isnull().sum().sum()
        if remaining_missing > 0:
            issues.append(f"Still {remaining_missing} missing values after imputation")
        
        status = ValidationStatus.PASS if remaining_missing == 0 else ValidationStatus.PARTIAL
        confidence = 0.9 if corrections else 1.0
        
        return cleaned_data, ValidationResult(
            status=status,
            issues=issues,
            corrections=corrections,
            confidence=confidence,
            metadata={"imputation_methods_used": len(corrections)}
        )
    
    async def _impute_numerical_columns(self, data: pd.DataFrame, numerical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Impute numerical columns using various methods"""
        cleaned_data = data.copy()
        corrections = []
        
        for col in numerical_cols:
            if cleaned_data[col].isnull().any():
                missing_count = cleaned_data[col].isnull().sum()
                total_count = len(cleaned_data)
                missing_percentage = (missing_count / total_count) * 100
                
                # Choose imputation method based on missing percentage and data characteristics
                if missing_percentage < 5:
                    # Use simple statistical methods for low missing percentage
                    if cleaned_data[col].skew() > 1:  # Highly skewed
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                        corrections.append(f"Imputed {col} using median (low missing %, skewed data)")
                    else:
                        cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
                        corrections.append(f"Imputed {col} using mean (low missing %, normal distribution)")
                
                elif missing_percentage < 20:
                    # Use KNN imputation for moderate missing percentage
                    try:
                        other_numeric_cols = [c for c in numerical_cols if c != col and not cleaned_data[c].isnull().all()]
                        if len(other_numeric_cols) >= 2:
                            # Use KNN imputation with other numerical columns
                            imputer_data = cleaned_data[[col] + other_numeric_cols].copy()
                            imputer = KNNImputer(n_neighbors=min(5, len(imputer_data.dropna())))
                            imputed_values = imputer.fit_transform(imputer_data)
                            cleaned_data[col] = imputed_values[:, 0]
                            corrections.append(f"Imputed {col} using KNN with {len(other_numeric_cols)} features")
                        else:
                            # Fallback to median
                            cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                            corrections.append(f"Imputed {col} using median (insufficient features for KNN)")
                    except Exception as e:
                        self.logger.error(f"KNN imputation failed for {col}: {e}")
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                        corrections.append(f"Imputed {col} using median (KNN failed)")
                
                else:
                    # Use Random Forest for high missing percentage
                    try:
                        cleaned_data, rf_correction = self._rf_impute_numerical(cleaned_data, col, numerical_cols)
                        corrections.append(rf_correction)
                    except Exception as e:
                        self.logger.error(f"RF imputation failed for {col}: {e}")
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                        corrections.append(f"Imputed {col} using median (RF failed)")
        
        return cleaned_data, corrections
    
    def _rf_impute_numerical(self, data: pd.DataFrame, target_col: str, numerical_cols: List[str]) -> Tuple[pd.DataFrame, str]:
        """Impute numerical column using Random Forest"""
        cleaned_data = data.copy()
        
        # Prepare features (other numerical columns without too many missing values)
        feature_cols = [col for col in numerical_cols if col != target_col and 
                       cleaned_data[col].isnull().sum() / len(cleaned_data) < 0.5]
        
        if len(feature_cols) < 2:
            raise ValueError("Insufficient features for Random Forest imputation")
        
        # Get complete cases for training
        complete_mask = cleaned_data[feature_cols + [target_col]].notna().all(axis=1)
        complete_data = cleaned_data[complete_mask]
        
        if len(complete_data) < 10:
            raise ValueError("Insufficient complete cases for training")
        
        # Train Random Forest
        X = complete_data[feature_cols]
        y = complete_data[target_col]
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Predict missing values
        missing_mask = cleaned_data[target_col].isnull()
        if missing_mask.any():
            X_missing = cleaned_data.loc[missing_mask, feature_cols]
            if not X_missing.isnull().any().any():  # Only if no missing values in features
                predictions = rf.predict(X_missing)
                cleaned_data.loc[missing_mask, target_col] = predictions
                return cleaned_data, f"Imputed {target_col} using Random Forest with {len(feature_cols)} features"
        
        raise ValueError("Cannot predict due to missing values in features")
    
    async def _impute_categorical_columns(self, data: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Impute categorical columns using various methods"""
        cleaned_data = data.copy()
        corrections = []
        
        for col in categorical_cols:
            if cleaned_data[col].isnull().any():
                missing_count = cleaned_data[col].isnull().sum()
                total_count = len(cleaned_data)
                missing_percentage = (missing_count / total_count) * 100
                
                if missing_percentage < 10:
                    # Use mode for low missing percentage
                    mode_value = cleaned_data[col].mode()
                    if len(mode_value) > 0:
                        cleaned_data[col].fillna(mode_value[0], inplace=True)
                        corrections.append(f"Imputed {col} using mode (low missing %)")
                
                else:
                    # Use Random Forest for higher missing percentage
                    try:
                        cleaned_data, rf_correction = self._rf_impute_categorical(cleaned_data, col, categorical_cols)
                        corrections.append(rf_correction)
                    except Exception as e:
                        self.logger.error(f"RF imputation failed for {col}: {e}")
                        # Fallback to mode or create "Unknown" category
                        mode_value = cleaned_data[col].mode()
                        if len(mode_value) > 0:
                            cleaned_data[col].fillna(mode_value[0], inplace=True)
                            corrections.append(f"Imputed {col} using mode (RF failed)")
                        else:
                            cleaned_data[col].fillna("Unknown", inplace=True)
                            corrections.append(f"Imputed {col} with 'Unknown' category (no mode available)")
        
        return cleaned_data, corrections
    
    def _rf_impute_categorical(self, data: pd.DataFrame, target_col: str, categorical_cols: List[str]) -> Tuple[pd.DataFrame, str]:
        """Impute categorical column using Random Forest"""
        cleaned_data = data.copy()
        
        # Prepare features (other categorical columns + numerical columns)
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in categorical_cols + numerical_cols if col != target_col and 
                       cleaned_data[col].isnull().sum() / len(cleaned_data) < 0.5]
        
        if len(feature_cols) < 2:
            raise ValueError("Insufficient features for Random Forest imputation")
        
        # Get complete cases for training
        complete_mask = cleaned_data[feature_cols + [target_col]].notna().all(axis=1)
        complete_data = cleaned_data[complete_mask]
        
        if len(complete_data) < 10:
            raise ValueError("Insufficient complete cases for training")
        
        # Encode categorical features
        encoders = {}
        X_encoded = complete_data[feature_cols].copy()
        
        for col in feature_cols:
            if cleaned_data[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(complete_data[col].astype(str))
                encoders[col] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(complete_data[target_col].astype(str))
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_encoded, y_encoded)
        
        # Predict missing values
        missing_mask = cleaned_data[target_col].isnull()
        if missing_mask.any():
            X_missing = cleaned_data.loc[missing_mask, feature_cols].copy()
            
            # Encode missing data features
            for col in feature_cols:
                if col in encoders:
                    # Handle unseen categories
                    try:
                        X_missing[col] = encoders[col].transform(X_missing[col].astype(str))
                    except ValueError:
                        # Handle unseen categories by using most frequent class
                        most_frequent = encoders[col].classes_[0]
                        X_missing[col] = X_missing[col].fillna(most_frequent).astype(str)
                        X_missing[col] = encoders[col].transform(X_missing[col])
            
            if not X_missing.isnull().any().any():  # Only if no missing values in features
                predictions_encoded = rf.predict(X_missing)
                predictions = target_encoder.inverse_transform(predictions_encoded)
                cleaned_data.loc[missing_mask, target_col] = predictions
                return cleaned_data, f"Imputed {target_col} using Random Forest with {len(feature_cols)} features"
        
        raise ValueError("Cannot predict due to missing values in features")

class SchemaValidationAgent:
    """Agent responsible for schema validation and correction"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def validate_schema(self, data: pd.DataFrame, expected_schema: Dict) -> ValidationResult:
        """Validate data against expected schema"""
        issues = []
        corrections = []
        
        # Check columns
        expected_cols = set(expected_schema.get('columns', []))
        actual_cols = set(data.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            issues.append(f"Missing columns: {list(missing_cols)}")
        
        if extra_cols:
            issues.append(f"Unexpected columns: {list(extra_cols)}")
        
        # Check data types
        type_issues = []
        for col, expected_type in expected_schema.get('types', {}).items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if not self._type_compatible(actual_type, expected_type):
                    type_issues.append(f"Column '{col}': expected {expected_type}, got {actual_type}")
        
        if type_issues:
            issues.extend(type_issues)
        
        # Generate AI-based corrections if issues found
        if issues:
            corrections = await self._generate_schema_corrections(data, expected_schema, issues)
        
        status = ValidationStatus.PASS if not issues else ValidationStatus.FAIL
        confidence = 1.0 if not issues else 0.5
        
        return ValidationResult(
            status=status,
            issues=issues,
            corrections=corrections,
            confidence=confidence,
            metadata={"schema_check": True}
        )
    
    def _type_compatible(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        type_mapping = {
            'int64': ['int', 'integer', 'numeric'],
            'Int64': ['int', 'integer', 'numeric'],  # Nullable integer
            'float64': ['float', 'numeric', 'decimal'],
            'object': ['str', 'string', 'text'],
            'datetime64[ns]': ['datetime', 'date', 'timestamp'],
            'bool': ['boolean', 'bool']
        }
        
        expected_lower = expected.lower()
        return expected_lower in type_mapping.get(actual, []) or actual == expected
    
    async def _generate_schema_corrections(self, data: pd.DataFrame, schema: Dict, issues: List[str]) -> List[str]:
        """Generate schema corrections using AI"""
        prompt = f"""
        Analyze the following data schema issues and provide specific correction steps:
        
        Expected Schema: {json.dumps(schema, indent=2)}
        Data Shape: {data.shape}
        Data Columns: {list(data.columns)}
        Data Types: {dict(data.dtypes.astype(str))}
        
        Issues Found:
        {chr(10).join(f"- {issue}" for issue in issues)}
        
        Provide specific, actionable correction steps in JSON format:
        {{
            "corrections": [
                "Step 1: ...",
                "Step 2: ..."
            ]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.groq_client.generate_completion(messages)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("corrections", [])
            return ["Manual schema correction required"]
        except Exception as e:
            self.logger.error(f"Error generating schema corrections: {e}")
            return ["Failed to generate automatic corrections"]

class OutlierDetectionAgent:
    """Agent for outlier detection and correction"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def detect_outliers(self, data: pd.DataFrame) -> ValidationResult:
        """Detect outliers in numeric columns"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return ValidationResult(
                status=ValidationStatus.PASS,
                issues=[],
                corrections=[],
                confidence=1.0,
                metadata={"outliers_detected": 0}
            )
        
        outlier_info = {}
        issues = []
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_info[col] = {
                    "count": len(outliers),
                    "percentage": (len(outliers) / len(data)) * 100,
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                }
                issues.append(f"Column '{col}': {len(outliers)} outliers ({(len(outliers)/len(data)*100):.2f}%)")
        
        if not issues:
            return ValidationResult(
                status=ValidationStatus.PASS,
                issues=[],
                corrections=[],
                confidence=1.0,
                metadata={"outliers_detected": 0}
            )
        
        corrections = await self._generate_outlier_corrections(data, outlier_info)
        
        return ValidationResult(
            status=ValidationStatus.FAIL,
            issues=issues,
            corrections=corrections,
            confidence=0.6,
            metadata={"outlier_info": outlier_info}
        )
    
    async def _generate_outlier_corrections(self, data: pd.DataFrame, outlier_info: Dict) -> List[str]:
        """Generate outlier correction strategy using AI"""
        prompt = f"""
        Analyze outliers in this dataset and recommend correction strategies:
        
        Dataset Shape: {data.shape}
        Outlier Information: {json.dumps(outlier_info, indent=2, default=str)}
        
        For each column with outliers, recommend:
        1. Whether to remove, cap, or transform outliers
        2. Specific method and parameters
        3. Justification for the approach
        
        Consider the context and potential business impact.
        
        Provide response in JSON format:
        {{
            "strategies": [
                {{
                    "column": "column_name",
                    "action": "remove/cap/transform",
                    "method": "specific_method",
                    "justification": "reasoning"
                }}
            ]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.groq_client.generate_completion(messages)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                strategies = result.get("strategies", [])
                return [f"Column {s['column']}: {s['action']} using {s['method']} - {s['justification']}" for s in strategies]
            return ["Review and handle outliers based on business context"]
        except Exception as e:
            self.logger.error(f"Error generating outlier corrections: {e}")
            return ["Apply standard outlier detection and correction methods"]

class DataCleaningOrchestrator:
    """Main orchestrator for the enhanced data cleaning pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.groq_client = GroqClient(config.groq_api_key, config.model_name)
        self.schema_agent = SchemaValidationAgent(self.groq_client)
        self.categorical_agent = CategoricalCleaningAgent(self.groq_client)
        self.numerical_agent = NumericalCleaningAgent(self.groq_client)
        self.datetime_agent = DateTimeCleaningAgent(self.groq_client)
        self.outlier_agent = OutlierDetectionAgent(self.groq_client)
        self.imputation_agent = AdvancedImputationAgent(self.groq_client)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize audit log
        self.audit_log = []
    
    async def clean_data(self, data: pd.DataFrame, expected_schema: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced data cleaning pipeline with comprehensive validation and cleaning"""
        self.logger.info(f"Starting enhanced data cleaning pipeline for dataset with shape {data.shape}")
        
        original_data = data.copy()
        current_data = data.copy()
        iteration = 0
        
        pipeline_results = {
            "iterations": [],
            "final_status": "incomplete",
            "improvements": [],
            "audit_log": [],
            "cleaning_summary": {
                "original_shape": original_data.shape,
                "original_missing_count": original_data.isnull().sum().sum(),
                "original_dtypes": dict(original_data.dtypes.astype(str))
            }
        }
        
        while iteration < self.config.max_iterations:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration}")
            
            # Phase 1: Data type specific cleaning
            if iteration == 1:
                current_data = await self._phase_1_data_cleaning(current_data)
            
            # Phase 2: Validation and correction
            iteration_results = await self._run_validation_cycle(
                current_data, expected_schema, iteration
            )
            
            pipeline_results["iterations"].append(iteration_results)
            
            # Phase 3: Advanced imputation (after cleaning)
            if self.config.advanced_imputation_enabled:
                current_data, imputation_results = await self.imputation_agent.impute_missing_values(current_data)
                iteration_results["imputation"] = asdict(imputation_results)
            
            # Check if we should continue
            if iteration_results["overall_status"] == ValidationStatus.PASS:
                self.logger.info("All validations passed. Pipeline complete.")
                pipeline_results["final_status"] = "success"
                break
            
            # Apply corrections if enabled
            if self.config.enable_auto_correction:
                current_data = await self._apply_corrections(
                    current_data, iteration_results["corrections"]
                )
            
            # Check for improvements
            if iteration > 1:
                prev_issues = len(pipeline_results["iterations"][-2]["all_issues"])
                curr_issues = len(iteration_results["all_issues"])
                if curr_issues >= prev_issues:
                    self.logger.warning("No improvement detected. Stopping iterations.")
                    pipeline_results["final_status"] = "stagnant"
                    break
        
        if iteration >= self.config.max_iterations:
            pipeline_results["final_status"] = "max_iterations_reached"
        
        # Generate final audit report
        audit_report = await self._generate_audit_report(
            original_data, current_data, pipeline_results
        )
        
        pipeline_results["audit_report"] = audit_report
        pipeline_results["cleaning_summary"]["final_shape"] = current_data.shape
        pipeline_results["cleaning_summary"]["final_missing_count"] = current_data.isnull().sum().sum()
        pipeline_results["cleaning_summary"]["final_dtypes"] = dict(current_data.dtypes.astype(str))
        
        return current_data, pipeline_results
    
    async def _phase_1_data_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Phase 1: Comprehensive data type specific cleaning"""
        self.logger.info("Phase 1: Data type specific cleaning")
        
        current_data = data.copy()
        
        # Clean categorical data
        if self.config.categorical_cleaning_enabled:
            self.logger.info("Cleaning categorical data...")
            current_data, _ = await self.categorical_agent.clean_categorical_data(current_data)
        
        # Clean numerical data
        if self.config.numerical_cleaning_enabled:
            self.logger.info("Cleaning numerical data...")
            current_data, _ = await self.numerical_agent.clean_numerical_data(current_data)
        
        # Clean datetime data
        if self.config.datetime_cleaning_enabled:
            self.logger.info("Cleaning datetime data...")
            current_data, _ = await self.datetime_agent.clean_datetime_data(current_data)
        
        return current_data
    
    async def _run_validation_cycle(self, data: pd.DataFrame, expected_schema: Optional[Dict], iteration: int) -> Dict:
        """Run a complete validation cycle"""
        results = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "validations": {},
            "all_issues": [],
            "corrections": [],
            "overall_status": ValidationStatus.PASS
        }
        
        # Schema validation
        if self.config.schema_validation_enabled and expected_schema:
            self.logger.info("Running schema validation")
            schema_result = await self.schema_agent.validate_schema(data, expected_schema)
            results["validations"]["schema"] = asdict(schema_result)
            results["all_issues"].extend(schema_result.issues)
            results["corrections"].extend(schema_result.corrections)
            
            if schema_result.status != ValidationStatus.PASS:
                results["overall_status"] = ValidationStatus.FAIL
        
        # Outlier detection
        if self.config.outlier_detection_enabled:
            self.logger.info("Running outlier detection")
            outlier_result = await self.outlier_agent.detect_outliers(data)
            results["validations"]["outliers"] = asdict(outlier_result)
            results["all_issues"].extend(outlier_result.issues)
            results["corrections"].extend(outlier_result.corrections)
            
            if outlier_result.status != ValidationStatus.PASS:
                results["overall_status"] = ValidationStatus.FAIL
        
        # Data quality check
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            results["all_issues"].append(f"Dataset has {missing_count} missing values")
            results["overall_status"] = ValidationStatus.FAIL
        
        self.logger.info(f"Validation cycle {iteration} complete. Found {len(results['all_issues'])} issues.")
        
        return results
    
    async def _apply_corrections(self, data: pd.DataFrame, corrections: List[str]) -> pd.DataFrame:
        """Apply corrections to the dataset"""
        self.logger.info(f"Applying {len(corrections)} corrections")
        
        corrected_data = data.copy()
        
        # Apply outlier corrections
        for correction in corrections:
            if "outlier" in correction.lower():
                numeric_cols = corrected_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if corrected_data[col].dtype in ['int64', 'float64', 'Int64']:
                        Q1 = corrected_data[col].quantile(0.25)
                        Q3 = corrected_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Cap outliers instead of removing them
                        corrected_data[col] = corrected_data[col].clip(lower_bound, upper_bound)
        
        return corrected_data
    
    async def _generate_audit_report(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame, results: Dict) -> Dict:
        """Generate comprehensive audit report"""
        report = {
            "pipeline_summary": {
                "start_time": results["iterations"][0]["timestamp"] if results["iterations"] else None,
                "end_time": datetime.now().isoformat(),
                "iterations_run": len(results["iterations"]),
                "final_status": results["final_status"]
            },
            "data_transformation": {
                "original_shape": original_data.shape,
                "final_shape": cleaned_data.shape,
                "rows_changed": original_data.shape[0] - cleaned_data.shape[0],
                "columns_changed": original_data.shape[1] - cleaned_data.shape[1]
            },
            "quality_metrics": {
                "original_missing_values": original_data.isnull().sum().sum(),
                "final_missing_values": cleaned_data.isnull().sum().sum(),
                "data_completeness": (1 - cleaned_data.isnull().sum().sum() / cleaned_data.size) * 100,
                "missing_reduction": original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum()
            },
            "data_type_changes": {
                "original_dtypes": dict(original_data.dtypes.astype(str)),
                "final_dtypes": dict(cleaned_data.dtypes.astype(str))
            },
            "column_analysis": {},
            "issues_resolved": [],
            "remaining_issues": []
        }
        
        # Analyze each column
        for col in cleaned_data.columns:
            col_analysis = {
                "original_type": str(original_data[col].dtype) if col in original_data.columns else "new_column",
                "final_type": str(cleaned_data[col].dtype),
                "original_missing": original_data[col].isnull().sum() if col in original_data.columns else 0,
                "final_missing": cleaned_data[col].isnull().sum(),
                "unique_values": cleaned_data[col].nunique()
            }
            
            if cleaned_data[col].dtype == 'object':
                col_analysis["sample_values"] = cleaned_data[col].dropna().unique()[:5].tolist()
            
            report["column_analysis"][col] = col_analysis
        
        # Collect all issues from iterations
        for iteration in results["iterations"]:
            report["issues_resolved"].extend(iteration["corrections"])
            if iteration["iteration"] == len(results["iterations"]):  # Last iteration
                report["remaining_issues"] = iteration["all_issues"]
        
        return report

# Example usage with enhanced sample data
async def main():
    """Example usage of the enhanced data cleaning pipeline"""
    
    # Configuration
    config = PipelineConfig(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        max_iterations=2,  # Reduced for faster execution
        confidence_threshold=0.8,
        enable_auto_correction=True,
        schema_validation_enabled=True,
        outlier_detection_enabled=True,
        categorical_cleaning_enabled=True,
        numerical_cleaning_enabled=True,
        datetime_cleaning_enabled=True,
        advanced_imputation_enabled=True
    )
    
    # Enhanced sample data with various data quality issues
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Frank', 'Grace', 'Henry', None, 'Ivy', 'alice', 'BOB'],
        'age': [25, 30, 35, None, 28, 150, 22, 29, 31, 26, 'Unknown', '35.5'],
        'salary': [50000, 60000, 55000, 65000, None, 58000, 62000, 59000, 61000, 57000, 'N/A', '70,000'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', None, 'HR', 'Finance', 'IT', 'HR', 'it', 'human resources'],
        'country': ['USA', 'usa', 'India', 'india', 'UK', 'United Kingdom', 'US', 'INDIA', None, 'Unknown', 'Error', 'usa'],
        'join_date': ['2020-01-15', '2019-06-20', '2021-03-10', None, '2020-12-01', 'Invalid', '2019-08-15', '2020-05-20', '2021-01-10', 'N/A', '2020/07/15', '15-Jan-2020'],
        'score': [85.5, 90.0, 78.5, 88.0, None, 92.5, 'Missing', 87.0, 89.5, 91.0, 'Error', 95.0]
    })
    
    # Expected schema
    expected_schema = {
        'columns': ['id', 'name', 'age', 'salary', 'department', 'country', 'join_date', 'score'],
        'types': {
            'id': 'int',
            'name': 'string',
            'age': 'int',
            'salary': 'float',
            'department': 'string',
            'country': 'string',
            'join_date': 'datetime',
            'score': 'float'
        }
    }
    
    # Initialize and run pipeline
    orchestrator = DataCleaningOrchestrator(config)
    
    print("Starting enhanced data cleaning pipeline...")
    print(f"Original data shape: {sample_data.shape}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
    
    try:
        cleaned_data, results = await orchestrator.clean_data(sample_data, expected_schema)
        
        print(f"\n{'='*50}")
        print("PIPELINE RESULTS")
        print(f"{'='*50}")
        print(f"Pipeline status: {results['final_status']}")
        print(f"Final data shape: {cleaned_data.shape}")
        print(f"Iterations run: {len(results['iterations'])}")
        print(f"Missing values reduced: {results['audit_report']['quality_metrics']['missing_reduction']}")
        print(f"Data completeness: {results['audit_report']['quality_metrics']['data_completeness']:.2f}%")
        
        print("\nCleaned Data:")
        print(cleaned_data.to_string())
        
        print(f"\nFinal data types: {dict(cleaned_data.dtypes)}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_data.to_csv(f"cleaned_data_{timestamp}.csv", index=False)
        
        with open(f"pipeline_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved:")
        print(f"- Cleaned data: cleaned_data_{timestamp}.csv")
        print(f"- Pipeline results: pipeline_results_{timestamp}.json")
        
        # Display summary of changes
        print(f"\n{'='*50}")
        print("CLEANING SUMMARY")
        print(f"{'='*50}")
        
        for col in cleaned_data.columns:
            if col in results["audit_report"]["column_analysis"]:
                col_info = results["audit_report"]["column_analysis"][col]
                print(f"Column '{col}':")
                print(f"  - Type: {col_info['original_type']} → {col_info['final_type']}")
                print(f"  - Missing: {col_info['original_missing']} → {col_info['final_missing']}")
                print(f"  - Unique values: {col_info['unique_values']}")
                if 'sample_values' in col_info:
                    print(f"  - Sample values: {col_info['sample_values']}")
                print()
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        logging.error(f"Pipeline error: {traceback.format_exc()}")
        
        # Fallback: return basic cleaned data
        print("Attempting basic cleanup without AI...")
        basic_cleaned = sample_data.copy()
        
        # Basic cleaning without AI
        for col in basic_cleaned.select_dtypes(include=['object']).columns:
            # Remove obvious null values
            null_patterns = ['nan', 'null', 'none', 'n/a', 'na', 'unknown', 'error', 'undefined', 'missing', 'blank', 'empty']
            for pattern in null_patterns:
                basic_cleaned[col] = basic_cleaned[col].replace(pattern, np.nan, regex=True)
        
        print("Basic cleanup completed")
        print(basic_cleaned.to_string())

if __name__ == "__main__":
    asyncio.run(main())