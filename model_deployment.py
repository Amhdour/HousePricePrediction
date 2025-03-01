import joblib
import os
import json
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from database import Base, SessionLocal

class ModelDeployment(Base):
    __tablename__ = "model_deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String)
    model_type = Column(String)
    feature_columns = Column(String)  # JSON string of features
    metrics = Column(String)  # JSON string of metrics
    deployed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)

class ModelManager:
    def __init__(self):
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save_model(self, model, model_type, feature_columns, metrics):
        """Save model and its metadata"""
        # Generate version based on timestamp
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model file
        model_path = os.path.join(self.model_dir, f"model_{version}.joblib")
        joblib.dump(model, model_path)
        
        # Save to database
        db = SessionLocal()
        try:
            deployment = ModelDeployment(
                model_version=version,
                model_type=model_type,
                feature_columns=json.dumps(feature_columns),
                metrics=json.dumps(metrics)
            )
            db.add(deployment)
            db.commit()
            db.refresh(deployment)
            return version
        finally:
            db.close()
    
    def load_model(self, version):
        """Load model by version"""
        model_path = os.path.join(self.model_dir, f"model_{version}.joblib")
        return joblib.load(model_path)
    
    def get_deployment_history(self):
        """Get model deployment history"""
        db = SessionLocal()
        try:
            deployments = db.query(ModelDeployment).order_by(
                ModelDeployment.created_at.desc()
            ).all()
            return deployments
        finally:
            db.close()
    
    def deploy_model(self, version):
        """Deploy a specific model version"""
        db = SessionLocal()
        try:
            # Set all models as not deployed
            db.query(ModelDeployment).update({"deployed": False})
            
            # Set selected model as deployed
            deployment = db.query(ModelDeployment).filter(
                ModelDeployment.model_version == version
            ).first()
            
            if deployment:
                deployment.deployed = True
                deployment.deployed_at = datetime.utcnow()
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    def get_current_model(self):
        """Get currently deployed model"""
        db = SessionLocal()
        try:
            deployment = db.query(ModelDeployment).filter(
                ModelDeployment.deployed == True
            ).first()
            
            if deployment:
                model = self.load_model(deployment.model_version)
                feature_columns = json.loads(deployment.feature_columns)
                return model, feature_columns, deployment.model_version
            return None, None, None
        finally:
            db.close()
