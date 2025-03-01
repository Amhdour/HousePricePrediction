from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class HouseData(Base):
    __tablename__ = "house_data"
    
    id = Column(Integer, primary_key=True, index=True)
    square_feet = Column(Float)
    bedrooms = Column(Integer)
    bathrooms = Column(Float)
    age = Column(Integer)
    lot_size = Column(Float)
    garage_spaces = Column(Integer)
    price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    square_feet = Column(Float)
    bedrooms = Column(Integer)
    bathrooms = Column(Float)
    age = Column(Integer)
    lot_size = Column(Float)
    garage_spaces = Column(Integer)
    predicted_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_house_data(data_dict):
    """Save house data to database"""
    db = SessionLocal()
    try:
        house_data = HouseData(**data_dict)
        db.add(house_data)
        db.commit()
        return house_data
    finally:
        db.close()

def save_prediction(data_dict):
    """Save prediction to database"""
    db = SessionLocal()
    try:
        prediction = PredictionHistory(**data_dict)
        db.add(prediction)
        db.commit()
        return prediction
    finally:
        db.close()

def get_recent_predictions(limit=10):
    """Get recent predictions"""
    db = SessionLocal()
    try:
        return db.query(PredictionHistory).order_by(
            PredictionHistory.created_at.desc()
        ).limit(limit).all()
    finally:
        db.close()
