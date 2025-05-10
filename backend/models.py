from base import db, Base
from sqlalchemy.orm import relationship
from datetime import datetime


class Event(Base):
    __tablename__ = 'events'

    id = db.Column(db.Integer, primary_key=True, index=True)
    intent = db.Column(db.String)
    status = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime)
    notification_text = db.Column(db.String)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'notification_text': self.notification_text
        }


class Contact(Base):
    __tablename__ = 'phone_contacts'

    id = db.Column(db.Integer, primary_key=True, index=True)
    phone_number = db.Column(db.String) 
    name = db.Column(db.String) 
