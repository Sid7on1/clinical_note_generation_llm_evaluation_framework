import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import logging
import logging.config
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

# Set up logging
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'mutual_learning_system.log',
            'maxBytes': 1024 * 1024 * 10,  # 10MB
            'backupCount': 5,
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
})

logger = logging.getLogger(__name__)

@dataclass
class ClinicianFeedback:
    """Dataclass to represent clinician feedback"""
    label: str
    confidence: float

class LearningOutcome(Enum):
    """Enum to represent learning outcomes"""
    ACCURACY = 1
    PRECISION = 2
    RECALL = 3
    F1_SCORE = 4

class MutualLearningSystem:
    """Class to implement mutual learning system"""
    def __init__(self, model: str = 'logistic_regression', config: Dict = None):
        self.model = model
        self.config = config if config else {}
        self.model_preferences = {}
        self.learning_outcomes = {}
        self.scaler = StandardScaler()
        self.pipeline = None

    def capture_clinician_feedback(self, data: pd.DataFrame, labels: List[str]) -> List[ClinicianFeedback]:
        """Capture clinician feedback"""
        feedback = []
        for i, row in data.iterrows():
            label = labels[i]
            confidence = self.predict(row)
            feedback.append(ClinicianFeedback(label, confidence))
        return feedback

    def update_model_preferences(self, feedback: List[ClinicianFeedback]) -> None:
        """Update model preferences based on clinician feedback"""
        for feedback_item in feedback:
            label = feedback_item.label
            confidence = feedback_item.confidence
            if label not in self.model_preferences:
                self.model_preferences[label] = []
            self.model_preferences[label].append(confidence)

    def suggest_clinical_insights(self, data: pd.DataFrame, labels: List[str]) -> List[Tuple[str, float]]:
        """Suggest clinical insights based on model preferences"""
        insights = []
        for i, row in data.iterrows():
            label = labels[i]
            confidence = self.predict(row)
            insights.append((label, confidence))
        return insights

    def track_learning_outcomes(self, data: pd.DataFrame, labels: List[str], feedback: List[ClinicianFeedback]) -> Dict[LearningOutcome, float]:
        """Track learning outcomes"""
        outcomes = {}
        for outcome in LearningOutcome:
            outcomes[outcome] = 0.0
        for i, row in data.iterrows():
            label = labels[i]
            confidence = feedback[i].confidence
            if label in self.model_preferences:
                accuracy = accuracy_score([label] * len(self.model_preferences[label]), self.model_preferences[label])
                outcomes[LearningOutcome.ACCURACY] += accuracy
                outcomes[LearningOutcome.PRECISION] += precision_score(self.model_preferences[label], [label] * len(self.model_preferences[label]))
                outcomes[LearningOutcome.RECALL] += recall_score(self.model_preferences[label], [label] * len(self.model_preferences[label]))
                outcomes[LearningOutcome.F1_SCORE] += f1_score(self.model_preferences[label], [label] * len(self.model_preferences[label]))
        for outcome in outcomes:
            outcomes[outcome] /= len(data)
        return outcomes

    def generate_mutual_reports(self, data: pd.DataFrame, labels: List[str], feedback: List[ClinicianFeedback]) -> Dict[str, float]:
        """Generate mutual reports"""
        reports = {}
        for label in set(labels):
            reports[label] = 0.0
        for i, row in data.iterrows():
            label = labels[i]
            confidence = feedback[i].confidence
            if label in self.model_preferences:
                reports[label] += confidence
        for label in reports:
            reports[label] /= len(data)
        return reports

    def predict(self, data: pd.DataFrame) -> float:
        """Predict using the model"""
        if not self.pipeline:
            if self.model == 'logistic_regression':
                self.pipeline = Pipeline([('scaler', self.scaler), ('lr', LogisticRegression())])
            elif self.model == 'random_forest':
                self.pipeline = Pipeline([('scaler', self.scaler), ('rf', RandomForestClassifier())])
            else:
                raise ValueError('Invalid model')
        try:
            return self.pipeline.predict(data)[0]
        except NotFittedError:
            raise ValueError('Model not fitted')

def load_config(config_file: str) -> Dict:
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    config_file = 'config.json'
    config = load_config(config_file)
    model = MutualLearningSystem(config['model'], config)
    data = pd.read_csv('data.csv')
    labels = data['label']
    feedback = model.capture_clinician_feedback(data.drop('label', axis=1), labels)
    model.update_model_preferences(feedback)
    insights = model.suggest_clinical_insights(data.drop('label', axis=1), labels)
    outcomes = model.track_learning_outcomes(data.drop('label', axis=1), labels, feedback)
    reports = model.generate_mutual_reports(data.drop('label', axis=1), labels, feedback)
    logger.info('Insights: %s', insights)
    logger.info('Outcomes: %s', outcomes)
    logger.info('Reports: %s', reports)

if __name__ == '__main__':
    main()