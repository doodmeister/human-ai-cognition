from datetime import datetime
import logging
from typing import Dict, Any, Optional
import boto3
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for thresholds
LOW_SALIENCE_THRESHOLD = 0.3
HIGH_SALIENCE_THRESHOLD = 0.7
FATIGUE_DECREMENT = 0.1
ATTENTION_INCREMENT = 0.1

class MetaFeedbackManager:
    """
    Manages meta-cognitive feedback, including logging dream consolidation events
    and adjusting cognitive states like fatigue and attention.
    """
    def __init__(self):
        """
        Initialize the MetaFeedbackManager.
        """
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = "HumanAI/Metacognition"

    def send_feedback(self, event: str, count: int = 1, avg_salience: float = 0.0, 
                     metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Send metacognitive feedback metrics to CloudWatch.
        
        Args:
            event: Event type identifier
            count: Number of items processed
            avg_salience: Average salience score
            metrics: Additional custom metrics
        """
        timestamp = datetime.utcnow()
        metric_data = [
            {
                'MetricName': f'{event}_count',
                'Value': count,
                'Unit': 'Count',
                'Timestamp': timestamp
            },
            {
                'MetricName': f'{event}_salience',
                'Value': avg_salience,
                'Unit': 'None',
                'Timestamp': timestamp
            }
        ]

        # Add procedural memory specific metrics
        if metrics and metrics.get('memory_type') == 'procedural':
            proc_metrics = [
                {
                    'MetricName': 'procedure_execution_success',
                    'Value': 1 if metrics.get('execution_success', False) else 0,
                    'Unit': 'Count',
                    'Timestamp': timestamp
                },
                {
                    'MetricName': 'procedure_execution_count',
                    'Value': metrics.get('execution_count', 0),
                    'Unit': 'Count',
                    'Timestamp': timestamp
                },
                {
                    'MetricName': 'procedure_learning_rate',
                    'Value': metrics.get('learning_rate', 0.0),
                    'Unit': 'None',
                    'Timestamp': timestamp
                }
            ]
            metric_data.extend(proc_metrics)

        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data
            )
        except Exception as e:
            print(f"Failed to send metrics: {str(e)}")

    def log_procedure_metrics(self, name: str, success: bool, execution_time: float) -> None:
        """
        Log specific metrics for procedural memory operations.
        """
        try:
            timestamp = datetime.utcnow()
            metric_data = [
                {
                    'MetricName': 'procedure_execution',
                    'Value': 1 if success else 0,
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ProcedureName', 'Value': name}]
                },
                {
                    'MetricName': 'procedure_execution_time',
                    'Value': execution_time,
                    'Unit': 'Seconds',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ProcedureName', 'Value': name}]
                }
            ]

            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data
            )
        except Exception as e:
            logger.error(f"Failed to log procedure metrics: {str(e)}")
