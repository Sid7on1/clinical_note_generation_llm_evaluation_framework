import logging
import pandas as pd
import numpy as np
import psutil
from typing import Dict, List, Tuple
from datetime import datetime
from threading import Lock
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Constants
AB_TEST_DURATION = 30  # days
ROLL_OUT_THRESHOLD = 0.8  # 80% success rate
FEEDBACK_COLLECTION_INTERVAL = 7  # days

# Configuration
class DeploymentConfig:
    def __init__(self, ab_test_duration: int = AB_TEST_DURATION, roll_out_threshold: float = ROLL_OUT_THRESHOLD, feedback_collection_interval: int = FEEDBACK_COLLECTION_INTERVAL):
        self.ab_test_duration = ab_test_duration
        self.roll_out_threshold = roll_out_threshold
        self.feedback_collection_interval = feedback_collection_interval

# Data structures
@dataclass
class ClinicalProgram:
    id: int
    name: str
    description: str

@dataclass
class DeploymentStatus:
    program_id: int
    status: str
    start_date: datetime
    end_date: datetime

@dataclass
class Feedback:
    program_id: int
    feedback: str
    rating: int

# Exception classes
class DeploymentError(Exception):
    pass

class InvalidConfigError(DeploymentError):
    pass

class InvalidProgramError(DeploymentError):
    pass

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lock for thread safety
lock = Lock()

class DeploymentManager:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.programs: Dict[int, ClinicalProgram] = {}
        self.deployment_status: Dict[int, DeploymentStatus] = {}
        self.feedback: Dict[int, List[Feedback]] = {}

    def setup_ab_test(self, program_id: int) -> None:
        """
        Set up A/B testing for a clinical program.

        Args:
        program_id (int): The ID of the clinical program.

        Raises:
        InvalidProgramError: If the program ID is not found.
        """
        with lock:
            if program_id not in self.programs:
                raise InvalidProgramError(f"Program ID {program_id} not found")
            program = self.programs[program_id]
            logger.info(f"Setting up A/B testing for program {program.name}")
            # Implement A/B testing logic here
            self.deployment_status[program_id] = DeploymentStatus(program_id, "ab_testing", datetime.now(), datetime.now() + pd.Timedelta(days=self.config.ab_test_duration))

    def monitor_performance(self, program_id: int) -> Tuple[float, float]:
        """
        Monitor the performance of a clinical program.

        Args:
        program_id (int): The ID of the clinical program.

        Returns:
        Tuple[float, float]: The success rate and failure rate of the program.

        Raises:
        InvalidProgramError: If the program ID is not found.
        """
        with lock:
            if program_id not in self.programs:
                raise InvalidProgramError(f"Program ID {program_id} not found")
            program = self.programs[program_id]
            logger.info(f"Monitoring performance for program {program.name}")
            # Implement performance monitoring logic here
            success_rate = np.random.uniform(0, 1)  # Replace with actual success rate calculation
            failure_rate = 1 - success_rate
            return success_rate, failure_rate

    def handle_rollout(self, program_id: int) -> None:
        """
        Handle the rollout of a clinical program.

        Args:
        program_id (int): The ID of the clinical program.

        Raises:
        InvalidProgramError: If the program ID is not found.
        """
        with lock:
            if program_id not in self.programs:
                raise InvalidProgramError(f"Program ID {program_id} not found")
            program = self.programs[program_id]
            logger.info(f"Handling rollout for program {program.name}")
            # Implement rollout logic here
            success_rate, _ = self.monitor_performance(program_id)
            if success_rate >= self.config.roll_out_threshold:
                self.deployment_status[program_id].status = "rolled_out"
                logger.info(f"Program {program.name} rolled out successfully")

    def collect_feedback(self, program_id: int) -> List[Feedback]:
        """
        Collect feedback for a clinical program.

        Args:
        program_id (int): The ID of the clinical program.

        Returns:
        List[Feedback]: A list of feedback for the program.

        Raises:
        InvalidProgramError: If the program ID is not found.
        """
        with lock:
            if program_id not in self.programs:
                raise InvalidProgramError(f"Program ID {program_id} not found")
            program = self.programs[program_id]
            logger.info(f"Collecting feedback for program {program.name}")
            # Implement feedback collection logic here
            feedback = [Feedback(program_id, "Example feedback", 5)]  # Replace with actual feedback collection
            self.feedback[program_id] = feedback
            return feedback

    def generate_deployment_reports(self) -> Dict[int, Dict[str, str]]:
        """
        Generate deployment reports for all clinical programs.

        Returns:
        Dict[int, Dict[str, str]]: A dictionary of deployment reports for each program.
        """
        with lock:
            reports = {}
            for program_id, program in self.programs.items():
                report = {
                    "program_name": program.name,
                    "deployment_status": self.deployment_status.get(program_id, DeploymentStatus(program_id, "not_deployed", datetime.now(), datetime.now())).status
                }
                reports[program_id] = report
            return reports

    def add_program(self, program: ClinicalProgram) -> None:
        """
        Add a clinical program to the deployment manager.

        Args:
        program (ClinicalProgram): The clinical program to add.
        """
        with lock:
            self.programs[program.id] = program

    def remove_program(self, program_id: int) -> None:
        """
        Remove a clinical program from the deployment manager.

        Args:
        program_id (int): The ID of the clinical program to remove.

        Raises:
        InvalidProgramError: If the program ID is not found.
        """
        with lock:
            if program_id not in self.programs:
                raise InvalidProgramError(f"Program ID {program_id} not found")
            del self.programs[program_id]

def main():
    config = DeploymentConfig()
    deployment_manager = DeploymentManager(config)
    program = ClinicalProgram(1, "Example Program", "This is an example program")
    deployment_manager.add_program(program)
    deployment_manager.setup_ab_test(1)
    deployment_manager.handle_rollout(1)
    feedback = deployment_manager.collect_feedback(1)
    reports = deployment_manager.generate_deployment_reports()
    logger.info(f"Deployment reports: {reports}")

if __name__ == "__main__":
    main()