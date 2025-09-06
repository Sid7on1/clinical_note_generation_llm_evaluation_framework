import logging
import re
from typing import Dict, List
from spacy import displacy
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SOAP_SECTION_ORDER = ['S', 'O', 'A', 'P']
MEDICAL_ACCURACY_THRESHOLD = 0.8
OT_STANDARDS_THRESHOLD = 0.9

class SoapValidator:
    """
    Validates generated SOAP notes against clinical standards, checks for medical accuracy, 
    and ensures compliance with occupational therapy documentation requirements.
    """

    def __init__(self, soap_note: str):
        """
        Initializes the SoapValidator with a SOAP note.

        Args:
        soap_note (str): The SOAP note to be validated.
        """
        self.soap_note = soap_note
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def validate_soap_sections(self) -> bool:
        """
        Validates the SOAP note sections against the standard order (S, O, A, P).

        Returns:
        bool: True if the sections are in the correct order, False otherwise.
        """
        sections = self.soap_note.split('\n\n')
        section_names = [section.split('\n')[0] for section in sections]
        return section_names == SOAP_SECTION_ORDER

    def check_medical_accuracy(self) -> float:
        """
        Checks the medical accuracy of the SOAP note using a simple keyword extraction method.

        Returns:
        float: A score between 0 and 1 representing the medical accuracy of the SOAP note.
        """
        medical_keywords = ['diagnosis', 'treatment', 'medication', 'symptoms']
        keywords_found = 0
        for keyword in medical_keywords:
            if keyword in self.soap_note.lower():
                keywords_found += 1
        return keywords_found / len(medical_keywords)

    def verify_ot_standards(self) -> float:
        """
        Verifies the compliance of the SOAP note with occupational therapy documentation requirements.

        Returns:
        float: A score between 0 and 1 representing the compliance of the SOAP note with OT standards.
        """
        ot_keywords = ['goals', 'interventions', 'outcomes', 'evaluation']
        keywords_found = 0
        for keyword in ot_keywords:
            if keyword in self.soap_note.lower():
                keywords_found += 1
        return keywords_found / len(ot_keywords)

    def flag_clinical_errors(self) -> List[str]:
        """
        Flags potential clinical errors in the SOAP note.

        Returns:
        List[str]: A list of potential clinical errors found in the SOAP note.
        """
        errors = []
        # Check for missing sections
        sections = self.soap_note.split('\n\n')
        section_names = [section.split('\n')[0] for section in sections]
        for section in SOAP_SECTION_ORDER:
            if section not in section_names:
                errors.append(f"Missing {section} section")
        # Check for medical accuracy
        medical_accuracy = self.check_medical_accuracy()
        if medical_accuracy < MEDICAL_ACCURACY_THRESHOLD:
            errors.append(f"Low medical accuracy ({medical_accuracy:.2f})")
        # Check for OT standards compliance
        ot_standards = self.verify_ot_standards()
        if ot_standards < OT_STANDARDS_THRESHOLD:
            errors.append(f"Low OT standards compliance ({ot_standards:.2f})")
        return errors

    def suggest_corrections(self, errors: List[str]) -> Dict[str, str]:
        """
        Suggests corrections for the flagged clinical errors.

        Args:
        errors (List[str]): The list of flagged clinical errors.

        Returns:
        Dict[str, str]: A dictionary with error messages as keys and suggested corrections as values.
        """
        corrections = {}
        for error in errors:
            if "Missing" in error:
                corrections[error] = f"Add the missing {error.split(' ')[1]} section"
            elif "Low medical accuracy" in error:
                corrections[error] = "Review the SOAP note for medical accuracy and add relevant keywords"
            elif "Low OT standards compliance" in error:
                corrections[error] = "Review the SOAP note for OT standards compliance and add relevant keywords"
        return corrections

    def validate(self) -> Dict[str, str]:
        """
        Validates the SOAP note and returns a dictionary with validation results.

        Returns:
        Dict[str, str]: A dictionary with validation results, including errors and suggested corrections.
        """
        validation_results = {}
        if not self.validate_soap_sections():
            validation_results["soap_sections"] = "Invalid SOAP section order"
        medical_accuracy = self.check_medical_accuracy()
        validation_results["medical_accuracy"] = f"{medical_accuracy:.2f}"
        ot_standards = self.verify_ot_standards()
        validation_results["ot_standards"] = f"{ot_standards:.2f}"
        errors = self.flag_clinical_errors()
        validation_results["errors"] = errors
        corrections = self.suggest_corrections(errors)
        validation_results["corrections"] = corrections
        return validation_results

class SoapNote:
    """
    Represents a SOAP note.
    """

    def __init__(self, text: str):
        """
        Initializes the SoapNote with the text.

        Args:
        text (str): The text of the SOAP note.
        """
        self.text = text

    def validate(self) -> Dict[str, str]:
        """
        Validates the SOAP note.

        Returns:
        Dict[str, str]: A dictionary with validation results.
        """
        validator = SoapValidator(self.text)
        return validator.validate()

def main():
    soap_note_text = """
S
Subjective: The patient reports feeling dizzy and lightheaded.

O
Objective: The patient's blood pressure is 120/80 mmHg.

A
Assessment: The patient is diagnosed with hypertension.

P
Plan: The patient will be prescribed medication to lower blood pressure.
"""
    soap_note = SoapNote(soap_note_text)
    validation_results = soap_note.validate()
    logger.info(validation_results)

if __name__ == "__main__":
    main()