"""
Configuration file for Answer Evaluation System
Modify these settings to customize the system behavior
"""

import os

class Config:
    # Application Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Upload Settings
    UPLOAD_FOLDER = 'uploads'
    RESULT_FOLDER = 'results'
    VECTOR_DB_FOLDER = 'vector_db'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # OCR Settings
    OCR_DPI = 300  # DPI for PDF to image conversion
    OCR_LANGUAGE = 'eng'  # Tesseract language (eng, hin, etc.)
    
    # Tesseract Path (Windows - uncomment and modify if needed)
    # TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Vector Database Settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model
    
    # Evaluation Settings
    SIMILARITY_WEIGHTS = {
        'cosine': 0.35,      # TF-IDF + Cosine similarity weight
        'sequence': 0.25,    # Sequence matcher weight
        'word_overlap': 0.25, # Word overlap weight
        'keyword': 0.15      # Keyword matching weight
    }
    
    # Grading Scale (similarity_threshold: marks_percentage)
    GRADING_SCALE = {
        0.90: 1.00,   # 90%+ similarity → 100% marks
        0.80: 0.95,   # 80-90% similarity → 95% marks
        0.70: 0.85,   # 70-80% similarity → 85% marks
        0.60: 0.75,   # 60-70% similarity → 75% marks
        0.50: 0.65,   # 50-60% similarity → 65% marks
        0.40: 0.55,   # 40-50% similarity → 55% marks
        0.30: 0.45,   # 30-40% similarity → 45% marks
        0.20: 0.30,   # 20-30% similarity → 30% marks
        0.10: 0.20,   # 10-20% similarity → 20% marks
        0.00: 0.10    # <10% similarity → 10% marks (for attempt)
    }
    
    # Grade Thresholds (percentage: grade)
    GRADE_THRESHOLDS = {
        90: 'A+',
        80: 'A',
        70: 'B+',
        60: 'B',
        50: 'C',
        40: 'D',
        0: 'F'
    }
    
    # Feedback Messages
    FEEDBACK_MESSAGES = {
        0.90: "Excellent answer! Matches the key answer very closely.",
        0.80: "Very good answer! Most key points covered.",
        0.70: "Good answer. Covered major points.",
        0.60: "Satisfactory answer. Some important points covered.",
        0.50: "Average answer. Missing several key points.",
        0.40: "Below average. Many key points missing.",
        0.30: "Weak answer. Most key points not covered.",
        0.20: "Poor answer. Very few relevant points.",
        0.00: "Inadequate answer. Does not match expected content."
    }
    
    # PDF Report Settings
    REPORT_PAGE_SIZE = 'A4'  # A4, Letter, etc.
    REPORT_TITLE = "Answer Paper Evaluation Report"
    REPORT_LOGO = None  # Path to logo image (optional)
    
    # Question Pattern Regex (for extracting questions)
    QUESTION_PATTERNS = [
        r'Q(\d+)[\.\)]\s*(.*?)\s*[\(\[](\d+)\s*marks?[\)\]]',
        r'(\d+)[\.\)]\s*(.*?)\s*[\(\[](\d+)\s*marks?[\)\]]',
        r'Question\s+(\d+)[\.\)]\s*(.*?)\s*[\(\[](\d+)\s*marks?[\)\]]',
    ]
    
    # Answer Pattern Regex (for extracting answers)
    ANSWER_PATTERNS = [
        r'(?:Answer|Ans|A)[\.\:\s]*(\d+)[\.\)]\s*(.*?)(?=(?:Answer|Ans|A)[\.\:\s]*\d+|\Z)',
        r'Q(\d+)[\.\)]\s*(?:Answer|Ans|A)[\.\:\s]*(.*?)(?=Q\d+|\Z)',
        r'(\d+)[\.\)]\s*(.*?)(?=\d+\.|\Z)',
    ]
    
    # Minimum Text Length for Valid Answer
    MIN_ANSWER_LENGTH = 10  # characters
    
    # Enable/Disable Features
    ENABLE_OCR = True
    ENABLE_VECTOR_SEARCH = True
    ENABLE_DETAILED_LOGGING = True
    
    @staticmethod
    def get_grade(percentage):
        """Get grade based on percentage"""
        for threshold, grade in sorted(Config.GRADE_THRESHOLDS.items(), reverse=True):
            if percentage >= threshold:
                return grade
        return 'F'
    
    @staticmethod
    def get_feedback(similarity_score):
        """Get feedback based on similarity score"""
        for threshold, feedback in sorted(Config.FEEDBACK_MESSAGES.items(), reverse=True):
            if similarity_score >= threshold:
                return feedback
        return "Inadequate answer."
    
    @staticmethod
    def calculate_marks_from_similarity(similarity_score, max_marks):
        """Calculate marks based on similarity score"""
        for threshold, percentage in sorted(Config.GRADING_SCALE.items(), reverse=True):
            if similarity_score >= threshold:
                return max_marks * percentage
        return max_marks * 0.10  # Minimum 10% for attempt

# Development Config
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

# Production Config
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    # Add production-specific settings
    # SECRET_KEY = os.environ.get('SECRET_KEY')

# Testing Config
class TestingConfig(Config):
    TESTING = True
    DEBUG = True

# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
