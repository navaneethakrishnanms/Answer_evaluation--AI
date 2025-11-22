"""
AI Answer Evaluation System - Main Application
Uses DeepSeek-OCR + Ollama LLM for intelligent answer evaluation
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import json
from pdf_processor import PDFProcessor
from ollama_evaluator import OllamaEvaluator
from pdf_generator import ResultPDFGenerator, OCRtoPDFConverter
import traceback
import re
from flask_cors import CORS
from workflow import EvaluationWorkflow, WorkflowError

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Create required folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize components
print("="*70)
print("🔧 Initializing AI Answer Evaluation System...")
print("="*70)

# PDF Processor with DeepSeek-OCR
print("\n[*] Initializing PDF Processor with DeepSeek-OCR...")
try:
    pdf_processor = PDFProcessor(use_deepseek=True)
    print("[+] PDF Processor initialized")
except Exception as e:
    print(f"[!] Failed to initialize PDF Processor: {e}")
    exit(1)

# Ollama Evaluator
OLLAMA_MODEL = "llama3"
print(f"\n[*] Initializing Ollama LLM (Model: {OLLAMA_MODEL})...")
try:
    ollama_evaluator = OllamaEvaluator(model_name=OLLAMA_MODEL)
    print(f"[+] Ollama evaluator initialized with {OLLAMA_MODEL}")
except Exception as e:
    print(f"[!] Warning: Ollama not available: {e}")
    print("    Please ensure:")
    print("    1. Ollama is installed (https://ollama.ai)")
    print(f"    2. Model is pulled: ollama pull {OLLAMA_MODEL}")
    ollama_evaluator = None

# PDF Generators
result_pdf_generator = ResultPDFGenerator()
ocr_pdf_converter = OCRtoPDFConverter()

# Workflow Orchestrator
workflow = EvaluationWorkflow(
    pdf_processor=pdf_processor,
    evaluator=ollama_evaluator,
    result_pdf_generator=result_pdf_generator,
    ocr_pdf_converter=ocr_pdf_converter,
    upload_folder=app.config['UPLOAD_FOLDER'],
    result_folder=app.config['RESULT_FOLDER']
)

print("\n" + "="*70)
print("✓ System Initialization Complete!")
print("="*70 + "\n")

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _extract_file_from_request():
    if 'file' not in request.files:
        raise WorkflowError('No file provided', status_code=400)

    file = request.files['file']
    if file.filename == '':
        raise WorkflowError('No file selected', status_code=400)

    if not allowed_file(file.filename):
        raise WorkflowError('Invalid file format. Only PDF allowed', status_code=400)

    return file


def _save_uploaded_file(prefix: str, file_storage):
    filename = secure_filename(file_storage.filename)
    stored_name = f"{prefix}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file_storage.save(filepath)
    return stored_name, filepath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_question_paper', methods=['POST'])
def upload_question_paper():
    """Upload and process question paper PDF"""
    try:
        file = _extract_file_from_request()
        stored_name, filepath = _save_uploaded_file('question_paper', file)

        print(f"\n{'='*70}")
        print(f"📋 PROCESSING QUESTION PAPER: {stored_name}")
        print(f"{'='*70}")

        questions = workflow.process_question_paper(filepath)

        total_marks = sum(q.max_marks for q in questions)
        print(f"\n✓ Question paper processed successfully!")
        print(f"   Total questions: {len(questions)}")
        print(f"   Total marks: {total_marks}")
        print(f"{'='*70}\n")

        return jsonify({
            'success': True,
            'message': 'Question paper uploaded and processed successfully',
            'questions_count': len(questions),
            'total_marks': total_marks,
            'questions': [
                {
                    'number': q.number,
                    'marks': q.max_marks,
                    'preview': q.text[:100] + '...' if len(q.text) > 100 else q.text
                }
                for q in questions
            ]
        })

    except WorkflowError as e:
        return jsonify({'success': False, 'error': e.message}), e.status_code
    except Exception as e:
        print(f"✗ Error uploading question paper: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload_answer_key', methods=['POST'])
def upload_answer_key():
    """Upload and process answer key PDF"""
    try:
        file = _extract_file_from_request()
        stored_name, filepath = _save_uploaded_file('answer_key', file)

        print(f"\n{'='*70}")
        print(f"✅ PROCESSING ANSWER KEY: {stored_name}")
        print(f"{'='*70}")

        answers = workflow.process_answer_key(filepath)

        print(f"\n✓ Answer key processed successfully")
        print(f"   Total answers: {len(answers)}")
        print(f"{'='*70}\n")

        return jsonify({
            'success': True,
            'message': 'Answer key uploaded and processed successfully',
            'answers_count': len(answers),
            'answers': [
                {
                    'number': a.number,
                    'preview': a.text[:100] + '...' if len(a.text) > 100 else a.text
                }
                for a in answers
            ]
        })

    except WorkflowError as e:
        return jsonify({'success': False, 'error': e.message}), e.status_code
    except Exception as e:
        print(f"✗ Error uploading answer key: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/evaluate_student_paper', methods=['POST'])
def evaluate_student_paper():
    """Evaluate student's handwritten answer paper using Ollama LLM"""
    try:
        file = _extract_file_from_request()
        student_name = request.form.get('student_name', 'Student')

        stored_name, filepath = _save_uploaded_file('student', file)

        print(f"\n{'='*70}")
        print(f"📝 EVALUATING STUDENT PAPER")
        print(f"{'='*70}")
        print(f"✓ Student: {student_name}")
        print(f"✓ File: {stored_name}")
        print(f"{'='*70}\n")

        workflow_result = workflow.evaluate_student_paper(
            filepath=filepath,
            filename=stored_name,
            student_name=student_name
        )

        evaluation_result = workflow_result['evaluation']

        print(f"\n{'='*70}")
        print(f"✓ EVALUATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Student: {student_name}")
        print(f"Marks: {evaluation_result['obtained_marks']:.2f}/{evaluation_result['total_marks']}")
        print(f"Percentage: {evaluation_result['percentage']:.2f}%")
        print(f"Grade: {evaluation_result['grade']}")
        print(f"Result PDF: {workflow_result['result_filename']}")
        print(f"{'='*70}\n")

        return jsonify({
            'success': True,
            'message': f'Paper evaluated successfully using AI concept analysis with {OLLAMA_MODEL}',
            'result_file': workflow_result['result_filename'],
            'ocr_pdf_file': workflow_result['ocr_pdf_filename'],
            'student_name': student_name,
            'total_marks': evaluation_result['total_marks'],
            'obtained_marks': evaluation_result['obtained_marks'],
            'percentage': evaluation_result['percentage'],
            'grade': evaluation_result['grade'],
            'evaluation_method': evaluation_result['evaluation_method'],
            'question_wise_summary': workflow_result['question_wise_summary']
        })

    except WorkflowError as e:
        return jsonify({'success': False, 'error': e.message}), e.status_code
    except Exception as e:
        print(f"\n✗ ERROR IN EVALUATION PIPELINE")
        print("="*70)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("="*70 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download_result/<filename>')
def download_result(filename):
    """Download result PDF"""
    try:
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if os.path.exists(result_path):
            return send_file(result_path, as_attachment=True)
        return jsonify({'success': False, 'error': 'Result file not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download_ocr_pdf/<filename>')
def download_ocr_pdf(filename):
    """Download OCR-extracted text PDF"""
    try:
        ocr_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(ocr_pdf_path):
            return send_file(ocr_pdf_path, as_attachment=True)
        return jsonify({'success': False, 'error': 'OCR PDF not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_status')
def get_status():
    """Get system status"""
    try:
        snapshot = workflow.status_snapshot()
        questions = snapshot['questions_count']
        answers = snapshot['answers_count']
        
        # Check DeepSeek initialization status
        deepseek_status = "not_available"
        if pdf_processor.deepseek_ocr is not None:
            if pdf_processor.deepseek_ocr.initialized:
                deepseek_status = "ready"
            else:
                deepseek_status = "initializing"

        status = {
            'question_paper_loaded': questions > 0,
            'answer_key_loaded': answers > 0,
            'ollama_available': ollama_evaluator is not None,
            'ollama_model': OLLAMA_MODEL,
            'deepseek_available': deepseek_status == "ready",
            'deepseek_status': deepseek_status,
            'questions_count': questions,
            'answers_count': answers
        }
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset_database', methods=['POST'])
def reset_database():
    """Clear all data from memory"""
    try:
        workflow.reset_state()
        print("\n✓✓ All data cleared successfully\n")
        return jsonify({
            'success': True, 
            'message': 'All data cleared successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AI-POWERED ANSWER EVALUATION SYSTEM - READY!")
    print("="*70)
    print("📝 Features:")
    print("  ✓ DeepSeek-OCR for handwritten text extraction")
    print(f"  ✓ Ollama {OLLAMA_MODEL} for concept-based evaluation")
    print("  ✓ Generous marking with encouraging feedback")
    print("  ✓ Automated question & answer key processing")
    print("="*70)
    print("🌐 Server starting on http://localhost:5000")
    print("="*70 + "\n")
    
    # Disable reloader to prevent threading issues with heavy models
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
