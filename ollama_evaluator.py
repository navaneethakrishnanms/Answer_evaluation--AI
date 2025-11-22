"""
Ollama-based Answer Evaluator
Performs concept-based evaluation using local LLM
"""

import json
import os
import subprocess
import re
from typing import Dict, List, Optional

class OllamaEvaluator:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Ollama evaluator
        
        Args:
            model_name: Ollama model to use (e.g., 'llama3.1:latest', 'mistral')
        """
        default_model = os.environ.get("OLLAMA_MODEL", "llama3-gpu:latest")
        self.model_name = model_name or default_model
        self.verify_ollama()
    
    def verify_ollama(self):
        """Verify Ollama is installed and model is available"""
        try:
            # Check if Ollama is installed
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                raise Exception("Ollama is not installed or not in PATH. Please install Ollama from https://ollama.ai")
            
            # Check if the model is available
            if self.model_name not in result.stdout:
                print(f"⚠️  Model '{self.model_name}' not found. Attempting to pull...")
                self.pull_model()
            
            print(f"✅ Ollama ready with model: {self.model_name}")
            
        except FileNotFoundError:
            raise Exception("Ollama is not installed. Please install from https://ollama.ai")
    
    def pull_model(self):
        """Pull the specified model if not available"""
        print(f"📥 Pulling model {self.model_name}... This may take a few minutes.")
        try:
            subprocess.run(
                ["ollama", "pull", self.model_name],
                check=True
            )
            print(f"✅ Model {self.model_name} downloaded successfully!")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to pull model {self.model_name}: {str(e)}")
    
    def _evaluate_one_mark_question(
        self,
        question_number: int,
        correct_answer: str,
        student_answer: str,
        max_marks: float
    ) -> Dict:
        """
        Evaluate 1-mark questions with strict answer key comparison.
        Awards either 1 mark or 0 marks (no partial credit).
        """
        print(f"⚡ Q{question_number}: Using strict answer key matching (1-mark question)")
        
        # Normalize both answers for comparison
        def normalize(text: str) -> str:
            # Convert to lowercase, remove extra spaces, punctuation
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalize spaces
            return text
        
        correct_normalized = normalize(correct_answer)
        student_normalized = normalize(student_answer)
        
        # Check for exact match or key phrase match
        is_correct = False
        feedback = ""
        
        # Strategy 1: Exact match after normalization
        if correct_normalized == student_normalized:
            is_correct = True
            feedback = "✓ Correct answer!"
        
        # Strategy 2: Check if answer key text is contained in student answer
        elif correct_normalized in student_normalized:
            is_correct = True
            feedback = "✓ Correct answer identified!"
        
        # Strategy 3: Check if student answer contains key parts of answer
        # Split answer key into words and check if majority are present
        elif len(correct_normalized) > 0:
            correct_words = set(correct_normalized.split())
            student_words = set(student_normalized.split())
            
            # If answer key is very short (like "B PID control"), require high match
            if len(correct_words) <= 5:
                match_ratio = len(correct_words & student_words) / len(correct_words)
                if match_ratio >= 0.8:  # 80% of words must match
                    is_correct = True
                    feedback = "✓ Correct answer!"
                else:
                    is_correct = False
                    feedback = f"✗ Incorrect. Expected: {correct_answer.strip()}"
            else:
                # For longer answers, require 70% match
                match_ratio = len(correct_words & student_words) / len(correct_words)
                if match_ratio >= 0.7:
                    is_correct = True
                    feedback = "✓ Correct answer!"
                else:
                    is_correct = False
                    feedback = f"✗ Incorrect. Expected: {correct_answer.strip()}"
        else:
            is_correct = False
            feedback = "✗ No valid answer provided"
        
        awarded_marks = 1.0 if is_correct else 0.0
        
        print(f"   Answer Key: {correct_answer.strip()[:50]}")
        print(f"   Student: {student_answer.strip()[:50]}")
        print(f"   Result: {'CORRECT' if is_correct else 'WRONG'} -> {awarded_marks} mark")
        
        # Remove Unicode characters from feedback for PDF compatibility
        feedback_clean = feedback.replace('✓', '[Correct]').replace('✗', '[Wrong]')
        
        return {
            "Question_Number": question_number,
            "Concept_Match_Score": 1.0 if is_correct else 0.0,
            "Awarded_Marks": awarded_marks,
            "Max_Marks": max_marks,
            "Feedback": feedback_clean,
            "Error": False
        }
    
    def evaluate_answer(
        self,
        question_number: int,
        question_text: str,
        max_marks: float,
        correct_answer: str,
        student_answer: str
    ) -> Dict:
        """
        Evaluate a single answer using Ollama LLM
        
        Args:
            question_number: Question number
            question_text: The question text
            max_marks: Maximum marks for this question
            correct_answer: The correct answer from answer key
            student_answer: Student's answer
            
        Returns:
            Dictionary with evaluation results
        """
        
        # FOR 1-MARK QUESTIONS: Use strict answer key matching (1 or 0 only)
        if max_marks == 1.0 or max_marks == 1:
            return self._evaluate_one_mark_question(
                question_number=question_number,
                correct_answer=correct_answer,
                student_answer=student_answer,
                max_marks=max_marks
            )
        
        # FOR OTHER QUESTIONS: Use liberal concept-based evaluation
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(
            question_text=question_text,
            max_marks=max_marks,
            correct_answer=correct_answer,
            student_answer=student_answer
        )
        
        try:
            # Call Ollama API
            response = self._call_ollama(prompt)
            
            # Parse the response
            evaluation = self._parse_evaluation_response(
                response=response,
                question_number=question_number,
                max_marks=max_marks
            )
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Error evaluating Q{question_number}: {str(e)}")
            # Return fallback evaluation
            return {
                "Question_Number": question_number,
                "Concept_Match_Score": 0.0,
                "Awarded_Marks": 0.0,
                "Max_Marks": max_marks,
                "Feedback": f"Error in evaluation: {str(e)}",
                "Error": True
            }
    
    def _create_evaluation_prompt(
        self,
        question_text: str,
        max_marks: float,
        correct_answer: str,
        student_answer: str
    ) -> str:
        """Create the evaluation prompt for Ollama with liberal marking"""
        
        prompt = f"""You are a kind and encouraging teacher who evaluates student answers liberally and generously. Your goal is to reward understanding and effort, not to penalize small mistakes.

**Question:**
{question_text}

**Maximum Marks:** {max_marks}

**Model Answer (Reference):**
{correct_answer}

**Student's Answer:**
{student_answer}

**Evaluation Guidelines (VERY IMPORTANT - Follow These Rules):**

1. **Be GENEROUS and LIBERAL** - Award marks for any correct concepts, even if not perfectly expressed
2. **Partial credit is encouraged** - If student shows any understanding, give at least 60-70% marks
3. **Different wording is acceptable** - Students don't need to match exact words
4. **Focus on concepts, not memorization** - Reward understanding over exact recall
5. **Give benefit of doubt** - If meaning is unclear but seems right, award marks
6. **Encourage effort** - Even incomplete answers deserve substantial credit if concepts are present
7. **Minor errors don't matter** - Small spelling or grammar mistakes should NOT reduce marks
8. **Related concepts count** - If student mentions related valid points, reward them

**Marking Scale (Use this as guidance):**
- If student shows ANY correct understanding: Give at least 60% of max marks
- If student covers main concept but misses details: Give 70-80% of max marks
- If student shows good understanding with minor gaps: Give 85-95% of max marks
- Only give less than 50% if answer is completely wrong or irrelevant

**Your Response:**
Respond ONLY with a valid JSON object (no other text, no markdown):
{{
  "concept_match_score": <float 0-1, be generous, typically 0.6-1.0>,
  "awarded_marks": <float 0-{max_marks}, award liberally>,
  "feedback": "<encouraging 1-2 sentence feedback highlighting what student did well>"
}}

Remember: BE KIND, BE GENEROUS, REWARD EFFORT AND UNDERSTANDING!
"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and get response"""
        try:
            # Use subprocess to call Ollama with UTF-8 encoding
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid chars instead of crashing
                check=True,
                timeout=120  # 2 minute timeout
            )
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            raise Exception("Ollama request timed out after 2 minutes")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Ollama execution failed: {e.stderr}")
    
    def _parse_evaluation_response(
        self,
        response: str,
        question_number: int,
        max_marks: float
    ) -> Dict:
        """Parse Ollama's JSON response"""
        
        try:
            # Remove any markdown formatting if present
            response = response.strip()
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Extract values
            concept_score = float(data.get("concept_match_score", 0.0))
            awarded_marks = float(data.get("awarded_marks", 0.0))
            feedback = data.get("feedback", "No feedback provided")
            
            # Validate and constrain values
            concept_score = max(0.0, min(1.0, concept_score))
            awarded_marks = max(0.0, min(max_marks, awarded_marks))
            
            return {
                "Question_Number": question_number,
                "Concept_Match_Score": round(concept_score, 3),
                "Awarded_Marks": round(awarded_marks, 2),
                "Max_Marks": max_marks,
                "Feedback": feedback,
                "Error": False
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON response: {response[:200]}")
            # Try to extract information from non-JSON response
            return self._fallback_parse(response, question_number, max_marks)
        except Exception as e:
            print(f"⚠️  Error parsing response: {str(e)}")
            return {
                "Question_Number": question_number,
                "Concept_Match_Score": 0.0,
                "Awarded_Marks": 0.0,
                "Max_Marks": max_marks,
                "Feedback": "Could not parse evaluation response",
                "Error": True
            }
    
    def _fallback_parse(self, response: str, question_number: int, max_marks: float) -> Dict:
        """Fallback parsing when JSON parsing fails"""
        
        # Try to extract numbers and keywords from response
        awarded_marks = 0.0
        concept_score = 0.0
        
        # Look for marks in text
        marks_patterns = [
            r'(\d+\.?\d*)\s*(?:marks?|points?)',
            r'(?:awarded|given|scored)\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*out of'
        ]
        
        for pattern in marks_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                awarded_marks = min(float(match.group(1)), max_marks)
                break
        
        # Estimate concept score
        if awarded_marks > 0:
            concept_score = awarded_marks / max_marks
        
        # Extract first sentence as feedback
        sentences = response.split('.')
        feedback = sentences[0].strip() if sentences else "Evaluation completed"
        
        return {
            "Question_Number": question_number,
            "Concept_Match_Score": round(concept_score, 3),
            "Awarded_Marks": round(awarded_marks, 2),
            "Max_Marks": max_marks,
            "Feedback": feedback[:200],  # Limit feedback length
            "Error": False
        }
    
    def evaluate_all_answers(
        self,
        questions: List[Dict],
        answer_key: List[Dict],
        student_answers: List[Dict]
    ) -> Dict:
        """
        Evaluate all student answers
        
        Args:
            questions: List of question dictionaries with 'question_number', 'question_text', 'max_marks'
            answer_key: List of answer dictionaries with 'question_number', 'answer_text'
            student_answers: List of student answer dictionaries with 'question_number', 'student_answer'
            
        Returns:
            Complete evaluation report
        """
        
        print(f"\n{'='*60}")
        print(f"🎓 Starting Concept-Based Evaluation with {self.model_name}")
        print(f"{'='*60}\n")
        
        # Create lookups
        answer_key_dict = {a['question_number']: a['answer_text'] for a in answer_key}
        student_answers_dict = {s['question_number']: s['student_answer'] for s in student_answers}
        
        # Debug: Show what's in the answer key
        print(f"🔍 DEBUG: Answer key mapping:")
        for q_num, ans_text in sorted(answer_key_dict.items()):
            print(f"   Q{q_num}: {ans_text[:80]}...")
        print()
        
        evaluation_results = []
        total_marks = 0
        obtained_marks = 0
        
        for question in questions:
            q_num = question['question_number']
            q_text = question['question_text']
            max_marks = question['max_marks']
            
            total_marks += max_marks
            
            # Get answer key and student answer
            correct_answer = answer_key_dict.get(q_num, "Answer key not available")
            student_answer = student_answers_dict.get(q_num, "")
            
            if not student_answer:
                # Student didn't answer
                evaluation_results.append({
                    "Question_Number": q_num,
                    "Concept_Match_Score": 0.0,
                    "Awarded_Marks": 0.0,
                    "Max_Marks": max_marks,
                    "Feedback": "Question not answered",
                    "Error": False
                })
                continue
            
            print(f"📝 Evaluating Question {q_num} (Max: {max_marks} marks)...")
            
            # Evaluate using Ollama
            evaluation = self.evaluate_answer(
                question_number=q_num,
                question_text=q_text,
                max_marks=max_marks,
                correct_answer=correct_answer,
                student_answer=student_answer
            )
            
            evaluation_results.append(evaluation)
            obtained_marks += evaluation['Awarded_Marks']
            
            print(f"   ✓ Awarded: {evaluation['Awarded_Marks']}/{max_marks} marks")
            print(f"   📊 Concept Match: {evaluation['Concept_Match_Score']*100:.1f}%")
            print(f"   💬 {evaluation['Feedback']}\n")
        
        # Calculate percentage and grade
        percentage = (obtained_marks / total_marks * 100) if total_marks > 0 else 0
        grade = self._calculate_grade(percentage)
        
        print(f"\n{'='*60}")
        print(f"📊 Evaluation Complete!")
        print(f"{'='*60}")
        print(f"Total Marks: {obtained_marks:.2f}/{total_marks}")
        print(f"Percentage: {percentage:.2f}%")
        print(f"Grade: {grade}")
        print(f"{'='*60}\n")
        
        return {
            "total_marks": total_marks,
            "obtained_marks": round(obtained_marks, 2),
            "percentage": round(percentage, 2),
            "grade": grade,
            "question_wise_results": evaluation_results,
            "evaluation_method": f"Concept-Based (Ollama {self.model_name})"
        }
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade from percentage"""
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B+"
        elif percentage >= 60:
            return "B"
        elif percentage >= 50:
            return "C"
        elif percentage >= 40:
            return "D"
        else:
            return "F"
