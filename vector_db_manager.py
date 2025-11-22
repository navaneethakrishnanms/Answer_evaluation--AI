import chromadb
from chromadb.config import Settings
import json
import os
from sentence_transformers import SentenceTransformer

class VectorDBManager:
    def __init__(self, persist_directory='./vector_db'):
        """Initialize ChromaDB for storing questions and answers"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collections
        try:
            self.question_collection = self.client.get_or_create_collection(
                name="question_papers",
                metadata={"description": "Question papers with marks"}
            )
            
            self.answer_collection = self.client.get_or_create_collection(
                name="answer_keys",
                metadata={"description": "Answer keys for questions"}
            )
            
            print("✅ Collections initialized successfully")
        except Exception as e:
            print(f"Error initializing collections: {str(e)}")
            # Set collections to None if initialization fails
            self.question_collection = None
            self.answer_collection = None
    
    def store_question_paper(self, questions_data):
        """Store question paper in vector database"""
        try:
            # Check if collection is initialized
            if self.question_collection is None:
                print("⚠️ Question collection not initialized, reinitializing...")
                self.question_collection = self.client.get_or_create_collection(
                    name="question_papers",
                    metadata={"description": "Question papers with marks"}
                )
            
            # Clear existing questions
            existing_ids = self.question_collection.get()['ids']
            if existing_ids:
                self.question_collection.delete(ids=existing_ids)
            
            if not questions_data:
                print("No questions to store")
                return
            
            # Prepare data for storage
            documents = []
            metadatas = []
            ids = []
            
            for q in questions_data:
                doc_text = f"Question {q['question_number']}: {q['question_text']}"
                documents.append(doc_text)
                
                metadatas.append({
                    'question_number': q['question_number'],
                    'max_marks': q['max_marks'],
                    'question_text': q['question_text']
                })
                
                ids.append(f"q_{q['question_number']}")
            
            # Add to collection
            self.question_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Stored {len(questions_data)} questions in vector DB")
            
        except Exception as e:
            print(f"Error storing question paper: {str(e)}")
            raise
    
    def store_answer_key(self, answers_data):
        """Store answer key in vector database"""
        try:
            # Check if collection is initialized
            if self.answer_collection is None:
                print("⚠️ Answer collection not initialized, reinitializing...")
                self.answer_collection = self.client.get_or_create_collection(
                    name="answer_keys",
                    metadata={"description": "Answer keys for questions"}
                )
            
            # Clear existing answers
            existing_ids = self.answer_collection.get()['ids']
            if existing_ids:
                self.answer_collection.delete(ids=existing_ids)
            
            if not answers_data:
                print("No answers to store")
                return
            
            # Prepare data for storage
            documents = []
            metadatas = []
            ids = []
            
            for a in answers_data:
                doc_text = f"Answer {a['question_number']}: {a['answer_text']}"
                documents.append(doc_text)
                
                metadatas.append({
                    'question_number': a['question_number'],
                    'answer_text': a['answer_text']
                })
                
                ids.append(f"a_{a['question_number']}")
            
            # Add to collection
            self.answer_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Stored {len(answers_data)} answers in vector DB")
            
        except Exception as e:
            print(f"Error storing answer key: {str(e)}")
            raise
    
    def get_question_by_number(self, question_number):
        """Retrieve a specific question by number"""
        try:
            result = self.question_collection.get(
                ids=[f"q_{question_number}"],
                include=['metadatas']
            )
            
            if result['ids']:
                return result['metadatas'][0]
            return None
            
        except Exception as e:
            print(f"Error retrieving question {question_number}: {str(e)}")
            return None
    
    def get_answer_by_number(self, question_number):
        """Retrieve a specific answer by question number"""
        try:
            result = self.answer_collection.get(
                ids=[f"a_{question_number}"],
                include=['metadatas']
            )
            
            if result['ids']:
                return result['metadatas'][0]
            return None
            
        except Exception as e:
            print(f"Error retrieving answer {question_number}: {str(e)}")
            return None
    
    def get_all_questions(self):
        """Retrieve all questions"""
        try:
            result = self.question_collection.get(include=['metadatas'])
            
            questions = []
            for metadata in result['metadatas']:
                questions.append(metadata)
            
            # Sort by question number
            questions = sorted(questions, key=lambda x: x['question_number'])
            return questions
            
        except Exception as e:
            print(f"Error retrieving all questions: {str(e)}")
            return []
    
    def get_all_answers(self):
        """Retrieve all answers"""
        try:
            result = self.answer_collection.get(include=['metadatas'])
            
            answers = []
            for metadata in result['metadatas']:
                answers.append(metadata)
            
            # Sort by question number
            answers = sorted(answers, key=lambda x: x['question_number'])
            return answers
            
        except Exception as e:
            print(f"Error retrieving all answers: {str(e)}")
            return []
    
    def has_question_paper(self):
        """Check if question paper is loaded"""
        try:
            result = self.question_collection.get()
            return len(result['ids']) > 0
        except:
            return False
    
    def has_answer_key(self):
        """Check if answer key is loaded"""
        try:
            result = self.answer_collection.get()
            return len(result['ids']) > 0
        except:
            return False
    
    def search_similar_answers(self, query_text, n_results=3):
        """Search for similar answers using semantic search"""
        try:
            results = self.answer_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['metadatas', 'distances']
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching similar answers: {str(e)}")
            return None
    
    def clear_all_data(self):
        """Clear all stored data"""
        try:
            # Delete question collection
            question_ids = self.question_collection.get()['ids']
            if question_ids:
                self.question_collection.delete(ids=question_ids)
            
            # Delete answer collection
            answer_ids = self.answer_collection.get()['ids']
            if answer_ids:
                self.answer_collection.delete(ids=answer_ids)
            
            print("All data cleared from vector DB")
            
        except Exception as e:
            print(f"Error clearing data: {str(e)}")
