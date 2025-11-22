"""
Debug script to check what's stored in ChromaDB
"""
import chromadb
import os

# Initialize ChromaDB client
persist_directory = './vector_db'
client = chromadb.PersistentClient(path=persist_directory)

print("="*60)
print("ChromaDB Contents Checker")
print("="*60)

# Get all collections
collections = client.list_collections()
print(f"\nFound {len(collections)} collections:")
for coll in collections:
    print(f"  - {coll.name}")

# Check question collection
try:
    question_collection = client.get_collection(name="question_papers")
    question_data = question_collection.get(include=['metadatas', 'documents'])
    
    print("\n" + "="*60)
    print("QUESTION PAPERS COLLECTION")
    print("="*60)
    print(f"Total questions stored: {len(question_data['ids'])}")
    
    if question_data['ids']:
        print("\nQuestions found:")
        for i, (qid, metadata, doc) in enumerate(zip(question_data['ids'], 
                                                       question_data['metadatas'], 
                                                       question_data['documents'])):
            print(f"\n{i+1}. ID: {qid}")
            print(f"   Question Number: {metadata.get('question_number')}")
            print(f"   Max Marks: {metadata.get('max_marks')}")
            print(f"   Text: {metadata.get('question_text', 'N/A')[:100]}...")
    else:
        print("\n⚠️ No questions found in database!")
        
except Exception as e:
    print(f"\n❌ Error reading question collection: {e}")

# Check answer collection
try:
    answer_collection = client.get_collection(name="answer_keys")
    answer_data = answer_collection.get(include=['metadatas', 'documents'])
    
    print("\n" + "="*60)
    print("ANSWER KEYS COLLECTION")
    print("="*60)
    print(f"Total answers stored: {len(answer_data['ids'])}")
    
    if answer_data['ids']:
        print("\nAnswers found:")
        for i, (aid, metadata, doc) in enumerate(zip(answer_data['ids'], 
                                                      answer_data['metadatas'], 
                                                      answer_data['documents'])):
            print(f"\n{i+1}. ID: {aid}")
            print(f"   Question Number: {metadata.get('question_number')}")
            print(f"   Answer: {metadata.get('answer_text', 'N/A')[:100]}...")
    else:
        print("\n⚠️ No answers found in database!")
        
except Exception as e:
    print(f"\n❌ Error reading answer collection: {e}")

print("\n" + "="*60)
print("END OF REPORT")
print("="*60)
