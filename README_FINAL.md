# 🎓 AI-Powered Answer Evaluation System

**Evaluate handwritten student papers with AI using DeepSeek OCR + Ollama LLM**

---

## 🌟 Features

✨ **Handwritten Text Recognition** - Extracts text from phone camera photos using DeepSeek OCR  
🧠 **Concept-Based Evaluation** - Uses Ollama LLM for intelligent, human-like marking  
📄 **3-Stage Processing** - OCR → Text PDF → Extraction for maximum accuracy  
📊 **Detailed Reports** - Professional PDF results with marks, feedback, and grades  
🔒 **100% Local** - All processing on your machine, complete privacy  
⚡ **Fast & Efficient** - Evaluates 10 questions in ~2-3 minutes  

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Install Ollama
```bash
# Download from: https://ollama.ai
# Then pull the model:
ollama pull llama3.1:latest
```

### 2️⃣ Install Python Dependencies
```bash
pip install -r requirements_complete.txt
```

### 3️⃣ Run the System
```bash
# Windows:
start_complete_system.bat

# Linux/Mac:
python app_ollama_complete.py
```

Open browser: **http://localhost:5000**

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART_CHECKLIST.md](QUICKSTART_CHECKLIST.md)** | Step-by-step checklist to get started |
| **[COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md)** | Comprehensive usage guide |
| **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** | Technical overview and architecture |

---

## 🔄 How It Works

```
1. Upload Question Paper → Extract questions + marks
2. Upload Answer Key    → Extract correct answers  
3. Upload Student Paper → Process with 3 stages:
   
   Stage 1: DeepSeek OCR extracts handwritten text
   Stage 2: FPDF2 creates searchable text-based PDF
   Stage 3: PyPDF2 extracts clean, parseable text
   
4. Ollama LLM evaluates concept-based understanding
5. Generate detailed result PDF with marks + feedback
```

---

## 💻 System Requirements

### Required:
- Python 3.8+
- Ollama (from https://ollama.ai)
- 8GB RAM minimum
- 10GB free disk space

### Optional (for faster OCR):
- NVIDIA GPU with CUDA support
- 16GB RAM recommended

---

## 📦 Installation

### Full Installation:

```bash
# Step 1: Install Ollama
# Download from https://ollama.ai and install

# Step 2: Pull LLM model
ollama pull llama3.1:latest

# Step 3: Install Python packages
pip install -r requirements_complete.txt

# Step 4: Test the system
python test_complete_system_new.py

# Step 5: Run the application
python app_ollama_complete.py
```

### Verify Installation:
```bash
# Check Ollama
ollama list

# Should show:
# NAME                ID              SIZE    MODIFIED
# llama3.1:latest     xxx             4.7GB   X days ago
```

---

## 🎯 Usage Example

### 1. Prepare Your Files:
- **Question Paper PDF** - Questions with marks (e.g., "Q1. What is...? (5 marks)")
- **Answer Key PDF** - Correct answers (e.g., "Answer 1: ...")
- **Student Paper PDF** - Handwritten answers from phone camera

### 2. Upload in Order:
1. Upload Question Paper → System extracts 10 questions
2. Upload Answer Key → System extracts 10 answers
3. Upload Student Paper → System processes and evaluates

### 3. Download Results:
- Click "Download Result PDF"
- View detailed evaluation with marks and feedback

---

## 📊 Sample Output

The result PDF includes:

```
╔════════════════════════════════════════╗
║  Answer Paper Evaluation Report        ║
╠════════════════════════════════════════╣
║  Total Marks:      50                  ║
║  Obtained Marks:   42.5                ║
║  Percentage:       85.00%              ║
║  Grade:            A                   ║
╚════════════════════════════════════════╝

Question-wise Evaluation:
┌────┬───────────┬──────────┬─────────────┬──────────────┐
│ Q# │ Max Marks │ Obtained │ Concept     │ Feedback     │
├────┼───────────┼──────────┼─────────────┼──────────────┤
│ 1  │ 5         │ 4.5      │ 90%         │ Excellent... │
│ 2  │ 5         │ 4.0      │ 80%         │ Good...      │
│ 3  │ 5         │ 3.5      │ 70%         │ Average...   │
│... │ ...       │ ...      │ ...         │ ...          │
└────┴───────────┴──────────┴─────────────┴──────────────┘
```

---

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_complete_system_new.py
```

Expected output:
```
✅ TEST 1: Ollama installed
✅ TEST 2: PDF Processor initialized
✅ TEST 3: OCR Converter ready
✅ TEST 4: Sample files found
✅ TEST 5: DeepSeek OCR loaded
✅ TEST 6: FPDF2 working
✅ TEST 7: ReportLab working
✅ TEST 8: Ollama evaluation working

ALL TESTS PASSED! System ready to use!
```

---

## 🔧 Troubleshooting

### "Ollama is not installed"
```bash
# Install from https://ollama.ai
ollama pull llama3.1:latest
```

### "No text extracted"
- Ensure handwriting is clear
- Use 300+ DPI when scanning
- Check good lighting and contrast

### "Module not found"
```bash
pip install -r requirements_complete.txt --upgrade
```

### Slow Performance
- Use lighter model: `phi3:latest` or `mistral:latest`
- Enable GPU if available
- Reduce number of questions for testing

See [COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md) for detailed troubleshooting.

---

## 📁 Project Structure

```
answer_evaluation_app/
├── app_ollama_complete.py              # Main Flask app
├── pdf_processor_ollama.py             # PDF processing + OCR
├── ocr_to_text_pdf_converter.py        # OCR → Text PDF
├── ollama_evaluator.py                 # Concept evaluation
├── deepseek_ocr.py                     # OCR engine
├── requirements_complete.txt           # Dependencies
├── start_complete_system.bat           # Quick start
└── test_complete_system_new.py         # Testing
```

---

## 🎓 Key Advantages

### vs Traditional OCR Systems:
✅ **3-stage processing** ensures maximum accuracy  
✅ **Text-based PDF** intermediate for reliability  
✅ **Concept understanding** not just keywords  
✅ **Flexible evaluation** accepts paraphrasing  

### vs Manual Marking:
✅ **10x faster** - Minutes instead of hours  
✅ **Consistent** - Same standards every time  
✅ **Detailed feedback** - AI explains reasoning  
✅ **Scalable** - Handle any number of papers  

---

## 🌐 Technology Stack

- **Frontend:** HTML/CSS/JavaScript
- **Backend:** Flask (Python)
- **OCR:** DeepSeek-OCR (Transformer-based)
- **AI Evaluation:** Ollama (Local LLM)
- **PDF Processing:** PyPDF2, FPDF2, ReportLab
- **ML Framework:** PyTorch

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| OCR Accuracy | 85-95% |
| Evaluation Time (10Q) | 2-3 minutes |
| Processing Stages | 3 (OCR→PDF→Extract) |
| Privacy | 100% Local |
| Concept Understanding | Yes (LLM-based) |

---

## 🔐 Privacy

- ✅ All processing **100% local**
- ✅ No data sent to cloud
- ✅ No external API calls
- ✅ Complete data ownership
- ✅ Student papers stay private

---

## 🎯 Use Cases

- **Teachers:** Automate answer paper marking
- **Tutors:** Quick evaluation with detailed feedback
- **Educational Institutions:** Scale up evaluation process
- **Students:** Self-assessment with AI feedback
- **Online Courses:** Automated grading system

---

## 🚦 System Status Indicators

### ✅ Working Correctly:
- Question paper uploads and shows question count
- Answer key uploads and shows answer count
- Student paper creates text PDF in uploads/
- Evaluation completes without errors
- Result PDF downloads with all details

### ❌ Needs Attention:
- Ollama not responding
- OCR extraction fails
- No text extracted from PDFs
- Evaluation errors in console

---

## 📞 Support

1. **Quick Issues:** Check [QUICKSTART_CHECKLIST.md](QUICKSTART_CHECKLIST.md)
2. **Detailed Help:** See [COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md)
3. **Technical:** Read [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
4. **Testing:** Run `python test_complete_system_new.py`

---

## 🎉 Success Metrics

You'll know the system is working when:
- ✅ Ollama evaluates and provides feedback
- ✅ Text-based PDFs are created from handwritten input
- ✅ PyPDF2 extracts text successfully
- ✅ Result PDF shows concept match scores
- ✅ Grades and marks are accurately calculated
- ✅ No errors in the console

---

## 📝 License

This project uses:
- DeepSeek-OCR: Apache 2.0
- Ollama: MIT License
- FPDF2: LGPL
- Flask: BSD License

---

## 🙏 Credits

Built with:
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) for handwritten text extraction
- [Ollama](https://ollama.ai) for local LLM inference
- [FPDF2](https://github.com/py-pdf/fpdf2) for PDF generation
- [PyPDF2](https://github.com/py-pdf/pypdf) for PDF text extraction
- [ReportLab](https://www.reportlab.com/) for result PDFs
- [Flask](https://flask.palletsprojects.com/) for web interface

---

## 🚀 Get Started Now!

```bash
# 1. Install Ollama from https://ollama.ai
ollama pull llama3.1:latest

# 2. Install dependencies
pip install -r requirements_complete.txt

# 3. Test the system
python test_complete_system_new.py

# 4. Run the app
start_complete_system.bat

# 5. Open browser
# http://localhost:5000

# 6. Upload papers and evaluate!
```

---

**Built with ❤️ for educators and students**

*Making AI-powered education accessible to everyone!* 🎓✨
