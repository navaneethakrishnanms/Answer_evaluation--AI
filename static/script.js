// Global variables
let currentResultFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setupDragAndDrop();
});

// Check status of uploaded files
async function checkStatus() {
    try {
        const response = await fetch('/get_status');
        const data = await response.json();
        
        if (data.success) {
            updateStatusDisplay(data.status);
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

function updateStatusDisplay(status) {
    const questionStatus = document.getElementById('questionPaperStatus');
    const answerStatus = document.getElementById('answerKeyStatus');
    
    if (status.question_paper_loaded) {
        questionStatus.textContent = 'Loaded ✓';
        questionStatus.classList.add('loaded');
    } else {
        questionStatus.textContent = 'Not Loaded';
        questionStatus.classList.remove('loaded');
    }
    
    if (status.answer_key_loaded) {
        answerStatus.textContent = 'Loaded ✓';
        answerStatus.classList.add('loaded');
    } else {
        answerStatus.textContent = 'Not Loaded';
        answerStatus.classList.remove('loaded');
    }
    
    // Enable/disable evaluate button
    const evaluateBtn = document.getElementById('evaluateBtn');
    if (status.question_paper_loaded && status.answer_key_loaded) {
        evaluateBtn.disabled = false;
    } else {
        evaluateBtn.disabled = true;
    }
}

// Setup drag and drop
function setupDragAndDrop() {
    const uploadAreas = [
        { area: 'questionUploadArea', input: 'questionPaper' },
        { area: 'answerUploadArea', input: 'answerKey' },
        { area: 'studentUploadArea', input: 'studentPaper' }
    ];
    
    uploadAreas.forEach(item => {
        const area = document.getElementById(item.area);
        const input = document.getElementById(item.input);
        
        area.addEventListener('click', () => input.click());
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', () => {
            area.classList.remove('dragover');
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                updateFileName(item.area, e.dataTransfer.files[0].name);
            }
        });
        
        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                updateFileName(item.area, e.target.files[0].name);
            }
        });
    });
}

function updateFileName(areaId, fileName) {
    const area = document.getElementById(areaId);
    let fileNameDiv = area.querySelector('.file-name');
    
    if (!fileNameDiv) {
        fileNameDiv = document.createElement('div');
        fileNameDiv.className = 'file-name';
        area.appendChild(fileNameDiv);
    }
    
    fileNameDiv.textContent = `📎 ${fileName}`;
    fileNameDiv.classList.add('show');
}

// Upload question paper
async function uploadQuestionPaper() {
    const fileInput = document.getElementById('questionPaper');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('questionMessage', 'Please select a PDF file', 'error');
        return;
    }
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showMessage('questionMessage', 'Please upload a PDF file only', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showProgress('questionProgress');
    showMessage('questionMessage', 'Uploading and processing question paper...', 'info');
    
    try {
        const response = await fetch('/upload_question_paper', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideProgress('questionProgress');
        
        if (data.success) {
            showMessage('questionMessage', 
                `✓ Question paper uploaded successfully! Found ${data.questions_count} questions.`, 
                'success');
            checkStatus();
        } else {
            showMessage('questionMessage', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        hideProgress('questionProgress');
        showMessage('questionMessage', `Error: ${error.message}`, 'error');
    }
}

// Upload answer key
async function uploadAnswerKey() {
    const fileInput = document.getElementById('answerKey');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('answerMessage', 'Please select a PDF file', 'error');
        return;
    }
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showMessage('answerMessage', 'Please upload a PDF file only', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showProgress('answerProgress');
    showMessage('answerMessage', 'Uploading and processing answer key...', 'info');
    
    try {
        const response = await fetch('/upload_answer_key', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideProgress('answerProgress');
        
        if (data.success) {
            showMessage('answerMessage', 
                `✓ Answer key uploaded successfully! Found ${data.answers_count} answers.`, 
                'success');
            checkStatus();
        } else {
            showMessage('answerMessage', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        hideProgress('answerProgress');
        showMessage('answerMessage', `Error: ${error.message}`, 'error');
    }
}

// Evaluate student paper
async function evaluateStudentPaper() {
    const fileInput = document.getElementById('studentPaper');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('studentMessage', 'Please select a PDF file', 'error');
        return;
    }
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showMessage('studentMessage', 'Please upload a PDF file only', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showProgress('studentProgress');
    showMessage('studentMessage', 'Starting evaluation... This may take 2-5 minutes for the first student paper (AI model is loading and processing handwriting).', 'info');
    
    try {
        // Increase timeout to 10 minutes for first OCR loading
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes
        
        const response = await fetch('/evaluate_student_paper', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const data = await response.json();
        
        hideProgress('studentProgress');
        
        if (data.success) {
            showMessage('studentMessage', '✓ Evaluation completed successfully!', 'success');
            displayResults(data);
            currentResultFile = data.result_file;
        } else {
            showMessage('studentMessage', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        hideProgress('studentProgress');
        if (error.name === 'AbortError') {
            showMessage('studentMessage', 'Error: Request timed out. The AI model may still be loading. Please try again in a minute.', 'error');
        } else {
            showMessage('studentMessage', `Error: ${error.message}`, 'error');
        }
    }
}

// Display results
function displayResults(data) {
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    
    // Update summary
    document.getElementById('totalMarks').textContent = data.total_marks;
    document.getElementById('obtainedMarks').textContent = data.obtained_marks;
    document.getElementById('percentage').textContent = data.percentage + '%';
    
    // Calculate and display grade
    const grade = calculateGrade(data.percentage);
    document.getElementById('gradeValue').textContent = grade;
    
    // Update table
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '';
    
    const rows = data.question_wise_summary || [];
    rows.forEach(question => {
        const row = tbody.insertRow();
        const percentage = question.max_marks > 0 
            ? ((question.obtained_marks / question.max_marks) * 100).toFixed(1) 
            : 0;
        
        row.innerHTML = `
            <td><strong>Q${question.question_number}</strong></td>
            <td>${question.max_marks}</td>
            <td>${question.obtained_marks}</td>
            <td>${percentage}%</td>
            <td>${question.feedback || 'N/A'}</td>
        `;
        
        // Color code based on percentage
        if (percentage >= 80) {
            row.style.background = '#d4edda';
        } else if (percentage >= 60) {
            row.style.background = '#fff3cd';
        } else if (percentage < 50) {
            row.style.background = '#f8d7da';
        }
    });
}

// Calculate grade
function calculateGrade(percentage) {
    if (percentage >= 90) return 'A+';
    if (percentage >= 80) return 'A';
    if (percentage >= 70) return 'B+';
    if (percentage >= 60) return 'B';
    if (percentage >= 50) return 'C';
    if (percentage >= 40) return 'D';
    return 'F';
}

// Download result PDF
function downloadResult() {
    if (currentResultFile) {
        window.location.href = `/download_result/${currentResultFile}`;
    } else {
        alert('No result file available to download');
    }
}

// Helper functions
function showProgress(elementId) {
    document.getElementById(elementId).style.display = 'block';
}

function hideProgress(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

function showMessage(elementId, message, type) {
    const messageElement = document.getElementById(elementId);
    messageElement.textContent = message;
    messageElement.className = `message ${type}`;
    messageElement.style.display = 'block';
}
