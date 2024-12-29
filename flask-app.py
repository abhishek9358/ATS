from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ResumeAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_text_from_docx(self, docx_file):
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def analyze_job_description(self, job_description):
        doc = nlp(job_description)
        
        # Extract key skills and requirements
        skills = []
        requirements = []
        
        for ent in doc.ents:
            if ent.label_ in ["SKILL", "PRODUCT"]:
                skills.append(ent.text)
            elif ent.label_ in ["REQUIREMENT"]:
                requirements.append(ent.text)
                
        # Extract additional keywords using noun chunks
        keywords = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to phrases of 3 words or less
                keywords.append(chunk.text.lower())
                
        return {
            "skills": list(set(skills)),
            "requirements": list(set(requirements)),
            "keywords": list(set(keywords))
        }
    
    def calculate_ats_score(self, resume_text, job_description):
        # Vectorize texts
        texts = [resume_text, job_description]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage
        ats_score = int(similarity * 100)
        
        # Analyze missing keywords
        job_analysis = self.analyze_job_description(job_description)
        resume_text_lower = resume_text.lower()
        
        missing_keywords = []
        
        # Check for missing skills and requirements
        all_important_terms = (
            job_analysis['skills'] + 
            job_analysis['requirements'] + 
            job_analysis['keywords']
        )
        
        for term in all_important_terms:
            if term.lower() not in resume_text_lower:
                missing_keywords.append(term)
        
        return {
            "score": ats_score,
            "missing_keywords": list(set(missing_keywords))
        }
        
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400
    
    resume_file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if not job_description:
        return jsonify({"error": "No job description provided"}), 400
    
    if resume_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(resume_file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        analyzer = ResumeAnalyzer()
        
        # Extract text from resume
        if resume_file.filename.endswith('.pdf'):
            resume_text = analyzer.extract_text_from_pdf(resume_file)
        else:
            resume_text = analyzer.extract_text_from_docx(resume_file)
        
        # Calculate ATS score
        analysis_result = analyzer.calculate_ats_score(resume_text, job_description)
        
        return jsonify({
            "success": True,
            "ats_score": analysis_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)