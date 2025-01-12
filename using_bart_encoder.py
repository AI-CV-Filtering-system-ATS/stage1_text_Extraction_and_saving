# python version : 3.11

# Please note that you need to download tesseract and poppler and set them up as environment variables in your machine

# IMPORTING LIBRARIES
import os
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# Configuration for pytesseract
myconfig = r'--oem 3 --psm 6'


# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Initialize the encoder model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

job_role_keywords = {
    "Software Developer": [
        "programming", "python", "java", "c++", "git", "version control", "algorithms",
        "data structures", "object-oriented", "testing", "debugging", "problem-solving",
        "api", "full stack", "backend", "frontend", "sql", "nosql", "rest", "graphql",
        "typescript", "json", "xml", "design patterns", "code optimization"
    ],

    "Data Scientist": [
        "machine learning", "data analysis", "python", "r", "statistics", "data mining",
        "modeling", "big data", "data visualization", "deep learning", "pandas", "numpy",
        "matplotlib", "neural networks", "artificial intelligence", "regression",
        "classification", "clustering", "scikit-learn", "tensorflow", "keras", "hadoop",
        "spark", "databricks", "etl", "data engineering", "data pipelines"
    ],

    "Web Developer": [
        "html", "css", "javascript", "web", "react", "angular", "frontend", "backend",
        "php", "sql", "mysql", "bootstrap", "node.js", "express", "ui/ux", "web design",
        "responsive design", "jquery", "ajax", "api integration", "web sockets", "django",
        "flask", "tailwind", "svelte", "wordpress", "seo", "cross-browser compatibility"
    ],

    "Mobile App Developer": [
        "android", "ios", "flutter", "react native", "kotlin", "swift", "mobile app",
        "development", "mobile ui", "mobile ux", "cross-platform", "native app",
        "app store", "google play", "firebase", "push notifications", "in-app purchases",
        "sqlite", "admob", "geolocation", "augmented reality", "camera integration"
    ],

    "Project Manager": [
        "leadership", "project management", "team", "planning", "agile", "scrum", "jira",
        "kanban", "waterfall", "communication", "risk management", "budgeting",
        "stakeholder management", "scheduling", "milestones", "resource allocation",
        "time management", "project charter", "gantt chart", "product roadmap",
        "cost estimation", "change management", "team performance", "conflict resolution"
    ],

    "DevOps Engineer": [
        "docker", "kubernetes", "ci/cd", "automation", "aws", "azure", "cloud",
        "infrastructure", "terraform", "ansible", "jenkins", "linux", "bash scripting",
        "monitoring", "logging", "scaling", "containers", "orchestration", "security",
        "microservices", "networking", "prometheus", "grafana", "cloudformation",
        "cloudwatch", "load balancing", "serverless", "elk stack", "splunk", "helm"
    ],

    "Cybersecurity Specialist": [
        "network security", "penetration testing", "vulnerability assessment", "firewalls",
        "intrusion detection", "incident response", "encryption", "ethical hacking", "siem",
        "splunk", "malware analysis", "forensics", "iso 27001", "nist", "gdpr", "pci dss",
        "threat hunting", "cyber risk", "phishing", "security audits", "zero trust"
    ],

    "Database Administrator": [
        "sql", "mysql", "postgresql", "oracle", "nosql", "mongodb", "database design",
        "query optimization", "etl", "data migration", "backup", "recovery", "indexing",
        "replication", "sharding", "database security", "schema design", "t-sql",
        "stored procedures", "data integrity", "performance tuning", "cloud databases"
    ],

    "AI/ML Engineer": [
        "machine learning", "deep learning", "tensorflow", "keras", "pytorch",
        "computer vision", "natural language processing", "neural networks",
        "scikit-learn", "huggingface", "transformers", "reinforcement learning",
        "data preprocessing", "feature engineering", "model evaluation", "hyperparameter tuning",
        "gradient descent", "convolutional neural networks", "recurrent neural networks"
    ],

    "Cloud Engineer": [
        "aws", "azure", "google cloud", "cloud architecture", "cloud migration", "terraform",
        "cloud security", "scaling", "load balancing", "cloud storage", "s3", "ec2", "cloudformation",
        "networking", "vpc", "cloud monitoring", "serverless computing", "kubernetes",
        "cloud orchestration", "cost optimization"
    ],

    "Game Developer": [
        "unity", "unreal engine", "c#", "c++", "game physics", "3d modeling", "animation",
        "game design", "level design", "ai for games", "shader programming", "vr",
        "ar", "game engines", "multiplayer", "game optimization", "debugging", "game scripting"
    ]
}


def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)


def ocr_image_with_word_boxes(image, myconfig):
    return pytesseract.image_to_data(image, config=myconfig)


def pil_to_opencv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def extract_text_from_pdf(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    high_confidence_words = []

    for pil_img in images:
        img = pil_to_opencv(pil_img)
        data = ocr_image_with_word_boxes(img, myconfig)

        for i, line in enumerate(data.splitlines()):
            if i == 0:
                continue
            fields = line.split("\t")
            if len(fields) >= 12 and fields[11].strip():
                word = fields[11]
                try:
                    confidence = float(fields[10])  # Convert to float
                    if confidence >= 80.0:  # Use float for comparison
                        high_confidence_words.append(word)
                except ValueError:
                    continue  # In case of any parsing errors, skip this line

    return " ".join(high_confidence_words)


def summarize_text(text, min_length=200, max_length=250):
    if len(text.split()) < min_length:
        return text
    summary = summarizer(text, min_length=min_length, max_length=max_length, truncation=True)
    return summary[0]["summary_text"]


def match_job_roles(raw_text, job_role_keywords):
    job_matches = {}
    words = raw_text.lower().split()

    for job_role, keywords in job_role_keywords.items():
        matched_keywords = [keyword for keyword in keywords if keyword in words]
        job_matches[job_role] = len(matched_keywords)

    sorted_matches = sorted(job_matches.items(), key=lambda x: x[1], reverse=True)
    possible_roles = [f"{role} (Matched Keywords: {count})" for role, count in sorted_matches if count > 0]
    return possible_roles


def process_cv_folder(folder_path):
    all_cv_details = {}
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            print(f"Processing {file}...")
            extracted_text = extract_text_from_pdf(pdf_path)
            summarized_text = summarize_text(extracted_text)
            possible_roles = match_job_roles(extracted_text, job_role_keywords)

            all_cv_details[file] = {
                "Raw Text": extracted_text,
                "Summarized Text": summarized_text,
                "Possible Job Roles": possible_roles,
            }
    return all_cv_details


def main():
    folder_path = input("Enter the path to the folder containing CV PDFs: ").strip()
    if not os.path.exists(folder_path):
        print("Invalid folder path. Exiting...")
        return

    all_cv_details = process_cv_folder(folder_path)

    for cv, details in all_cv_details.items():
        print(f"\nDetails for {cv}:")
        print("Summarized Text:", details["Summarized Text"])
        print("Possible Job Roles:", details["Possible Job Roles"])

    while True:
        prompt = input("\nWould you like to give a custom prompt for ranking (yes/no)? ").strip().lower()
        if prompt == "no":
            print("Exiting...")
            break
        elif prompt == "yes":
            custom_prompt = input("Enter your custom prompt: ").strip().lower()

            # Vectorize summarized CVs and User Prompt using TF-IDF
            summarized_texts = [details["Raw Text"] for details in all_cv_details.values()]

            #################### PREV VERSION (Using TFIDF)
            # tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10)  # Limit to top features
            # cv_tfidf_matrix = tfidf_vectorizer.fit_transform(summarized_texts)  # Fit and transform summarized CV texts
            # custom_prompt_tfidf = tfidf_vectorizer.transform([custom_prompt])  # Transform user prompt
            #
            # # Calculate cosine similarity between user prompt and each summarized CV
            # similarities = cosine_similarity(custom_prompt_tfidf, cv_tfidf_matrix)[0]  # 1D array of similarities
            #
            # # Rank CVs by similarity score
            # ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order
            # ranked_cvs = [(list(all_cv_details.keys())[i], similarities[i]) for i in ranked_indices]
            ###################

            # Using BART Encoder model to encode the texts and get similarity
            cv_embeddings = encoder.encode(summarized_texts, convert_to_tensor=True)
            prompt_embedding = encoder.encode(custom_prompt, convert_to_tensor=True)

            # Calculate cosine similarity
            cosine_scores = util.cos_sim(prompt_embedding, cv_embeddings)

            # Rank CVs by similarity
            ranked_indices = cosine_scores[0].argsort(descending=True)
            ranked_cvs = [(list(all_cv_details.keys())[i], cosine_scores[0][i].item()) for i in ranked_indices]

            # Display ranked CVs
            print("\nRanked CVs based on similarity to user prompt:")
            for rank, (cv_path, similarity) in enumerate(ranked_cvs, 1):
                print(f"Rank {rank}: {cv_path} (Similarity Score: {similarity:.2f})")

            # Display the most relevant CV's summarized text
            if ranked_cvs:
                most_relevant_cv_path = ranked_cvs[0][0]
                print(f"\nMost Relevant CV ({most_relevant_cv_path}):\n")
                print(all_cv_details[most_relevant_cv_path]["Summarized Text"])
        else:
            print("Invalid input. Please type 'yes' or 'no'.")


if __name__ == "__main__":
    print("Welcome to the CV Analyzer!")
    main()