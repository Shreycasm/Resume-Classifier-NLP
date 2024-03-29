import textract
import ast
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
import string
import regex as re
from nltk.corpus import stopwords
import streamlit as st
from pyresparser import ResumeParser
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import pickle
from collections import Counter

try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()

# preprocessing function
def preprocess(text):
    if text.startswith("b'"):
        text = text[2:]
    cleaned_text = re.sub(r'[\n\t]', ' ', text)
    cleaned_text = cleaned_text.replace('\x07', '')
    cleaned_text = re.sub(r'(\\n|\\t|\\x07|\\\\)', ' ', cleaned_text)
    cleaned_text = re.sub(r'\'b\'|\'"', '', cleaned_text)
    cleaned_text = re.sub(r'(\\x[0-9a-fA-F]{2}|\\xc7\\x81|\\xe2\\x80\\x99)', '', cleaned_text)
    cleaned_text = re.sub(r'[\uf0b7/]', ' ', cleaned_text)
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    stemmed_text = ' '.join(stemmed_tokens)
    stemmed_text = stemmed_text.lower()

    
    return stemmed_text


# FUNCTION TO EXTRACT EXPERIENCE
def extract_experience(text):
    pattern = r'(\d+(\.\d+)?)\+?(?:\s*years?|yrs?|Fresher)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return f'{match.group(1)} Years'
    else:
        return "Not found"
    


# FUNCTION TO EXTRACT LANGUAGE
languages = ["english", "telugu", "kannada", "gujarati", "marathi", "hindi", "spanish", "french", 
             "malyalam", "japanese", "tamil", "bengali", "spanish", "sanskrit"]

lang_regex = {lang: re.compile(r'\b{}\b'.format(lang), re.IGNORECASE) for lang in languages}
def detect_languages(text):
    detected_languages = []
    for lang, regex in lang_regex.items():
        if regex.search(text):
            detected_languages.append(lang)
    if not detected_languages:
        detected_languages.append("Not Mentioned")

    return detected_languages


# FUNCTION TO COLOR SKILLS
def color_skills(top, candidate_skills):
    parts = len(top) // 3
    first_part = top[:10]
    second_part = top[10:parts*2]
    third_part = top[parts*2:]

    top_10_skills = []
    moderate_skills = []
    extra_skills = []
    for i in candidate_skills:
        if i in first_part:
            top_10_skills.append(f"<span style='color:green'><b>{i.capitalize()}</b></span>")
        elif i in second_part:
            moderate_skills.append(f"<span style='color:orange'><b>{i.capitalize()}</b></span>")
        elif i in third_part:
            extra_skills.append(f"<span style='color:blue'><b>{i.capitalize()}</b></span>")
        else:
            extra_skills.append(f"<span style='color:blue'><b>{i.capitalize()}</b></span>")

    return top_10_skills, moderate_skills, extra_skills

def skill_excluded(top_20, candidate_skills):
    not_mentioned = []
    for i in top_20:
        if i  not in candidate_skills:
            not_mentioned.append(f"<span style='color:red'><b>{i.capitalize()}</b></span>")
    
    return not_mentioned

# Extract Educations
def extract_education_from_resume(text):
    education = []
    education_keywords = ['Bsc', 'B. Pharmacy', 'B Pharmacy', 'Msc', 'M. Pharmacy', 'Ph.D', 'Bachelor','Bachelors', 
                          'Master', 'Masters' ,"b.sc",
                         "m.sc", "b.ca", "bca", "m.ca", "mca", "ba", "b.a", "engineering", "B.tech", "btech",
                         "mtech", "mtech", "b.e", "m.e", 'diploma', "B.com", 'bcom', "ma"]

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education

# Standardzing Education
def standardize_qualification(qualification):
    quals_lower = [i.lower() for i in qualification]
    if any(i in ["msc", "masters", "master", "mca", "mtech", "m.e", "m.ca", "m.tech"] for i in quals_lower):
        return "Masters"
    
    elif any(i in ["bsc", "bachelors", "bachelor", "bca", "btech", "b.e", "b.ca", "b.tech",
                     "b.sc", "bsc", "engineering"] for i in quals_lower):
        return "Bachelors"
    
    else:
        return "Not Mentioned"

def read_docx(file_path):
    text = textract.process(file_path)
    return text.decode('utf-8')


def save_uploaded_file(uploaded_file):
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return os.path.join("temp", uploaded_file.name)
