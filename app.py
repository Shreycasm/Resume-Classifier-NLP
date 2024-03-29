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
from utils import (preprocess, extract_experience, detect_languages, color_skills, skill_excluded,
                   extract_education_from_resume, standardize_qualification, read_docx, save_uploaded_file)



st.set_page_config(page_title ="Resume Classifier")

# LOADING MODEL
with open("./artifacts/grid_search.pkl", "rb") as f:
    model = pickle.load(f)

# LOADING PREPROCESSED RANNING CSV
train_df = pd.read_csv("./artifacts/final_df.csv")

def main():
    st.markdown(f'<h1 style="text-align:center;">Resume Classifier</h1>', unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])
    button = st.button("Get Details")

    if button:
        if uploaded_file is None:
            st.warning("Please upload a file before clicking 'Get Details'.")
        else:
            tab1, tab2 = st.tabs(["Details", "Resume"])
            with tab1:
                name = []
                content = []
                skills = []

                file_path = save_uploaded_file(uploaded_file)
                content.append(read_docx(file_path))
                data = ResumeParser(uploaded_file).get_extracted_data()
                skills.append(data["skills"])

                file_name = uploaded_file.name

                if len(file_name.split("_")) != 1:
                    if (file_name.split("_")[1].split('.')[0] == "Hexaware" or
                            file_name.split("_")[1].split('.')[0] == "(Hexaware)"):
                        name.append(file_name.split("_")[0])
                    else:
                        name.append(file_name.split("_")[1].split('.')[0])
                else:
                    name.append(file_name.split('.')[0])

                df = pd.DataFrame(data={"content": content,
                                         "name": name,
                                         "skills": skills})

                df["clean"] = df["content"].apply(preprocess)
                df["exp"] = df["content"].apply(extract_experience)
                df['detected_languages'] = df['content'].apply(detect_languages)
                df["qualification"] = df["content"].apply(extract_education_from_resume)
                df['qualification'] = df['qualification'].apply(standardize_qualification)

                prediction = model.predict(df["content"])
                if prediction == 0:
                    label = 'PeopleSoft'
                elif prediction == 1:
                    label = "React Developer"
                elif prediction == 2:
                    label = "SQL Developer"
                else:
                    label = "Workday"

                detected_langs = df['detected_languages'][0]
                formatted_langs = ", ".join(lang.capitalize() for lang in detected_langs)

                train_df["skills"] = train_df["skills"].apply(ast.literal_eval)
                all_skills = [skill for sublist in train_df[train_df["label"] == prediction[0]]["skills"] for skill
                              in sublist]
                skill_counts = Counter(all_skills)
                top_skills = skill_counts.most_common()
                top_20_skills = skill_counts.most_common(25)
                all_skills = [skill[0] for skill in top_skills]
                top_20_skills = [skill[0] for skill in top_20_skills]
                candidate_skills = df["skills"][0]
                imp, mod, extra = color_skills(all_skills, candidate_skills)
                not_mentioned = skill_excluded(top_20_skills, candidate_skills)

                st.subheader(f'*Name:* {df["name"][0]}')
                st.write(f"*Experience:* {df['exp'][0]}")
                st.write(f"*Qualification:* {df['qualification'][0]} ")
                st.write(f"*Languages known:* {formatted_langs}")
                st.markdown(f'*Important Skills:* {", ".join(imp)}', unsafe_allow_html=True)
                st.markdown(f'*Moderate Skills:* {", ".join(mod)}', unsafe_allow_html=True)
                st.markdown(f'*Extra Skills:* {", ".join(extra)}', unsafe_allow_html=True)
                st.markdown(f"*Missing Required Skills:* {', '.join(not_mentioned)}", unsafe_allow_html=True)

                st.markdown(f'<h3 style="text-align:center;">Potential candidate for <span style="color:purple">{label}</span> Profile</h3>', 
                            unsafe_allow_html=True)


            with tab2:
                st.write(df["content"][0])
    else:
        st.warning("Click the 'Get Details' button after uploading the file.")


if __name__ == "__main__":
    main()



















