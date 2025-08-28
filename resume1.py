#!/usr/bin/env python
# coding: utf-8

# In[1]:

import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def extract_sorted_skills(file_path):
    # API request inside the function 
    url = "http://crmgpu6-10042:37991/image/api/resume/parse"
    payload = {}
    files = [
        ('resume', (file_path.split("/")[-1], open(file_path, 'rb'), 'application/pdf'))
    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    parsed_data = response.json().get('parsed_json', {})

    # Load the model 
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    sections = []
    weights = []

    # Define weights for each section
    section_weights = {
        "work_experience": 10,
        "profile": 10,
        "achievement": 10,
        "certification": 10,
        "project": 10
    }

    # Work experiences
    for exp in parsed_data.get('work_experiences', []):
        work_desc = exp.get('work_description')
        if isinstance(work_desc, list):
            work_desc = " ".join(work_desc)
        if work_desc and work_desc.strip():
            sections.append(work_desc)
            weights.append(section_weights["work_experience"])

    # Profile description
    profile_desc = parsed_data.get('primary_information', {}).get('profile_description', '')
    if profile_desc and profile_desc.strip():
        sections.append(profile_desc)
        weights.append(section_weights["profile"])

    # Achievements
    for ach_desc in parsed_data.get("achievements", []):
        if ach_desc:
            sections.append(ach_desc)
            weights.append(section_weights["achievement"])

    # Certifications
    for cert in parsed_data.get("certifications_description", []):
        sections.append(cert)
        weights.append(section_weights["certification"])

    # Projects
    for proj in parsed_data.get("projects_description", []):
        sections.append(proj)
        weights.append(section_weights["project"])

    # Skills
    skills = []
    for skill in parsed_data.get('skills', {}).get('hard_skills', []):
        if skill.get('parsed_value'):
            skills.append(skill.get('parsed_value'))

    for skill in parsed_data.get('skills', {}).get('soft_skills', []):
        if skill.get('parsed_value'):
            skills.append(skill.get('parsed_value'))

    # Embeddings Similarity 
    if sections and skills:
        work_embeddings = model.encode(sections, convert_to_tensor=True)
        skill_embeddings = model.encode(skills, convert_to_tensor=True)

        similarity = util.cos_sim(work_embeddings, skill_embeddings).cpu().numpy()

        
        weighted_similarity = (similarity.T * weights).T  

        
        df = pd.DataFrame(weighted_similarity, 
                          index=[f'w_{i+1}' for i in range(len(sections))], 
                          columns=skills)

        
        sorted_skills = df.mean().sort_values(ascending=False)

        return sorted_skills
    else:
        return None



file_path = "C:/Users/shari/Desktop/Zoho/Accounting and Finance/v10Accounting and finance.pdf"
sorted_skills = extract_sorted_skills(file_path)

if sorted_skills is not None:
    print("Sorted Skills by Similarity:\n", sorted_skills)
else:
    print("No skills or sections available.")
