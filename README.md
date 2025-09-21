# Smart Meeting Participant Recommender System

## Overview
The Smart Meeting Participant Recommender System recommends the most relevant team members for a meeting based on their skills, roles, departments, and seniority. It uses NLP preprocessing (tokenization, stopword removal, lemmatization) via NLTK and TF-IDF vectorization to compute semantic similarity between a meeting description and participant profiles. Recommendations include a transparent explanation of the decision process.

## Features
- Preprocesses text with NLTK (tokenization, stopword removal, lemmatization)
- Builds TF-IDF representations of participant profiles
- Computes cosine similarity between meeting descriptions and profiles
- Extracts top contributing terms to explain each recommendation
- Simple command-line example included
