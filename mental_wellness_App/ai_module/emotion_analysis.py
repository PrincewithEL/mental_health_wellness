import pandas as pd
import numpy as np
from django.contrib.staticfiles import finders
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_response_data() -> pd.DataFrame:
    try:
        path = finders.find('Dataset.csv')
        if path is None:
            raise FileNotFoundError("Dataset.csv not found in static files.")
        
        data = pd.read_csv(path)
        required_columns = ['statement', 'status']  # Updated column names to fit CSV structure
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Dataset must contain these columns: {required_columns}")
            
        data = data.dropna(subset=['statement', 'status'])
        data = data.drop_duplicates()
        data['statement'] = data['statement'].str.lower()
        data['status'] = data['status'].str.lower()
        
        logging.info(f"Loaded response data: {data.shape[0]} rows, {data.shape[1]} columns")
        logging.info("\nFirst few rows of the dataset:")
        logging.info(data.head())
        
        return data
    
    except Exception as e:
        logging.error(f"Error loading response data: {str(e)}")
        raise

def vectorize_contexts(data: pd.DataFrame) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    vectorizer = TfidfVectorizer(max_features=1000)
    context_vectors = vectorizer.fit_transform(data['statement'])  # Update to use 'statement'
    return vectorizer, context_vectors

def analyze_emotion(user_message: str) -> str:
    message = user_message.lower()
    
    anger_keywords = [
        'angry', 'mad', 'furious', 'rage', 'hate', 'frustrated', 'annoyed',
        'irritated', 'outraged', 'hostile', 'bitter', 'enraged', 'irate',
        'livid', 'infuriated', 'agitated', 'resentful'
    ]
    
    sadness_keywords = [
        'sad', 'depressed', 'worthless', 'hopeless', 'miserable', 'lonely',
        'hurt', 'despair', 'grief', 'heartbroken', 'gloomy', 'disappointed',
        'unhappy', 'devastated', 'down', 'blue', 'melancholy', 'helpless'
    ]
    
    anxiety_keywords = [
        'anxious', 'worried', 'scared', 'afraid', 'nervous', 'panic', 'stress',
        'tense', 'uneasy', 'restless', 'frightened', 'fearful', 'terrified',
        'apprehensive', 'concerned', 'overwhelmed', 'distressed'
    ]
    
    if any(f" {word} " in f" {message} " for word in anger_keywords):
        return "angry"
    elif any(f" {word} " in f" {message} " for word in sadness_keywords):
        return "sad"
    elif any(f" {word} " in f" {message} " for word in anxiety_keywords):
        return "anxious"
    
    return "neutral"

def find_response(user_message: str, responses: pd.DataFrame, vectorizer: TfidfVectorizer, context_vectors, emotion: str = None) -> str:
    try:
        user_message_vectorized = vectorizer.transform([user_message.lower()])
        similarity_scores = (context_vectors @ user_message_vectorized.T).toarray()
        best_match_idx = similarity_scores.argmax()
        
        if similarity_scores[best_match_idx][0] < 0.1:
            if emotion == "angry":
                return "I understand you're feeling angry. Would you like to tell me more about what's causing these feelings?"
            elif emotion == "sad":
                return "I hear that you're feeling down. I'm here to listen. Would you like to share what's troubling you?"
            elif emotion == "anxious":
                return "It sounds like you're dealing with anxiety. Remember to take deep breaths. Would you like to talk about what's making you feel this way?"
            else:
                return "I'm here to listen and support you. Could you tell me more about what's on your mind?"
        
        return responses.iloc[best_match_idx]['status']  # Update to return 'status'
    
    except Exception as e:
        logging.error(f"Error finding response: {str(e)}")
        return "I'm here to listen and support you. Could you tell me more about what you're experiencing?"

def process_user_input(user_message: str, responses: pd.DataFrame, vectorizer: TfidfVectorizer, context_vectors) -> str:
    try:
        emotion = analyze_emotion(user_message)
        response = find_response(
            user_message=user_message,
            responses=responses,
            vectorizer=vectorizer,
            context_vectors=context_vectors,
            emotion=emotion
        )
        return response
    
    except Exception as e:
        logging.error(f"Error processing user input: {str(e)}")
        return "I apologize, but I'm having trouble processing your message. Could you try expressing that in a different way?"

# Load data and vectorize statements on initialization
data = load_response_data()
vectorizer, context_vectors = vectorize_contexts(data)
