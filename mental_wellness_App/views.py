from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from .ai_module.emotion_analysis import load_response_data, find_response, analyze_emotion
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

class EmotionResponseView(View):
    response_data = load_response_data()
    vectorizer = TfidfVectorizer(max_features=1000)
    context_vectors = vectorizer.fit_transform(response_data['statement'])  # Update column name to 'statement'
    
    def get(self, request, *args, **kwargs):
        try:
            user_message = request.GET.get('message', '')
            detected_emotion = analyze_emotion(user_message)
            
            matching_response = find_response(
                user_message=user_message,
                responses=self.response_data,
                vectorizer=self.vectorizer,
                context_vectors=self.context_vectors,
                emotion=detected_emotion
            )
            
            return JsonResponse({
                'response': matching_response,
                'emotion': detected_emotion
            })
            
        except Exception as e:
            logging.error(f"Error in EmotionResponseView: {str(e)}")
            return JsonResponse({
                'response': "I'm here to support you. Could you please share more about what you're feeling?",
                'emotion': 'neutral'
            })
