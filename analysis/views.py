from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import DiabetesPredictionSerializer
from .prediction import predict_diabetes,get_feature_importance
import os
from django.conf import settings
import joblib
from datetime import datetime
# Create your views here.

class DiabetesPredictionView(APIView):
    """
    Prediction
    """
    def post(self, request):
        serializer = DiabetesPredictionSerializer(data=request.data)
        if serializer.is_valid():
            features = list(serializer.data.values())
            result = predict_diabetes(features)
            return Response(result)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FeatureImportanceView(APIView):
    """
    Returns feature importance
    """
    def get(self, request):
        importance = get_feature_importance()
        return Response(importance)

class HealthCheckView(APIView):
    """
    Basic API health status check endpoint
    """
    def get(self, request):
        try:
            model_path = os.path.join(settings.BASE_DIR, 'analysis', 'model', 'model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'analysis', 'model', 'scaler.pkl')
            
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            model_loaded = model is not None
            scaler_loaded = scaler is not None
            
            test_successful = False
            try:
                test_data = [2, 120, 70, 25.0, 33]
                test_result = predict_diabetes(test_data)
                test_successful = True
            except Exception as e:
                test_error = str(e)
            
            all_checks_passed = (
                model_exists and 
                scaler_exists and 
                model_loaded and 
                scaler_loaded and 
                test_successful
            )
            
            response_data = {
                "status": "healthy" if all_checks_passed else "unhealthy",
                "timestamp":datetime.now(),
                "checks": {
                    "model_file_exists": model_exists,
                    "scaler_file_exists": scaler_exists,
                    "model_loaded": model_loaded,
                    "scaler_loaded": scaler_loaded,
                    "prediction_test": test_successful
                },
                "paths": {
                    "model_path": model_path,
                    "scaler_path": scaler_path
                }
            }
            
            if not test_successful:
                response_data["test_error"] = test_error
            
            if all_checks_passed:
                return Response(response_data, status=status.HTTP_200_OK)
            else:
                return Response(response_data, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                
        except Exception as e:
            return Response({
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp":datetime.now()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)