from django.urls import path
from .views import DiabetesPredictionView,FeatureImportanceView,HealthCheckView


urlpatterns = [
    path('predict/',DiabetesPredictionView.as_view()),
    path('features/',FeatureImportanceView.as_view()),
    path('health/',HealthCheckView.as_view())
]