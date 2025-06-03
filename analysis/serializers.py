from rest_framework import serializers

class DiabetesPredictionSerializer(serializers.Serializer):
    Pregnancies = serializers.FloatField()
    Glucose = serializers.FloatField()
    BloodPressure = serializers.FloatField()
    BMI = serializers.FloatField()
    Age = serializers.FloatField()
