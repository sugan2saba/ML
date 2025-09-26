from rest_framework import serializers

class RecordSerializer(serializers.Serializer):
    # flexible serializer: accept any keys, we validate at runtime against model schema
    # You can tighten this later by listing required fields.
    # For now, allow arbitrary fields:
    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError("Each record must be an object")
        return data

class PredictRequestSerializer(serializers.Serializer):
    records = serializers.ListField(child=RecordSerializer(), allow_empty=False)

class PredictResponseItemSerializer(serializers.Serializer):
    prob = serializers.FloatField()
    pred = serializers.IntegerField()
