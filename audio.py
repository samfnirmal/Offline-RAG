import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# 1. Define where to save it (inside your project)
local_folder = "./local_models/emotion_model"

if not os.path.exists(local_folder):
    os.makedirs(local_folder)

print(f"⬇️  Downloading Emotion Model to '{local_folder}'...")

# 2. Download and Save locally
model_id = "superb/wav2vec2-base-superb-er"

# Save the Brain (Weights)
model = AutoModelForAudioClassification.from_pretrained(model_id)
model.save_pretrained(local_folder)

# Save the Eyes/Ears (Preprocessor)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
feature_extractor.save_pretrained(local_folder)

print(f"✅ DONE! The model is saved in {local_folder}")