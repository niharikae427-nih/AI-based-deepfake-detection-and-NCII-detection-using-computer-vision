from transformers import pipeline

print("Loading NCII safety model...")

classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection"
)

print("Model loaded successfully!")
print(classifier("test.jpg"))