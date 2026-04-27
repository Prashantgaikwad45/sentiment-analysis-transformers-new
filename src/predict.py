from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="model")

while True:
    text = input("Enter text: ")
    result = classifier(text)
    print(result)