# Classify text with the fine-tuned GPT-2 spam classifier.
# Train the bundle first with: python -m textclf.train tasks/spam.yaml

from textclf.predict import Classifier

INPUT_TEXT = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

classifier = Classifier.load("artifacts/spam")
print(classifier.predict([INPUT_TEXT])[0])
