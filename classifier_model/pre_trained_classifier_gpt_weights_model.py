# Classify text with the fine-tuned GPT-2 spam classifier.
# Train the bundle first with: python -m classifier_model.train tasks/spam.yaml
# Run from the repo root: python -m classifier_model.pre_trained_classifier_gpt_weights_model

from classifier_model.predict import Classifier

INPUT_TEXT = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

classifier = Classifier.load("artifacts/spam")
print(classifier.predict([INPUT_TEXT])[0])
