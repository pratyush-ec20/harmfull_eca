import torch
from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load pre-trained model and tokenizer once at the start
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('pytorch_model.bin', map_location=torch.device('cpu')), strict=False)
model.eval()

def preprocess_text(text):
    # Tokenize and encode the text
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # Convert to PyTorch tensor and add batch dimension
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    return input_ids

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_text = request.form['ECA RULE']

    # Preprocess the input text
    input_ids = preprocess_text(input_text)

    # Predict the class
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = torch.argmax(outputs.logits, dim=1).item()
    
    # Map predictions to class names
    class_names = ["No harm", "Personal damage harm", "Physical harm", "Cyber security harm"]
    prediction = class_names[predictions]
    
    return render_template('index.html', prediction_text='It is a {} and belongs to class {}'.format(prediction, predictions))

if __name__ == "__main__":
    app.run(debug=True)
