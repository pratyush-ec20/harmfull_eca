from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from transformers import logging as transformers_logging  
from transformers import logging as transformers_logging  

# Suppress specific warning messages from the transformers library
transformers_logging.set_verbosity_error()
import warnings
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
transformers_logging.set_verbosity_error()
input_text =input("Enter a sentence:")
# Load pre-trained model and tokenizerretweet
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformers_logging.set_verbosity_error()
def preprocess_text(text):
# Tokenize and encode the text
    transformers_logging.set_verbosity_error()
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # Convert to PyTorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    return input_ids
    

from transformers import BertForSequenceClassification

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.load_state_dict(torch.load('pytorch_model.bin', map_location=torch.device('cpu')), strict=False)

# Set the model to evaluation mode
model.eval()

    
with torch.no_grad():
    input_ids = preprocess_text(input_text)
    outputs = model(input_ids)

predictions = torch.argmax(outputs.logits, dim=1).item()
prediction=""
    
if(predictions==0):
    prediction="No harm"
    
elif(predictions==1):
    prediction="personal damage harm"

elif(predictions==2):   
    prediction="physical harm"

else:
    prediction="cyber security harm"
        
print("harm is :",prediction)