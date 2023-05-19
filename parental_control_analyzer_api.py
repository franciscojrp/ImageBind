import logging
from flask import Flask, request, jsonify
from functools import wraps
import tempfile
import torch
import data
from models import imagebind_model
from models.imagebind_model import ModalityType

print("Loading model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
print("Model loaded")

categories_list=["sex", "nudity", "violence", "fear", "other"]

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

@app.before_request
def log_request_info():
    logger.info('Request: %s %s', request.method, request.url)
    # logger.debug('Request Headers: %s', request.headers)
    # logger.debug('Request Body: %s', request.get_data(as_text=True))

@app.after_request
def log_response_info(response):
    logger.info('Response Status Code: %s', response.status_code)
    logger.info('Response Headers: %s', response.headers)
    logger.info('Response Body: %s', response.get_data(as_text=True))
    return response

# Token validation decorator
def require_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        # Check if the token is valid
        if token != 'CHANGE_ME':
            return jsonify({'error': 'Invalid token.'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/api/v1/analyze', methods=['POST'])
@require_token
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request.'}), 400
    
    image = request.files['image']
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp_file.name)
    
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(categories_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data([temp_file.name], device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

        scores = (
            torch.softmax(
                embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1
            )
            .squeeze(0)
            .tolist()
        )

        score_dict = {label: '{:.2f}'.format(score) for label, score in zip(categories_list, scores)}
        
        # Clean up the temporary file
        temp_file.close()
               
        return jsonify(score_dict), 200

if __name__ == '__main__':
    app.run()
