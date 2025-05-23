import os
import base64
import mimetypes
import logging # Added import
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Load environment variables from .env file
load_dotenv()

# 2. Create OpenAI client
# The client will automatically pick up OPENAI_API_KEY and OPENAI_API_BASE 
# from environment variables if they are set by load_dotenv().
try:
    client = OpenAI()
    # Check if the API key was loaded
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY not found or is empty. Make sure it's set in your .env file or environment variables.")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}") # Replaced print
    client = None # Ensure client is None if initialization fails

def get_image_mime_type(image_path):
    """Guesses the MIME type of an image file."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith('image'):
        return mime_type
    # Fallback for common types if mimetypes fails for some reason
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.png':
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.gif':
        return 'image/gif'
    elif ext == '.webp':
        return 'image/webp'
    return None

def image_to_data_url(image_path: str) -> str:
    """
    Reads an image file and returns its base64 encoded data URL.
    """
    mime_type = get_image_mime_type(image_path)
    if not mime_type:
        raise ValueError(f"Cannot determine a valid image MIME type for: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

def analyze_image_with_openai(image_data_url: str, question: str):
    """
    Submits an image (as data URL) and a question to OpenAI for analysis.
    """
    if not client:
        logging.warning("OpenAI client is not initialized. Cannot analyze image.") # Replaced print
        return None

    prompt_message = f"Answer user's question using the provided image. {question}"
    
    try:
        response = client.chat.completions.create(
            model="o4-mini",  # Using gpt-4o, can be changed to gpt-4-vision-preview if preferred
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
            max_completion_tokens=500, # Increased max_tokens for potentially more detailed answers
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}") # Replaced print
        return None

def main():
    """
    Main function to load image, convert to data URL, and get analysis from OpenAI.
    """
    if not client:
        logging.error("Exiting: OpenAI client failed to initialize.") # Replaced print
        return

    # Define the path to the test image
    # Assumes the script is run from the root of the GenAIDocumentUnderstanding directory
    test_image_relative_path = "data/test_images/data_table.png"
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the image
    # This assumes the 'data' folder is at the same level as the script, or in the workspace root
    # If script is in root, script_dir is workspace root.
    absolute_image_path = os.path.join(script_dir, test_image_relative_path)

    if not os.path.exists(absolute_image_path):
        logging.error(f"Error: Test image not found at {absolute_image_path}") # Replaced print
        # Try path relative to current working directory as a fallback
        # This might be useful if script is not in workspace root but CWD is.
        cwd_image_path = os.path.join(os.getcwd(), test_image_relative_path)
        if os.path.exists(cwd_image_path):
            absolute_image_path = cwd_image_path
            logging.info(f"Found image at CWD relative path: {absolute_image_path}") # Replaced print
        else:
            logging.error(f"Also tried CWD relative path: {cwd_image_path}, not found.") # Replaced print
            return
        
    logging.info(f"Using image: {absolute_image_path}") # Replaced print

    # 3. Convert image to data URL
    try:
        data_url = image_to_data_url(absolute_image_path)
        logging.info(f"Successfully converted image to data URL (first few chars): {data_url[:25]}...") # Replaced print
    except Exception as e:
        logging.error(f"Error converting image to data URL: {e}") # Replaced print
        return

    # 4. Define a question and submit to OpenAI
    user_question = "Based on the image, what is the Operating Profit for 2023'?" # Example question
    logging.info(f"Asking OpenAI the question: '{user_question}' using the image '{test_image_relative_path}'") # Replaced print
    
    analysis_result = analyze_image_with_openai(data_url, user_question)

    if analysis_result:
        logging.info("OpenAI's Analysis:") # Replaced print
        logging.info(analysis_result) # Replaced print
    else:
        logging.error("Failed to get analysis from OpenAI.") # Replaced print

if __name__ == "__main__":
    main()
