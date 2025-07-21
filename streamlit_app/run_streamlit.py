# import os

# def run_streamlit_app():
#     os.system("streamlit run winequality_flask/streamlit_app.py")

# if __name__ == "__main__":
#     run_streamlit_app()

import os
from pyngrok import ngrok, conf
import threading

# Set your ngrok authtoken
conf.get_default().auth_token = "2zu3fWOJYBvBYNk7kLVVBXTDo9w_61wS3Js7Tu6ibEGJqo4ps"

# Path to your Streamlit app file
STREAMLIT_APP_PATH = "streamlit_app.py"

# Function to run Streamlit app
def run_streamlit():
    os.system(f"streamlit run {STREAMLIT_APP_PATH}")

# Function to run ngrok and expose the app
def run_with_ngrok():
    thread = threading.Thread(target=run_streamlit)
    thread.start()

    public_url = ngrok.connect(8501)
    print(f"\n🔥 Your public URL is: {public_url}\n")

if __name__ == "__main__":
    run_with_ngrok()

