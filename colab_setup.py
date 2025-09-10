"""
🎓 AI Tutor - Google Colab Setup & Runner
Copy this entire code into a Google Colab cell and run it!

This script will:
1. Install required dependencies
2. Set up environment variables
3. Start both FastAPI backend and Gradio frontend
4. Provide sharing links for easy access
"""

# ============================================================================
# STEP 1: Install Dependencies (uncomment if running for the first time)
# ============================================================================

# Uncomment these lines if you haven't installed the dependencies yet:

# !pip install fastapi uvicorn gradio qdrant-client openai python-dotenv
# !pip install PyPDF2 python-docx python-multipart requests numpy
# !pip install sentence-transformers langchain-text-splitters
# !pip install google-generativeai langchain langchain-google-genai
# !pip install torch librosa soundfile transformers accelerate diffusers

# ============================================================================
# STEP 2: Environment Setup
# ============================================================================

import os
import threading
import time
import uvicorn
import logging
from IPython.display import display, HTML, Markdown

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 3: Environment Variables (EDIT THESE WITH YOUR API KEYS)
# ============================================================================

# 🔑 IMPORTANT: Set your API keys here
os.environ["QDRANT_URL"] = "your_qdrant_url_here"
os.environ["QDRANT_API_KEY"] = "your_qdrant_api_key_here"
os.environ["GEMINI_API_KEY"] = "your_gemini_api_key_here"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Optional

# Collection name (optional)
os.environ["COLLECTION_NAME"] = "Tutor_Documents"

# ============================================================================
# STEP 4: Backend Server Runner
# ============================================================================

backend_thread = None
backend_ready = False

def run_backend():
    """Run FastAPI backend server"""
    global backend_ready
    try:
        # Import your API
        from api import app
        
        # Configure uvicorn for Colab
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="warning",  # Reduce log noise
            access_log=False
        )
        
        server = uvicorn.Server(config)
        
        logger.info("🔧 Starting FastAPI backend...")
        backend_ready = True
        
        # Run server (this will block the thread)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
        
    except Exception as e:
        logger.error(f"❌ Backend error: {e}")
        raise

def check_backend():
    """Check if backend is responding"""
    import requests
    
    for i in range(30):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Backend is ready!")
                return True
        except:
            if i < 29:
                print(f"⏳ Waiting for backend... ({i+1}/30)")
                time.sleep(1)
    
    logger.error("❌ Backend failed to start")
    return False

# ============================================================================
# STEP 5: Main Application Runner
# ============================================================================

def run_ai_tutor_colab():
    """Main function to run AI Tutor in Google Colab"""
    
    # Display welcome message
    display(Markdown("""
    # 🎓 AI Tutor - Starting Up!
    
    **Please wait while we initialize the application...**
    
    📋 **Steps:**
    1. ⚙️ Starting FastAPI backend server...
    2. 🔍 Checking backend health...
    3. 🎨 Starting Gradio frontend...
    4. 🌐 Generating public sharing links...
    
    ---
    """))
    
    global backend_thread
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Wait for backend to be ready
        time.sleep(2)  # Give it a moment to start
        
        if not check_backend():
            display(HTML("""
            <div style="color: red; font-weight: bold;">
            ❌ Backend failed to start! Please check your API keys and try again.
            </div>
            """))
            return
        
        # Start Gradio frontend
        logger.info("🎨 Starting Gradio frontend...")
        
        # Import and create Gradio app
        from gradio_app_new import create_main_app
        
        app = create_main_app()
        
        # Display success message
        display(Markdown("""
        ## ✅ Backend Started Successfully!
        
        **Backend URLs:**
        - 🔗 API: `http://localhost:8000`
        - 📖 API Docs: `http://localhost:8000/docs`
        
        **🎨 Starting Gradio Frontend...**
        
        📱 **Your sharing link will appear below** ⬇️
        """))
        
        # Launch Gradio with sharing enabled
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,        # This creates the public sharing link
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=False,   # Don't try to open browser in Colab
            prevent_thread_lock=False
        )
        
    except Exception as e:
        display(HTML(f"""
        <div style="color: red; font-weight: bold;">
        ❌ Error starting AI Tutor: {str(e)}
        </div>
        """))
        logger.error(f"Error: {e}")
        raise

# ============================================================================
# STEP 6: Quick Setup Function
# ============================================================================

def setup_and_run():
    """One-click setup and run function"""
    
    # Check if API keys are set
    required_keys = ["QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key) or os.getenv(key) == f"your_{key.lower()}_here"]
    
    if missing_keys:
        display(HTML(f"""
        <div style="color: orange; font-weight: bold; padding: 10px; border: 2px solid orange; border-radius: 5px;">
        ⚠️ <strong>Missing API Keys!</strong><br>
        Please set these environment variables above:<br>
        {', '.join(missing_keys)}<br><br>
        Edit the environment variables section in this script and run again.
        </div>
        """))
        return
    
    # All good, start the application
    display(HTML("""
    <div style="color: green; font-weight: bold; padding: 10px; border: 2px solid green; border-radius: 5px;">
    🚀 <strong>All API keys found!</strong> Starting AI Tutor...
    </div>
    """))
    
    run_ai_tutor_colab()

# ============================================================================
# STEP 7: Run the Application
# ============================================================================

if __name__ == "__main__":
    setup_and_run()

# ============================================================================
# INSTRUCTIONS FOR USE:
# ============================================================================

print("""
🎓 AI TUTOR - GOOGLE COLAB INSTRUCTIONS:

1. ✏️  Edit the API keys section above (STEP 3)
2. ▶️  Run this cell 
3. 📱 Click the sharing link that appears
4. 🎉 Start using AI Tutor!

📋 Required API Keys:
   • Qdrant (Vector Database)
   • Google Gemini (AI Chat)
   • OpenAI (Optional, for better embeddings)

🔗 Sharing Link: Will appear below when ready
""")

# Run the setup automatically
setup_and_run()
