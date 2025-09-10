#!/usr/bin/env python3
"""
AI Tutor - Google Colab Runner
Optimized specifically for Google Colab environments.
Runs both FastAPI backend and Gradio frontend in a single notebook cell.
"""

import os
import threading
import time
import uvicorn
import logging
from contextlib import asynccontextmanager
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AITutorColab:
    def __init__(self):
        self.backend_thread = None
        self.backend_started = False
        
    def run_backend_server(self):
        """Run FastAPI backend in a separate thread"""
        try:
            from api import app
            
            # Configure uvicorn for Colab
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                access_log=False,  # Reduce log noise
                loop="asyncio"
            )
            
            server = uvicorn.Server(config)
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Signal that backend is starting
            self.backend_started = True
            logger.info("🔧 FastAPI backend starting on port 8000...")
            
            # Run the server
            loop.run_until_complete(server.serve())
            
        except Exception as e:
            logger.error(f"❌ Backend error: {e}")
            raise
    
    def check_backend_health(self, max_retries=20):
        """Check if backend is ready"""
        import requests
        
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Backend is ready!")
                    return True
            except Exception:
                if i < max_retries - 1:
                    logger.info(f"⏳ Waiting for backend... ({i+1}/{max_retries})")
                    time.sleep(1)
                else:
                    logger.error("❌ Backend failed to start within timeout")
                    return False
        return False
    
    def run(self):
        """Main run method for Colab"""
        logger.info("🚀 Starting AI Tutor for Google Colab...")
        
        try:
            # Start backend in separate thread
            self.backend_thread = threading.Thread(
                target=self.run_backend_server, 
                daemon=True,
                name="AITutor-Backend"
            )
            self.backend_thread.start()
            
            # Wait for backend to start
            while not self.backend_started:
                time.sleep(0.1)
            
            # Check backend health
            if not self.check_backend_health():
                raise Exception("Backend failed to start properly")
            
            # Import and start Gradio frontend
            logger.info("🎨 Starting Gradio frontend...")
            from gradio_app_new import create_main_app
            
            app = create_main_app()
            
            # Launch Gradio with Colab-optimized settings
            logger.info("🌐 Launching Gradio interface...")
            logger.info("📱 The sharing link will appear below:")
            
            # Launch with share=True for Colab access
            app.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,           # Enable public sharing for Colab
                debug=False,
                show_error=True,
                quiet=False,
                inbrowser=False,      # Don't try to open browser in Colab
                prevent_thread_lock=False  # Let Gradio handle the main thread
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise

# Convenience functions for different use cases
def run_ai_tutor():
    """Simple function to start AI Tutor in Colab"""
    runner = AITutorColab()
    runner.run()

def run_with_custom_ports(backend_port=8000, frontend_port=7860):
    """Run with custom ports if needed"""
    logger.info(f"🚀 Starting AI Tutor with custom ports: Backend={backend_port}, Frontend={frontend_port}")
    
    # Modify the ports in the environment
    os.environ['BACKEND_PORT'] = str(backend_port)
    os.environ['FRONTEND_PORT'] = str(frontend_port)
    
    runner = AITutorColab()
    runner.run()

# Special function for notebook environments
def run_in_notebook():
    """Optimized for Jupyter/Colab notebooks"""
    try:
        # Check if we're in a notebook
        get_ipython()
        logger.info("📝 Detected notebook environment")
        
        # Display startup message
        from IPython.display import display, HTML, Markdown
        
        display(Markdown("""
        # 🎓 AI Tutor Starting...
        
        **Please wait while the application initializes:**
        1. ⚙️ Starting FastAPI backend...
        2. 🎨 Starting Gradio frontend...
        3. 🌐 Generating sharing links...
        
        **Once ready, you'll see:**
        - ✅ Backend ready message
        - 📱 Gradio sharing link (click to access the app)
        """))
        
        # Start the application
        run_ai_tutor()
        
    except NameError:
        # Not in a notebook, use regular method
        logger.info("💻 Not in notebook environment, using standard startup")
        run_ai_tutor()

if __name__ == "__main__":
    # Auto-detect environment and run appropriately
    run_in_notebook()
