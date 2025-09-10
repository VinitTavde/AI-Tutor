#!/usr/bin/env python3
"""
AI Tutor - All-in-One Application Runner
Runs both FastAPI backend and Gradio frontend in a single script.
Perfect for Google Colab and single-process deployment.
"""

import os
import sys
import threading
import time
import uvicorn
import gradio as gr
from multiprocessing import Process
import signal
import atexit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global process references for cleanup
backend_process = None
frontend_process = None

def run_backend():
    """Run the FastAPI backend server"""
    try:
        # Import the FastAPI app
        from api import app
        
        # Run the backend server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Backend server error: {e}")
        raise

def run_frontend():
    """Run the Gradio frontend"""
    try:
        # Import the Gradio app
        from gradio_app_new import create_main_app
        
        # Create and launch the Gradio app
        app = create_main_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Enable sharing for Colab
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=False  # Don't auto-open browser in Colab
        )
    except Exception as e:
        logger.error(f"Frontend server error: {e}")
        raise

def check_backend_health():
    """Check if backend is ready"""
    import requests
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Backend is ready!")
                return True
        except:
            if i < max_retries - 1:
                logger.info(f"⏳ Waiting for backend... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                logger.error("❌ Backend failed to start")
                return False
    return False

def cleanup():
    """Cleanup function to stop all processes"""
    global backend_process, frontend_process
    
    logger.info("🧹 Cleaning up processes...")
    
    if backend_process and backend_process.is_alive():
        logger.info("Stopping backend process...")
        backend_process.terminate()
        backend_process.join(timeout=5)
        if backend_process.is_alive():
            backend_process.kill()
    
    if frontend_process and frontend_process.is_alive():
        logger.info("Stopping frontend process...")
        frontend_process.terminate()
        frontend_process.join(timeout=5)
        if frontend_process.is_alive():
            frontend_process.kill()
    
    logger.info("✅ Cleanup completed")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info("🛑 Received interrupt signal, shutting down...")
    cleanup()
    sys.exit(0)

def run_ai_tutor():
    """
    Main function to run both backend and frontend
    """
    global backend_process, frontend_process
    
    # Register cleanup functions
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting AI Tutor Application...")
    logger.info("📋 This will start both FastAPI backend and Gradio frontend")
    
    try:
        # Start backend process
        logger.info("🔧 Starting FastAPI backend on port 8000...")
        backend_process = Process(target=run_backend)
        backend_process.daemon = True
        backend_process.start()
        
        # Wait for backend to be ready
        if not check_backend_health():
            logger.error("❌ Failed to start backend. Exiting...")
            cleanup()
            return False
        
        # Start frontend process
        logger.info("🎨 Starting Gradio frontend on port 7860...")
        frontend_process = Process(target=run_frontend)
        frontend_process.daemon = True
        frontend_process.start()
        
        # Give frontend time to start
        time.sleep(3)
        
        logger.info("🎉 AI Tutor is running!")
        logger.info("🌐 Access URLs:")
        logger.info("   📊 Backend API: http://localhost:8000")
        logger.info("   📱 Gradio UI: http://localhost:7860")
        logger.info("   📖 API Docs: http://localhost:8000/docs")
        
        # In Colab, the Gradio app will show sharing links
        logger.info("📱 For Google Colab: Look for the Gradio sharing link above")
        
        # Keep the main process alive
        try:
            while True:
                # Check if processes are still alive
                if not backend_process.is_alive():
                    logger.error("❌ Backend process died unexpectedly")
                    break
                if not frontend_process.is_alive():
                    logger.error("❌ Frontend process died unexpectedly")
                    break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        
    except Exception as e:
        logger.error(f"❌ Error starting AI Tutor: {e}")
        return False
    finally:
        cleanup()
    
    return True

# Alternative threading-based approach for environments where multiprocessing doesn't work well
def run_ai_tutor_threaded():
    """
    Alternative approach using threading instead of multiprocessing.
    Better for some environments like Jupyter notebooks.
    """
    logger.info("🚀 Starting AI Tutor Application (Threading Mode)...")
    
    # Start backend in a thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to be ready
    if not check_backend_health():
        logger.error("❌ Failed to start backend. Exiting...")
        return False
    
    # Start frontend in main thread (Gradio needs to run in main thread)
    logger.info("🎨 Starting Gradio frontend...")
    run_frontend()
    
    return True

def main():
    """
    Main entry point with environment detection
    """
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
        logger.info("🔍 Detected Google Colab environment")
    except ImportError:
        in_colab = False
        logger.info("🔍 Detected local environment")
    
    # Check if running in Jupyter
    try:
        get_ipython()
        in_jupyter = True
        logger.info("🔍 Detected Jupyter environment")
    except NameError:
        in_jupyter = False
    
    # Choose appropriate method based on environment
    if in_colab or in_jupyter:
        logger.info("📝 Using threading mode for notebook environment")
        return run_ai_tutor_threaded()
    else:
        logger.info("💻 Using multiprocessing mode for local environment")
        return run_ai_tutor()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
