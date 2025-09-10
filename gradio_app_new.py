import gradio as gr
import requests
import json
import os
from typing import List, Tuple, Optional, Dict
import uuid
import hashlib
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def load_users():
    """Loads users from the JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_users(users_data):
    """Saves the users data to the JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users_data, f, indent=4)

def hash_password(password: str) -> str:
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verifies the provided password against the stored hash."""
    return stored_password == hash_password(provided_password)

def get_user_by_username(username: str) -> Optional[Tuple[str, Dict]]:
    """Finds a user by their username."""
    users = load_users()
    for user_id, user_data in users.items():
        if user_data.get("username") == username:
            return user_id, user_data
    return None

def register_new_user(username: str, password: str) -> Tuple[bool, str]:
    """Registers a new user."""
    if not username.strip() or not password.strip():
        return False, "Username and password cannot be empty."
        
    users = load_users()
    if get_user_by_username(username):
        return False, "Username already exists. Please choose another one."
        
    new_user_id = str(uuid.uuid4())
    hashed_password = hash_password(password)
    
    users[new_user_id] = {
        "username": username,
        "password": hashed_password,
        "stats": {
            "uploads": 0,
            "questions": 0,
            "summaries": 0,
            "quizzes_taken": 0,
            "topics_generated": 0,
            "flashcards_generated": 0
        },
        "badges": []
    }
    
    save_users(users)
    return True, f"User '{username}' registered successfully! You can now log in."

def authenticate_user(username: str, password: str) -> Optional[str]:
    """Authenticates a user and returns the user ID if successful."""
    user_info = get_user_by_username(username)
    if user_info:
        user_id, user_data = user_info
        if verify_password(user_data.get("password", ""), password):
            return user_id
    return None

def update_user_stat(user_id: str, stat_name: str, increment: int = 1):
    """Updates a specific stat for a user and saves it."""
    if not user_id or not stat_name:
        return
    
    users = load_users()
    if user_id in users:
        if "stats" not in users[user_id]:
            # In case 'stats' key is missing for some reason
            users[user_id]["stats"] = {
                "uploads": 0,
                "questions": 0,
                "summaries": 0,
                "quizzes_taken": 0,
                "topics_generated": 0,
                "flashcards_generated": 0
                }
        
        current_value = users[user_id]["stats"].get(stat_name, 0)
        users[user_id]["stats"][stat_name] = current_value + increment
        
        save_users(users)



# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# Global state to store uploaded documents and current session
session_state = {
    "uploaded_documents": [],
    "current_user_id": "",
    "current_document_id": "",
    "chat_history": []
}

def upload_and_process_document(file, user_id: str, chunk_size: int = 1000, overlap: int = 200, language: str = "auto") -> Tuple[bool, str, Dict]:
    """Upload document and return success status with document info."""
    if not file:
        return False, "No file uploaded", {}
    
    if not user_id.strip():
        return False, "User ID is required", {}
    
    try:
        # Prepare the file for upload
        files = {"file": (file.name, open(file.name, "rb"), "application/octet-stream")}
        data = {
            "user_id": user_id.strip(),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "language": language  # Add language parameter
        }
        
        # Send request to backend
        response = requests.post(f"{BACKEND_URL}/upload-document", files=files, data=data)
        files["file"][1].close()  # Close the file
        
        if response.status_code == 200:
            result = response.json()
            
            # Store document info in session
            doc_info = {
                "document_id": result['document_id'],
                "file_name": os.path.basename(file.name),
                "user_id": user_id.strip(),
                "upload_status": "✅ Successfully processed"
            }
            
            session_state["uploaded_documents"].append(doc_info)
            session_state["current_user_id"] = user_id.strip()
            session_state["current_document_id"] = result['document_id']
            
            update_user_stat(user_id.strip(), "uploads")
            
            return True, f"✅ Document processed successfully!", doc_info
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return False, f"Error: {error_detail}", {}
            
    except requests.exceptions.ConnectionError:
        return False, "Error: Cannot connect to backend server. Make sure the FastAPI server is running.", {}
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def process_youtube_link(url: str, user_id: str, chunk_size: int, overlap: int, language: str) -> Tuple[bool, str, Dict]:
    """Process YouTube link and return success status with document info."""
    if not url.strip():
        return False, "No URL provided", {}
    
    if not user_id.strip():
        return False, "User ID is required", {}
    
    try:
        data = {
            "url": url,
            "user_id": user_id,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "language": language
        }
        
        # Send request to backend
        response = requests.post(f"{BACKEND_URL}/process-url", data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Store document info in session
            doc_info = {
                "document_id": result['document_id'],
                "file_name": os.path.basename(result['filename']),
                "user_id": user_id.strip(),
                "upload_status": "✅ Successfully processed"
            }
            
            session_state["uploaded_documents"].append(doc_info)
            session_state["current_user_id"] = user_id.strip()
            session_state["current_document_id"] = result['document_id']
            
            update_user_stat(user_id.strip(), "uploads") # Count as an upload
            
            return True, f"✅ URL processed successfully!", doc_info
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return False, f"Error: {error_detail}", {}
            
    except requests.exceptions.ConnectionError:
        return False, "Error: Cannot connect to backend server.", {}
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def get_user_documents(user_id: str) -> List[Dict]:
    """Get all documents for a user."""
    if not user_id:
        return []
    return [doc for doc in session_state["uploaded_documents"] if doc["user_id"] == user_id]

def format_document_list(documents: List[Dict]) -> str:
    """Format document list for display."""
    if not documents:
        return "No documents uploaded yet."
    
    formatted = "## Your Documents\n\n"
    for i, doc in enumerate(documents, 1):
        formatted += f"""
        **{i}. {doc['file_name']}**
        - Document ID: `{doc['document_id'][:20]}...`
        - Status: {doc['upload_status']}

        ---
        """
    return formatted

def send_message(message: str, history: List[List[str]], page_mode: str = "all", page_numbers_input: str = ""):
    """Handles sending a message to the chat backend and streaming the response with page filtering."""
    if not message.strip():
        yield history, ""
        return
    
    history.append([message, ""])
    yield history, gr.update(value="") # Immediately clear input

    chat_user_id = session_state.get("current_user_id")
    chat_doc_id = session_state.get("current_document_id")
    current_chat_history = session_state.get("chat_history", []) # Get history from session state

    if not chat_user_id or not chat_doc_id:
        history[-1][1] = "Error: User ID or Document ID not found. Please select a document."
        yield history, ""
        return

    update_user_stat(chat_user_id, "questions")

    # NEW: Handle page filtering
    page_numbers = None
    if page_mode == "specific" and page_numbers_input.strip():
        # Get document info to check if it supports page filtering
        doc_info = get_document_info(chat_doc_id)
        if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
            page_numbers = parse_page_numbers(page_numbers_input)
            if not page_numbers:
                history[-1][1] = "Error: Invalid page numbers format. Use format like '1,3,5-7'."
                yield history, ""
                return

    chat_data = {
        "message": message.strip(),
        "user_id": chat_user_id,
        "document_id": chat_doc_id,
        "chat_history": current_chat_history, # Send chat history
        "page_numbers": page_numbers  # NEW: Include page filtering
    }
    
    full_response = ""
    try:
        with requests.post(f"{BACKEND_URL}/chat", json=chat_data, stream=True) as response:
            if response.status_code == 200:
                for token_chunk in response.iter_content(chunk_size=128):
                    if token_chunk:
                        decoded_chunk = token_chunk.decode("utf-8")
                        full_response += decoded_chunk
                        history[-1][1] = full_response
                        yield history, ""
                
                # After getting the full response, update the session state history
                current_chat_history.append([message, full_response])
                session_state["chat_history"] = current_chat_history

            else:
                error_text = response.text
                try:
                    error_json = response.json()
                    error_text = error_json.get("detail", error_text)
                except json.JSONDecodeError:
                    pass
                history[-1][1] = f"Error from backend: {error_text}"
                yield history, ""
                
    except requests.exceptions.ConnectionError:
        history[-1][1] = "Error: Cannot connect to the backend server."
        yield history, ""
    except Exception as e:
        history[-1][1] = f"An unexpected error occurred: {str(e)}"
        yield history, ""

def get_backend_status() -> str:
    """Check if the backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            return "✅ Backend server is running"
        else:
            return "Backend server is not responding correctly"
    except requests.exceptions.ConnectionError:
        return "Backend server is not running"
    except Exception as e:
        return f"Error checking backend: {str(e)}"

def format_document_info(document_id: str) -> str:
    """Format document information for display."""
    if not document_id:
        return "**No document selected**"
    
    # Find document info
    for doc in session_state["uploaded_documents"]:
        if doc["document_id"] == document_id:
            return f"""**{doc['file_name']}**
            Document ID: `{doc['document_id'][:20]}...`
            User: {doc['user_id']}
            Status: {doc.get('upload_status', '✅ Processed')}"""
    
    return "**Document not found**"

def supports_page_filtering(file_name: str) -> bool:
    """Check if a file supports page-based filtering (PDF, DOCX, TXT)."""
    if not file_name:
        return False
    file_extension = os.path.splitext(file_name)[1].lower()
    return file_extension in ['.pdf', '.docx', '.txt']

def parse_page_numbers(page_input: str) -> List[int]:
    """Parse page numbers from user input (e.g., '1,3,5-7' -> [1,3,5,6,7])."""
    if not page_input.strip():
        return []
    
    pages = []
    try:
        for part in page_input.split(','):
            part = part.strip()
            if '-' in part:
                # Handle ranges like "5-7"
                start, end = part.split('-')
                pages.extend(list(range(int(start.strip()), int(end.strip()) + 1)))
            else:
                # Handle single pages
                pages.append(int(part))
        
        # Remove duplicates and sort
        return sorted(list(set(pages)))
    except ValueError:
        return []

def get_document_info(document_id: str) -> Optional[Dict]:
    """Get document information by document_id."""
    for doc in session_state["uploaded_documents"]:
        if doc["document_id"] == document_id:
            return doc
    return None


def create_login_interface():
    """Creates the Gradio interface for login and registration."""
    with gr.Column(scale=1, min_width=400, elem_classes=["login-container"]) as login_col:
        gr.Markdown("# Welcome to AI Tutor", elem_classes=["login-title"])
        gr.Markdown("### Your personalized learning companion.", elem_classes=["login-subtitle"])
        
        with gr.Tabs(elem_classes=["login-tabs"]):
            with gr.TabItem("Login", elem_classes=["login-tab-item"]):
                login_username = gr.Textbox(label="Username", placeholder="Enter your username", elem_classes=["login-input"])
                login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password", elem_classes=["login-input"])
                login_button = gr.Button("Login", variant="primary", elem_classes=["login-button"])
                login_status = gr.Markdown()

            with gr.TabItem("Register", elem_classes=["register-tab-item"]):
                reg_username = gr.Textbox(label="Username", placeholder="Choose a username", elem_classes=["register-input"])
                reg_password = gr.Textbox(label="Password", type="password", placeholder="Choose a password", elem_classes=["register-input"])
                reg_confirm_password = gr.Textbox(label="Confirm Password", type="password", placeholder="Confirm your password", elem_classes=["register-input"])
                register_button = gr.Button("Register", variant="primary", elem_classes=["register-button"])
                registration_status = gr.Markdown()
    
    return (
        login_username, login_password, login_button, login_status,
        reg_username, reg_password, reg_confirm_password, register_button, registration_status
    )

def create_upload_interface():
    """Create the initial upload and generation interface."""
    with gr.Column(elem_classes=["upload-main-container"]):
        gr.Markdown("""
        # 🎓 AI Tutor - Get Started
        
        Begin by uploading a document/audio/video or generating a new learning module from a topic.
        """, elem_classes=["upload-intro-text"])
        
        # Backend status
        status_display = gr.Textbox(
            label="🔗 Backend Status",
            value=get_backend_status(),
            interactive=False,
            elem_classes=["status-box", "backend-status-display"]
        )
        
        refresh_btn = gr.Button("Refresh Status", size="sm", elem_classes=["refresh-status-btn"])
        
        gr.Markdown("---", elem_classes=["section-divider"])
        
        with gr.Tabs(elem_classes=["upload-tabs"]):
            with gr.TabItem("⬆️ Upload Document/Audio/Video", elem_classes=["upload-file-tab"]):
                with gr.Column(elem_classes=["upload-file-column"]):
                    gr.Markdown("### Upload your files for analysis.", elem_classes=["tab-description"])
                    file_input = gr.File(
                        label="Select Document, Audio, or Video File",
                        file_types=[
                            ".pdf", ".docx", ".txt",
                            ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma",
                            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", 
                            ".mpeg", ".mpg", ".3gp", ".3g2", ".f4v", ".asf", ".rm", ".rmvb"
                        ],
                        file_count="single",
                        height=150,
                        elem_classes=["file-upload-input"]
                    )
                    
                    # Language selection for audio files
                    language_dropdown = gr.Dropdown(
                        label="🌍 Language (for audio files)",
                        choices=[
                            ("Auto-detect", "auto"),
                            ("English", "en"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Italian", "it"),
                            ("Portuguese", "pt"),
                            ("Russian", "ru"),
                            ("Japanese", "ja"),
                            ("Korean", "ko"),
                            ("Chinese", "zh"),
                            ("Arabic", "ar"),
                            ("Hindi", "hi"),
                            ("Dutch", "nl"),
                            ("Turkish", "tr"),
                            ("Polish", "pl")
                        ],
                        value="auto",
                        info="Language selection is used only for audio files. Text documents ignore this setting.",
                        elem_classes=["language-dropdown"]
                    )
                    
                    with gr.Row(elem_classes=["chunk-settings-row"]):
                        chunk_size_input = gr.Slider(
                            label="📏 Chunk Size", minimum=500, maximum=2000, value=1000, step=100,
                            elem_classes=["chunk-size-slider"]
                        )
                        overlap_input = gr.Slider(
                            label="Overlap", minimum=50, maximum=500, value=200, step=50,
                            elem_classes=["overlap-slider"]
                        )
                    
                    upload_btn = gr.Button("Upload & Process Document/Audio/Video", variant="primary", size="lg", elem_classes=["upload-process-btn"])
            
            with gr.TabItem("🔗 Process from URL", elem_classes=["process-url-tab"]):
                 with gr.Column(elem_classes=["process-url-column"]):
                    gr.Markdown("### Process content directly from a web URL.", elem_classes=["tab-description"])
                    url_input = gr.Textbox(
                        label="Enter URL (YouTube, Vimeo, SoundCloud, etc.)",
                        placeholder="e.g., https://www.youtube.com/watch?v=...",
                        lines=2,
                        elem_classes=["url-input"]
                    )
                    
                    # Re-use language dropdown
                    url_language_dropdown = gr.Dropdown(
                        label="🌍 Language (for audio/video content)",
                        choices=[
                            ("Auto-detect", "auto"), 
                            ("English", "en"), 
                            ("Spanish", "es"), 
                            ("French", "fr"), 
                            ("German", "de"), 
                            ("Italian", "it"), 
                            ("Portuguese", "pt"), 
                            ("Russian", "ru"), 
                            ("Japanese", "ja"), 
                            ("Korean", "ko"), 
                            ("Chinese", "zh"), 
                            ("Arabic", "ar"), 
                            ("Hindi", "hi"), 
                            ("Dutch", "nl"), 
                            ("Turkish", "tr"), 
                            ("Polish", "pl"),
                            ("Finnish", "fi"),
                            ("Swedish", "sv"),
                            ("Danish", "da"),
                            ("Norwegian", "no"),
                            ("Czech", "cs"),
                            ("Hungarian", "hu"),
                            ("Romanian", "ro"),
                            ("Bulgarian", "bg"),
                            ("Croatian", "hr"),
                            ("Slovak", "sk"),
                            ("Slovenian", "sl"),
                            ("Estonian", "et"),
                            ("Latvian", "lv"),
                            ("Lithuanian", "lt"),
                            ("Ukrainian", "uk"),
                            ("Belarusian", "be"),
                            ("Macedonian", "mk"),
                            ("Maltese", "mt"),
                            ("Irish", "ga"),
                            ("Welsh", "cy"),
                            ("Icelandic", "is"),
                            ("Basque", "eu"),
                            ("Catalan", "ca"),
                            ("Galician", "gl"),
                            ("Hebrew", "he"),
                            ("Persian", "fa"),
                            ("Urdu", "ur"),
                            ("Bengali", "bn"),
                            ("Tamil", "ta"),
                            ("Telugu", "te"),
                            ("Malayalam", "ml"),
                            ("Kannada", "kn"),
                            ("Gujarati", "gu"),
                            ("Punjabi", "pa"),
                            ("Thai", "th"),
                            ("Vietnamese", "vi"),
                            ("Indonesian", "id"),
                            ("Malay", "ms"),
                            ("Filipino", "tl")
                        ],
                        value="auto",
                        elem_classes=["url-language-dropdown"]
                    )
                    
                    process_url_btn = gr.Button("🔗 Download & Process URL", variant="primary", size="lg", elem_classes=["process-url-btn"])

            with gr.TabItem("💡 Generate from Topic", elem_classes=["generate-topic-tab"]):
                 with gr.Column(elem_classes=["generate-topic-column"]):
                    gr.Markdown("### Generate a new learning module on a specific topic.", elem_classes=["tab-description"])
                    topic_input = gr.Textbox(
                        label="Enter Topic",
                        placeholder="e.g., 'Introduction to Quantum Computing'",
                        lines=2,
                        elem_classes=["topic-input"]
                    )
                    generate_topic_btn = gr.Button("🧠 Generate & Process", variant="primary", size="lg", elem_classes=["generate-topic-btn"])
        
        status_result = gr.Textbox(
            label="Processing Status",
            interactive=False,
            lines=2,
            visible=False,
            elem_classes=["processing-status-output"]
        )
        
        with gr.Column(visible=False, elem_classes=["success-section"]) as success_section:
            gr.Markdown("## Success!", elem_classes=["success-title"])
            proceed_btn = gr.Button("Go to Document Chat Interface", variant="primary", size="lg", elem_classes=["proceed-to-chat-btn"])
        
        refresh_btn.click(fn=get_backend_status, outputs=status_display)
        
        def handle_upload(file, chunk_size, overlap, language):
            user_id = session_state.get("current_user_id")
            if not user_id:
                return (
                    gr.update(),
                    gr.update(value="Error: User not logged in. Please log in first.", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            # Check if file is audio/video and show appropriate message
            file_ext = os.path.splitext(file.name)[1].lower() if file else ""
            audio_video_extensions = {
                '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma',  # Audio
                '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv',  # Video
                '.mpeg', '.mpg', '.3gp', '.3g2', '.f4v', '.asf', '.rm', '.rmvb'
            }
            is_audio_video = file_ext in audio_video_extensions
            
            if is_audio_video:
                processing_msg = "Transcribing audio/video, please wait... This may take a few minutes."
            else:
                processing_msg = "Processing your document, please wait..."

            yield (
                gr.update(value="Processing...", interactive=False),
                gr.update(value=processing_msg, visible=True),
                gr.update(visible=False)
            )
            success, message, doc_info = upload_and_process_document(file, user_id, chunk_size, overlap, language)
            if success:
                yield (
                    gr.update(value="Upload & Process Document/Audio/Video", interactive=True),
                    gr.update(value=message, visible=True),
                    gr.update(visible=True)
                )
            else:
                yield (
                    gr.update(value="Upload & Process Document/Audio/Video", interactive=True),
                    gr.update(value=f"Error: {message}", visible=True),
                    gr.update(visible=False)
                )
        
        def handle_process_url(url, chunk_size, overlap, language):
            user_id = session_state.get("current_user_id")
            if not user_id:
                return (
                    gr.update(),
                    gr.update(value="Error: User not logged in.", visible=True),
                    gr.update(visible=False),
                )

            yield (
                gr.update(value="Downloading...", interactive=False),
                gr.update(value="Downloading and processing from URL, please wait...", visible=True),
                gr.update(visible=False)
            )
            success, message, doc_info = process_youtube_link(url, user_id, chunk_size, overlap, language)
            
            if success:
                yield (
                    gr.update(value="🔗 Download & Process URL", interactive=True),
                    gr.update(value=message, visible=True),
                    gr.update(visible=True)
                )
            else:
                yield (
                    gr.update(value="🔗 Download & Process URL", interactive=True),
                    gr.update(value=f"Error: {message}", visible=True),
                    gr.update(visible=False)
                )

        def handle_generate_from_topic(topic: str):
            user_id = session_state.get("current_user_id")
            if not user_id:
                yield (
                    gr.update(value="Error: User not logged in. Please log in first.", interactive=False, visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
                return

            yield (
                gr.update(value="Generating...", interactive=False),
                gr.update(value=f"Generating content for '{topic}'...", visible=True),
                gr.update(visible=False),
                gr.update(visible=False)
            )
            
            try:
                response = requests.post(f"{BACKEND_URL}/generate-from-topic", json={"topic": topic, "user_id": user_id})
                if response.status_code == 200:
                    result = response.json()
                    doc_info = {
                        "document_id": result['document_id'],
                        "file_name": f"Generated: {topic}",
                        "user_id": user_id,
                        "upload_status": "✅ Successfully generated"
                    }
                    session_state["uploaded_documents"].append(doc_info)
                    session_state["current_user_id"] = user_id
                    session_state["current_document_id"] = result['document_id']
                    
                    update_user_stat(user_id, "topics_generated")
                    
                    generated_content = result.get("generated_content", "*Preview not available.*")

                    yield (
                        gr.update(value="🧠 Generate & Process", interactive=True),
                        gr.update(value=f"✅ Success! Module for '{topic}' is ready.", visible=True),
                        gr.update(value=f"## Generated Content Preview\n\n---\n\n{generated_content}", visible=True),
                        gr.update(visible=True)
                    )
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    yield (
                        gr.update(value="🧠 Generate & Process", interactive=True),
                        gr.update(value=f"Error: {error_detail}", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            except Exception as e:
                yield (
                    gr.update(value="🧠 Generate & Process", interactive=True),
                    gr.update(value=f"An unexpected error occurred: {str(e)}", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input, chunk_size_input, overlap_input, language_dropdown],
            outputs=[upload_btn, status_result, success_section]
        )
        
        process_url_btn.click(
            fn=handle_process_url,
            inputs=[url_input, chunk_size_input, overlap_input, url_language_dropdown],
            outputs=[process_url_btn, status_result, success_section]
        )

        generate_topic_btn.click(
            fn=handle_generate_from_topic,
            inputs=[topic_input],
            outputs=[generate_topic_btn, status_result, success_section]
        )
        
    return (
        file_input, chunk_size_input, overlap_input, upload_btn,
        topic_input, generate_topic_btn,
        status_result, success_section,
        proceed_btn, refresh_btn, status_display, language_dropdown,
        url_input, url_language_dropdown, process_url_btn # Add new components
    )

def create_chat_interface():
    """Create the document library + chat interface - user-friendly design"""
    with gr.Row(elem_classes=["main-interface"]):
        # Left side - Document Library (clean and organized)
        with gr.Column(scale=2, elem_classes=["document-library-panel"]):
            gr.Markdown("## 📚 Your Documents", elem_classes=["document-library-title"])
            document_selector = gr.Radio(
                label="Select a document to chat with",
                choices=[],
                elem_classes=["doc-selector", "document-selector-radio"]
            )
            document_info = gr.Markdown("Select a document to see details.", elem_classes=["document-info-display"])
            
            # NEW: Page filtering controls (only visible for file-based documents)
            with gr.Group(visible=False) as page_filter_group:
                gr.Markdown("### 📄 Page Filter", elem_classes=["page-filter-title"])
                page_mode = gr.Radio(
                    label="Page Selection",
                    choices=[("All Pages", "all"), ("Specific Pages", "specific")],
                    value="all",
                    elem_classes=["page-mode-radio"]
                )
                page_numbers_input = gr.Textbox(
                    label="Page Numbers (e.g., 1,3,5-7)",
                    placeholder="Enter page numbers or ranges...",
                    visible=False,
                    elem_classes=["page-numbers-input"]
                )
                page_info_display = gr.Markdown("", visible=False, elem_classes=["page-info-display"])
        
        # Right side - AI Chat Interface (focused and clean)
        with gr.Column(scale=4, elem_classes=["chat-interface-panel"]):
            with gr.Tabs():
                with gr.TabItem("💬 AI Tutor Chat", elem_classes=["chat-tab-item"]):
                    chat_history = gr.Chatbot(
                        label="Conversation with AI Tutor",
                        height=600,
                        show_label=False,
                        elem_classes=["chat-container", "chat-history-chatbot"],
                    )
                    with gr.Row():
                        message_input = gr.Textbox(
                            label="Ask anything about your documents",
                            placeholder="Type your question here...",
                            scale=5,
                            show_label=False,
                            lines=2,
                            elem_classes=["message-input-textbox"]
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["send-message-btn"])
                    clear_btn = gr.Button("Clear Chat", elem_classes=["clear-chat-btn"])

                with gr.TabItem("📝 Summarizer", elem_classes=["summarizer-tab-item"]):
                    summarize_btn = gr.Button("✨ Generate Summary", variant="primary", elem_classes=["generate-summary-btn"])
                    summary_output = gr.Markdown("Summary will appear here.", elem_classes=["summary-output-display"])
                
                with gr.TabItem("🧠 Flashcards", elem_classes=["flashcards-tab-item"]):
                    with gr.Row():
                        flashcards_limit_input = gr.Number(
                            label="Number of Flashcards",
                            value=10,
                            minimum=1,
                            maximum=50,
                            step=1,
                            elem_classes=["flashcards-limit-input"]
                        )
                        flashcards_btn = gr.Button("📚 Generate Flashcards", variant="primary", elem_classes=["generate-flashcards-btn"])
                    flashcards_output = gr.HTML("Flashcards will appear here.", elem_classes=["flashcards-output-display"])
                
                with gr.TabItem("🎯 Quiz", elem_classes=["quiz-tab-item"]):
                    with gr.Row():
                        quiz_limit_input = gr.Number(
                            label="Number of Questions",
                            value=5,
                            minimum=1,
                            maximum=20,
                            step=1,
                            elem_classes=["quiz-limit-input"]
                        )
                        quiz_level_input = gr.Dropdown(
                            label="Difficulty Level",
                            choices=[
                                ("Easy", "easy"),
                                ("Medium", "medium"),
                                ("Hard", "hard")
                            ],
                            value="easy",
                            elem_classes=["quiz-level-dropdown"]
                        )
                        quiz_format_input = gr.Dropdown(
                            label="Quiz Format",
                            choices=[
                                ("Multiple Choice", "multiple_choice"),
                                ("Fill in the Blanks", "fill_in_the_blanks"),
                                ("True/False", "true_false")
                            ],
                            value="multiple_choice",
                            elem_classes=["quiz-format-dropdown"]
                        )
                    quiz_btn = gr.Button("🎯 Generate Quiz", variant="primary", elem_classes=["generate-quiz-btn"], scale=1)
                    quiz_output = gr.HTML("Quiz will appear here.", elem_classes=["quiz-output-display"])
                    quiz_answer_box = gr.Textbox(label="Your Answer", visible=False, elem_classes=["quiz-answer-input"])
                    quiz_submit_btn = gr.Button("Submit Answer", visible=False, elem_classes=["quiz-submit-btn"])
                    quiz_feedback = gr.Markdown(visible=False, elem_classes=["quiz-feedback-display"])

                with gr.TabItem("🎙️ Podcast", elem_classes=["podcast-tab-item"]):
                    with gr.Row():
                        podcast_btn = gr.Button("📝 Generate Script", variant="secondary", elem_classes=["generate-podcast-btn"], scale=1)
                        voice_podcast_btn = gr.Button("🎙️ Generate Voice Podcast", variant="primary", elem_classes=["generate-voice-podcast-btn"], scale=2)
                    
                    podcast_output = gr.Markdown("Podcast script will appear here.", elem_classes=["podcast-output-display"])
                    
                    with gr.Row(visible=False) as voice_podcast_output_row:
                        with gr.Column():
                            voice_podcast_audio = gr.Audio(label="🎵 Generated Voice Podcast", type="numpy", elem_classes=["voice-podcast-audio"])
                        with gr.Column():
                            voice_podcast_log = gr.Textbox(label="Generation Log", lines=6, interactive=False, elem_classes=["voice-podcast-log"])

    return (
        chat_history, message_input, send_btn, clear_btn,
        document_selector, document_info, summarize_btn, summary_output,
        flashcards_btn, flashcards_output, quiz_btn, quiz_output,
        quiz_answer_box, quiz_submit_btn, quiz_feedback,
        page_filter_group, page_mode, page_numbers_input, page_info_display,  # NEW: Page filtering controls
        quiz_limit_input,  # NEW: Quiz limit input
        flashcards_limit_input,  # NEW: Flashcards limit input
        quiz_level_input,  # NEW: Quiz level input
        quiz_format_input,  # NEW: Quiz format input
        podcast_btn, podcast_output,  # NEW: Podcast components
        voice_podcast_btn, voice_podcast_audio, voice_podcast_log, voice_podcast_output_row  # NEW: Voice podcast components
    )

def create_my_communities_interface():
    """Creates the UI for viewing and managing the user's communities."""
    with gr.Column():
        gr.Markdown("# My Study Groups")
        my_communities_display = gr.Markdown("You haven't joined any communities yet.")
        refresh_my_communities_btn = gr.Button("🔄 Refresh")
    return my_communities_display, refresh_my_communities_btn

def create_explore_communities_interface():
    """Creates the UI for finding and joining new communities."""
    with gr.Column():
        gr.Markdown("# Explore Public Communities")
        with gr.Row():
            search_bar = gr.Textbox(placeholder="Search for communities by name...", scale=4)
            search_btn = gr.Button("Search", scale=1)
        
        explore_communities_display = gr.HTML("Loading public communities...")
        
        # Hidden components for join logic
        join_community_id_input = gr.Textbox(label="community_id_for_join", visible=False, elem_id="join_community_id_input_js")
        join_community_trigger_btn = gr.Button("Join Trigger", visible=False, elem_id="join_community_trigger_btn_js")

        gr.Markdown("---")
        gr.Markdown("## Create a New Community")
        with gr.Row():
            new_community_name = gr.Textbox(label="Community Name")
            new_community_desc = gr.Textbox(label="Description")
        new_community_public = gr.Checkbox(label="Public (anyone can join)", value=True)
        create_community_btn = gr.Button("Create Community", variant="primary")
        create_community_status = gr.Markdown()

    return (
        search_bar, search_btn, explore_communities_display,
        new_community_name, new_community_desc, new_community_public,
        create_community_btn, create_community_status,
        join_community_id_input, join_community_trigger_btn
    )

def create_badge_explorer_interface():
    """Creates the UI for exploring available badges."""
    with gr.Column():
        gr.Markdown("# 🏆 Badge Explorer")
        gr.Markdown("Here are all the badges you can earn. Your progress is updated as you use the app!")
        
        badge_display = gr.HTML("Loading badges...")
        
        refresh_button = gr.Button("🔄 Refresh My Badge Progress")

    return badge_display, refresh_button

def create_community_chat_interface():
    """Creates the UI for community chat."""
    with gr.Column(visible=False) as community_chat_view:
        with gr.Row():
            back_to_communities_btn = gr.Button("← Back to Communities", variant="secondary")
            community_name_display = gr.Markdown("# Community Chat")
        
        # Chat area
        community_chat_history = gr.Chatbot(
            value=[],
            label="Community Chat",
            height=400,
            container=True,
            elem_classes=["chat-container"]
        )
        
        # Message input area
        with gr.Row():
            community_message_input = gr.Textbox(
                placeholder="Type your message to the community...",
                scale=4,
                elem_classes=["chat-input"]
            )
            community_send_btn = gr.Button("Send", scale=1, variant="primary")
        
        # Hidden components for state management
        current_community_id = gr.Textbox(visible=False)
        community_refresh_btn = gr.Button("Refresh", visible=False)
        
    return (
        community_chat_view, back_to_communities_btn, community_name_display,
        community_chat_history, community_message_input, community_send_btn,
        current_community_id, community_refresh_btn
    )

def create_main_app():
    """Creates and manages the main Gradio application."""
    
    with gr.Blocks(theme=gr.themes.Soft(), css="""
        /* Global Styles */
        body {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background-color: #121212 !important; /* Deeper dark background */
            color: #E0E0E0 !important; /* Lighter text for contrast */
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .gradio-container {
            max-width: 1400px !important; /* Slightly narrower for better focus */
            margin: 20px auto !important; /* Add top/bottom margin */
            padding: 30px !important;
            background: #1E1E1E !important; /* Slightly lighter than body for depth */
            border-radius: 15px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important; /* More pronounced shadow */
            overflow: hidden; /* Ensures border-radius applies cleanly */
        }
        
        /* Main interface layout */
        .main-interface {
            gap: 40px !important; /* Increased gap for more breathing room */
            min-height: 85vh; /* Slightly taller */
            align-items: flex-start; /* Align columns to top */
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            margin-bottom: 20px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1); /* Subtle separator */
            padding-bottom: 10px;
        }

        h1 { font-size: 2.5em !important; }
        h2 { font-size: 2em !important; }
        h3 { font-size: 1.5em !important; }
        h4 { font-size: 1.2em !important; }

        /* Text and Labels */
        p, label, li {
            color: #C0C0C0 !important; /* Softer white */
            line-height: 1.6;
        }
        label { font-weight: 600 !important; margin-bottom: 8px !important; display: block; }
        
        /* Status box styling */
        .status-box {
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0; /* More vertical space */
            background: linear-gradient(145deg, #2A2A2A, #3A3A3A) !important; /* Smoother gradient */
            border-left: 5px solid #64B5F6; /* Brighter blue */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            font-weight: 600;
        }
        
        /* Document library styling */
        .document-list {
            background: linear-gradient(145deg, #252525 0%, #303030 100%) !important;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #3A3A3A;
            margin: 20px 0;
            max-height: 350px; /* Slightly taller */
            overflow-y: auto;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* Current document info styling */
        .current-doc-info {
            background: linear-gradient(145deg, #283728 0%, #3A4A3A 100%) !important; /* Greenish gradient */
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #81C784; /* Brighter green */
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            font-size: 0.95em;
        }
        
        /* Document selector styling */
        .doc-selector select {
            border: 2px solid #505050 !important; /* Darker border */
            border-radius: 10px !important;
            padding: 12px !important;
            font-size: 1em !important;
            background: #2A2A2A !important;
            color: #E0E0E0 !important;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.2) !important; /* Inner shadow */
            transition: all 0.3s ease;
        }
        .doc-selector select:hover {
            border-color: #707070 !important;
        }
        .doc-selector select:focus {
            outline: none !important;
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }
        
        /* User display styling */
        .user-display input {
            border: 1px solid #404040 !important;
            border-radius: 8px !important;
            padding: 10px 15px !important; /* More padding */
            font-size: 1em !important;
            background: #2A2A2A !important;
            color: #64B5F6 !important; /* Blue text for username */
            font-weight: 600 !important;
            text-align: center;
        }
        
        /* Chat interface styling */
        .chat-container {
            border: 1px solid #3A3A3A !important; /* Softer border */
            border-radius: 15px !important;
            background: #2A2A2A !important;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
            overflow: hidden !important;
        }
        
        /* Chat input styling */
        .chat-input textarea {
            border: 1px solid #4A4A4A !important; /* Slightly lighter border */
            border-radius: 10px !important;
            padding: 18px !important; /* More padding */
            font-size: 1.05em !important;
            resize: vertical !important; /* Allow vertical resize */
            min-height: 60px; /* Minimum height */
            font-family: 'Inter', sans-serif !important;
            background: #1E1E1E !important; /* Matches main container */
            color: #E0E0E0 !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        
        .chat-input textarea:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 4px rgba(100, 181, 246, 0.3) !important;
        }
        
        .chat-input textarea::placeholder {
            color: #909090 !important;
        }
        
        /* Send button styling */
        .send-btn button,
        button[variant="primary"],
        button[variant="secondary"] {
            border-radius: 10px !important;
            padding: 15px 25px !important;
            font-weight: 700 !important;
            font-size: 1.05em !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
            transition: all 0.2s ease-in-out !important; /* Smoother transition */
            text-transform: uppercase; /* Make button text uppercase */
            letter-spacing: 0.5px;
        }
        
        .send-btn button:hover,
        button[variant="primary"]:hover,
        button[variant="secondary"]:hover {
            transform: translateY(-2px) !important; /* More pronounced lift */
            box-shadow: 0 6px 15px rgba(0,0,0,0.4) !important;
        }
        
        button[variant="primary"] {
            background: linear-gradient(145deg, #64B5F6 0%, #2196F3 100%) !important; /* Brighter blue gradient */
            border: none !important;
            color: white !important;
        }
        
        button[variant="primary"]:hover {
            background: linear-gradient(145deg, #2196F3 0%, #1976D2 100%) !important; /* Darker blue on hover */
        }
        
        button[variant="secondary"] {
            background: linear-gradient(145deg, #8D8D8D 0%, #616161 100%) !important; /* Grey gradient */
            border: none !important;
            color: white !important;
        }
        
        button[variant="secondary"]:hover {
            background: linear-gradient(145deg, #616161 0%, #424242 100%) !important; /* Darker grey on hover */
        }
        
        /* Quick action buttons */
        .quick-actions button {
            margin: 5px !important;
            padding: 12px 18px !important; /* More padding */
            border-radius: 8px !important;
            font-size: 0.9em !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            border: 1px solid #505050 !important; /* Darker border */
            background: #2A2A2A !important;
            color: #BBBBBB !important; /* Softer text color */
        }
        
        .quick-actions button:hover {
            background: #3A3A3A !important;
            border-color: #64B5F6 !important;
            color: #64B5F6 !important; /* Blue on hover */
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        }
        
        /* Gradio specific overrides */
        .gr-form {
            background: transparent !important;
        }
        
        .gr-box {
            background: #2A2A2A !important;
            border: 1px solid #3A3A3A !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .gr-panel {
            background: #2A2A2A !important;
            border: 1px solid #3A3A3A !important;
            border-radius: 12px !important;
        }
        
        .gr-input {
            background: #1E1E1E !important; /* Darker input background */
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 10px !important;
            transition: border-color 0.3s ease;
        }
        .gr-input:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }

        /* Dropdown styling */
        .gr-dropdown-container .gr-dropdown {
            background: #1E1E1E !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            transition: border-color 0.3s ease;
        }
        .gr-dropdown-container .gr-dropdown:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }
        .gr-dropdown-container .gr-dropdown-choices {
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
        }
        .gr-dropdown-container .gr-dropdown-choice {
            color: #E0E0E0 !important;
        }
        .gr-dropdown-container .gr-dropdown-choice:hover {
            background-color: #3A3A3A !important;
        }
        
        /* File upload area */
        .file-upload {
            background: #2A2A2A !important;
            border: 2px dashed #4A4A4A !important;
            border-radius: 12px !important;
            color: #BBBBBB !important;
            padding: 30px !important;
            transition: border-color 0.3s ease;
        }
        .file-upload:hover {
            border-color: #64B5F6 !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1E1E1E; /* Matches container background */
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #4A4A4A;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #616161;
        }
        
        /* Chat bubbles */
        .message {
            background: #3A3A3A !important; /* Darker grey for readability */
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 12px !important;
            padding: 12px 18px !important;
            line-height: 1.5;
        }
        
        .message.user {
            background: linear-gradient(145deg, #2196F3 0%, #1976D2 100%) !important; /* Blue gradient for user */
            color: white !important;
            border: none !important;
            border-bottom-right-radius: 4px !important; /* Pointed corner */
        }
        
        .message.bot {
            background: linear-gradient(145deg, #3A3A3A 0%, #2A2A2A 100%) !important; /* Dark grey gradient for bot */
            color: #E0E0E0 !important;
            border: none !important;
            border-bottom-left-radius: 4px !important; /* Pointed corner */
        }
        
        /* Tabs styling */
        .gradio-tabs {
            background: transparent !important;
        }
        .gr-tab-nav {
            background: #2A2A2A !important;
            border-bottom: 1px solid #3A3A3A !important;
            border-radius: 12px 12px 0 0 !important;
            overflow: hidden;
        }
        
        .gr-tab-nav button {
            background: transparent !important;
            color: #BBBBBB !important;
            border: none !important;
            padding: 15px 25px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }
        
        .gr-tab-nav button.selected {
            background: #2196F3 !important; /* Blue for selected tab */
            color: white !important;
            border-radius: 0 !important; /* Remove individual button radius */
        }
        .gr-tab-nav button:hover {
            background-color: #3A3A3A !important;
            color: #E0E0E0 !important;
        }
        .gradio-tabs > div:first-child { /* Targets the tab content container */
            border: 1px solid #3A3A3A !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
            padding: 25px !important;
            background: #2A2A2A !important;
        }
        
        /* Responsive design */
        @media (max-width: 1024px) {
            .gradio-container {
                max-width: 98% !important;
                margin: 10px auto !important;
                padding: 20px !important;
            }
            .main-interface {
                flex-direction: column !important;
                gap: 25px !important;
            }
            .gr-tab-nav button {
                padding: 10px 15px !important;
                font-size: 0.9em !important;
            }
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 2em !important; }
            h2 { font-size: 1.7em !important; }
            .gradio-container {
                padding: 15px !important;
            }
            .send-btn button,
            button[variant="primary"],
            button[variant="secondary"] {
                font-size: 0.9em !important;
                padding: 12px 20px !important;
            }
        }
        
        /* Login/Registration Specific Styles */
        .login-container {
            padding: 30px; /* Add padding to the container */
            background: linear-gradient(145deg, #2A2A2A, #1E1E1E); /* Subtle gradient */
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
            margin: 50px auto; /* Center the login box */
            max-width: 500px; /* Max width for the login box */
            border: 1px solid #3A3A3A;
        }

        .login-title {
            text-align: center;
            margin-bottom: 5px !important;
            font-size: 2.2em !important; /* Larger title */
            color: #64B5F6 !important; /* Blue color for main title */
            border-bottom: none !important; /* Remove border from title */
            padding-bottom: 0 !important;
        }

        .login-subtitle {
            text-align: center;
            margin-bottom: 30px !important;
            color: #C0C0C0 !important;
            font-size: 1.1em !important;
            font-weight: 400 !important;
        }

        .login-tabs .gr-tab-nav {
            border-radius: 10px 10px 0 0 !important;
            background: #1E1E1E !important;
        }
        .login-tabs .gr-tab-nav button {
            font-size: 1.1em !important;
            padding: 12px 20px !important;
            font-weight: 700 !important;
            color: #BBBBBB !important;
        }
        .login-tabs .gr-tab-nav button.selected {
            background: #2196F3 !important;
            color: white !important;
        }
        .login-tabs > div:first-child { /* Targets the tab content container */
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            padding: 25px !important;
            background: #2A2A2A !important;
            border: 1px solid #3A3A3A !important;
        }

        .login-input input, .register-input input {
            padding: 15px !important;
            font-size: 1.05em !important;
            border-radius: 8px !important;
            background: #1E1E1E !important;
            border: 1px solid #4A4A4A !important;
            color: #E0E0E0 !important;
        }
        .login-input input:focus, .register-input input:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }

        .login-button button, .register-button button {
            width: 100%;
            margin-top: 20px !important;
            padding: 15px !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }
        
        /* Upload/Generate Specific Styles */
        .upload-main-container {
            padding: 20px;
            background: #1E1E1E; /* Matches main container */
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid #3A3A3A;
        }

        .upload-intro-text h1 {
            text-align: center;
            color: #64B5F6 !important;
        }
        .upload-intro-text p {
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 30px;
        }

        .backend-status-display {
            text-align: center;
            font-weight: 700 !important;
            font-size: 1.1em !important;
        }

        .refresh-status-btn button {
            width: fit-content;
            margin: 10px auto;
            display: block;
        }

        .section-divider {
            border-top: 1px solid #3A3A3A !important;
            margin: 30px 0 !important;
        }

        .upload-tabs .gr-tab-nav {
            border-radius: 10px 10px 0 0 !important;
            background: #1E1E1E !important;
        }
        .upload-tabs .gr-tab-nav button {
            font-size: 1.1em !important;
            padding: 12px 20px !important;
            font-weight: 700 !important;
            color: #BBBBBB !important;
        }
        .upload-tabs .gr-tab-nav button.selected {
            background: #2196F3 !important;
            color: white !important;
        }
        .upload-tabs > div:first-child { /* Targets the tab content container */
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            padding: 25px !important;
            background: #2A2A2A !important;
            border: 1px solid #3A3A3A !important;
        }

        .tab-description {
            font-size: 1.05em !important;
            color: #C0C0C0 !important;
            margin-bottom: 25px !important;
            text-align: center;
        }

        .file-upload-input .file-input-label,
        .language-dropdown label,
        .url-input label,
        .url-language-dropdown label,
        .topic-input label {
            font-size: 1.0em !important;
            color: #E0E0E0 !important;
            margin-bottom: 10px !important;
        }

        .chunk-settings-row {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .chunk-size-slider label, .overlap-slider label {
            text-align: center;
            margin-bottom: 15px !important;
        }
        .chunk-size-slider .gr-slider,
        .overlap-slider .gr-slider {
            background: #1E1E1E !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
        }
        .chunk-size-slider .gr-slider-track,
        .overlap-slider .gr-slider-track {
            background: #2196F3 !important;
        }

        .upload-process-btn button,
        .process-url-btn button,
        .generate-topic-btn button {
            width: 100%;
            margin-top: 20px !important;
            padding: 15px !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }

        .processing-status-output {
            margin-top: 20px !important;
            border-radius: 10px !important;
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            color: #E0E0E0 !important;
            font-weight: 600;
            text-align: center;
        }

        .success-section {
            text-align: center;
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(145deg, #283728 0%, #3A4A3A 100%) !important;
            border-left: 5px solid #81C784;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .success-title {
            color: #81C784 !important;
            margin-bottom: 15px !important;
            border-bottom: none !important;
            padding-bottom: 0 !important;
        }
        .proceed-to-chat-btn button {
            margin-top: 20px !important;
            padding: 15px 30px !important;
            font-size: 1.1em !important;
        }
        
        /* Flashcard Styling */
        .flashcard {
            background: linear-gradient(145deg, #252525 0%, #303030 100%);
            border: 1px solid #4A4A4A;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-family: 'Inter', sans-serif;
            transition: transform 0.2s ease-in-out;
        }
        .flashcard:hover {
            transform: translateY(-3px);
        }
        .flashcard .card-header {
            font-size: 1.1em;
            font-weight: bold;
            color: #64B5F6; /* Blue header */
            margin-bottom: 15px;
            border-bottom: 1px solid #3A3A3A;
            padding-bottom: 10px;
        }
        .flashcard .card-body p {
            margin: 0 0 12px 0;
            color: #E0E0E0;
        }
        .flashcard .card-body .question, .flashcard .card-body .answer {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 8px;
            color: #E0E0E0;
            border: 1px solid #3A3A3A;
            margin-top: 8px;
        }
        .flashcard .card-body p strong {
            color: #BBBBBB;
            font-weight: 600;
            display: block; /* Ensures strong tag takes its own line */
            margin-bottom: 5px;
        }

        /* Quiz Styling */
        .quiz-question {
            background: linear-gradient(145deg, #252525 0%, #303030 100%);
            border: 1px solid #4A4A4A;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-family: 'Inter', sans-serif;
            transition: transform 0.2s ease-in-out;
        }
        .quiz-question:hover {
            transform: translateY(-3px);
        }
        .quiz-question h4 {
            color: #81C784 !important; /* Green question header */
            margin-bottom: 15px;
            border-bottom: 1px solid #3A3A3A;
            padding-bottom: 10px;
        }
        .quiz-question ul {
            list-style-type: none;
            padding-left: 0;
            color: #E0E0E0;
            margin-top: 15px;
        }
        .quiz-question ul li {
            background-color: #1E1E1E;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
            border: 1px solid #3A3A3A;
            transition: background-color 0.2s ease;
        }
        .quiz-question ul li:hover {
            background-color: #2A2A2A;
        }
        .quiz-question p strong {
            color: #BBBBBB;
        }
        .quiz-question .answer {
            color: #6eff6e; /* Bright green for correct answer */
            font-weight: 600;
            margin-top: 15px;
            display: block;
        }
        
        /* Document Library Panel */
        .document-library-panel {
            background: linear-gradient(145deg, #1E1E1E 0%, #2A2A2A 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid #3A3A3A;
            height: fit-content; /* Adjust height based on content */
        }

        .document-library-title {
            color: #81C784 !important; /* Green for document titles */
            margin-bottom: 25px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
        }

        .document-selector-radio .gr-radio-group {
            display: flex; /* Make radio items stack nicely */
            flex-direction: column;
            gap: 10px;
        }
        .document-selector-radio label.radio-label {
            background: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 8px;
            padding: 12px 15px;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .document-selector-radio label.radio-label:hover {
            background: #3A3A3A;
            border-color: #64B5F6;
        }
        .document-selector-radio input[type="radio"]:checked + label.radio-label {
            background: #2196F3 !important;
            color: white !important;
            border-color: #2196F3 !important;
        }
        .document-info-display {
            margin-top: 20px !important;
            padding: 15px;
            background: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 10px;
            font-size: 0.95em;
            color: #C0C0C0;
            word-wrap: break-word;
        }

        /* Chat Interface Panel */
        .chat-interface-panel {
            background: linear-gradient(145deg, #1E1E1E 0%, #2A2A2A 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid #3A3A3A;
        }

        .chat-history-chatbot .gr-chatbot {
            border: none !important;
            background: transparent !important;
        }
        .chat-history-chatbot .gr-message {
            margin-bottom: 15px;
        }

        .message-input-textbox textarea {
            min-height: 80px; /* Taller input for chat */
        }
        .send-message-btn button {
            padding: 18px 25px !important;
        }
        .clear-chat-btn button {
            margin-top: 15px !important;
        }

        /* Summarizer, Flashcards, Quiz, Podcast Tabs */
        .summarizer-tab-item > div,
        .flashcards-tab-item > div,
        .quiz-tab-item > div,
        .podcast-tab-item > div {
            padding: 20px;
            background: #2A2A2A;
            border-radius: 12px;
            border: 1px solid #3A3A3A;
        }

        .generate-summary-btn button,
        .generate-flashcards-btn button,
        .generate-quiz-btn button,
        .generate-podcast-btn button,
        .generate-voice-podcast-btn button {
            width: 100%;
            margin-top: 20px !important;
            padding: 15px !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }

        .summary-output-display,
        .flashcards-output-display,
        .quiz-output-display,
        .podcast-output-display {
            margin-top: 20px !important;
            border-radius: 10px !important;
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            color: #E0E0E0 !important;
            font-weight: 600;
            text-align: left;
            padding: 20px;
            line-height: 1.6;
        }
                   
        .quiz-answer-input input {
            margin-top: 20px !important;
            border-radius: 8px !important;
        }
        .quiz-submit-btn button {
            margin-top: 15px !important;
        }
        .quiz-feedback-display {
            margin-top: 15px !important;
            text-align: center;
            font-weight: 600;
            font-size: 1.0em;
        }
        
        /* Quiz Limit Input Styling */
        .quiz-limit-input input {
            background: #1E1E1E !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 1.05em !important;
            text-align: center;
            width: 120px !important;
        }
        .quiz-limit-input input:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }
        .quiz-limit-input label {
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
        }
        
        /* Quiz Level Dropdown Styling */
        .quiz-level-dropdown .gr-dropdown {
            background: #1E1E1E !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 1.05em !important;
            width: 140px !important;
        }
        .quiz-level-dropdown .gr-dropdown:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }
        .quiz-level-dropdown label {
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
        }
        .quiz-level-dropdown .gr-dropdown-choices {
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
        }
        .quiz-level-dropdown .gr-dropdown-choice {
            color: #E0E0E0 !important;
            padding: 10px 15px !important;
        }
        .quiz-level-dropdown .gr-dropdown-choice:hover {
            background-color: #3A3A3A !important;
        }
        
        /* Quiz Format Dropdown Styling */
        .quiz-format-dropdown .gr-dropdown {
            background: #1E1E1E !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 1.05em !important;
            width: 160px !important;
        }
        .quiz-format-dropdown .gr-dropdown:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }
        .quiz-format-dropdown label {
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
        }
        .quiz-format-dropdown .gr-dropdown-choices {
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
        }
        .quiz-format-dropdown .gr-dropdown-choice {
            color: #E0E0E0 !important;
            padding: 10px 15px !important;
        }
        .quiz-format-dropdown .gr-dropdown-choice:hover {
            background-color: #3A3A3A !important;
        }
        
        /* Flashcards Limit Input Styling */
        .flashcards-limit-input input {
            background: #1E1E1E !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 1.05em !important;
            text-align: center;
            width: 120px !important;
        }
        .flashcards-limit-input input:focus {
            border-color: #64B5F6 !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.3) !important;
        }
        .flashcards-limit-input label {
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
        }
        
        /* Flashcard Styling */
        .flashcard {
            background: linear-gradient(145deg, #252525 0%, #303030 100%);
            border: 1px solid #4A4A4A;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-family: 'Inter', sans-serif;
            transition: transform 0.2s ease-in-out;
        }
        
        /* Podcast Script Styling */
        .podcast-output-display {
            font-family: 'Inter', sans-serif !important;
        }
        .podcast-output-display .speaker-1 {
            color: #64B5F6 !important; /* Blue for Speaker 1 */
            font-weight: 700 !important;
            margin-bottom: 8px !important;
            display: block !important;
            font-size: 1.1em !important;
        }
        .podcast-output-display .speaker-2 {
            color: #81C784 !important; /* Green for Speaker 2 */
            font-weight: 700 !important;
            margin-bottom: 8px !important;
            display: block !important;
            font-size: 1.1em !important;
        }
        .podcast-output-display .speaker-dialogue {
            color: #E0E0E0 !important;
            margin-bottom: 15px !important;
            padding-left: 20px !important;
            line-height: 1.6 !important;
            font-style: italic !important;
            font-size: 1.05em !important;
        }
        
        /* Voice Podcast Components */
        .voice-podcast-audio {
            border-radius: 10px !important;
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            margin-top: 15px !important;
        }
        
        .voice-podcast-log {
            border-radius: 10px !important;
            background: #2A2A2A !important;
            border: 1px solid #4A4A4A !important;
            color: #E0E0E0 !important;
            font-family: 'Monaco', 'Menlo', monospace !important;
            font-size: 0.9em !important;
            margin-top: 15px !important;
        }
        
        .generate-voice-podcast-btn button {
            background: linear-gradient(45deg, #6B46C1, #9333EA) !important;
            border: none !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 15px rgba(107, 70, 193, 0.3) !important;
        }
        
        .generate-voice-podcast-btn button:hover {
            background: linear-gradient(45deg, #7C3AED, #A855F7) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(107, 70, 193, 0.4) !important;
        }
        
         """) as demo:
        # State management
        user_id_state = gr.State("")
        doc_id_state = gr.State("")
        quiz_state = gr.State({})
        quiz_started = gr.State(False)
        chat_history_state = gr.State([]) # Initialize in-memory chat history

        # Interfaces
        with gr.Row(visible=True) as login_view:
            (
                login_username, login_password, login_button, login_status,
                reg_username, reg_password, reg_confirm_password, 
                register_button, registration_status
            ) = create_login_interface()

        with gr.Column(visible=False) as app_view:
            with gr.Row(elem_classes=["user-header"]):
                user_display = gr.Textbox(label="Logged In As", interactive=False, elem_classes=["user-display"], scale=3)
                logout_button = gr.Button("Logout", variant="secondary", scale=1)

            with gr.Tabs() as app_tabs:
                with gr.TabItem("📚 My Content & Chat") as chat_tab:
                    chat_interface_components = create_chat_interface()
                    (
                        chat_history, message_input, send_btn, clear_btn,
                        document_selector, document_info, summarize_btn, summary_output,
                        flashcards_btn, flashcards_output, quiz_btn, quiz_output,
                        quiz_answer_box, quiz_submit_btn, quiz_feedback,
                        page_filter_group, page_mode, page_numbers_input, page_info_display,  # NEW: Page filtering controls
                        quiz_limit_input,  # NEW: Quiz limit input
                        flashcards_limit_input,  # NEW: Flashcards limit input
                        quiz_level_input,  # NEW: Quiz level input
                        quiz_format_input,  # NEW: Quiz format input
                        podcast_btn, podcast_output,  # NEW: Podcast components
                        voice_podcast_btn, voice_podcast_audio, voice_podcast_log, voice_podcast_output_row  # NEW: Voice podcast components
                    ) = chat_interface_components

                with gr.TabItem("⬆️ Upload & Generate") as upload_tab:
                    upload_interface_components = create_upload_interface()
                    (
                        file_input, chunk_size_input, overlap_input, upload_btn,
                        topic_input, generate_topic_btn,
                        status_result, success_section, proceed_btn, 
                        refresh_btn, status_display, language_dropdown,
                        url_input, url_language_dropdown, process_url_btn
                    ) = upload_interface_components
        

        # Logout Logic
        def handle_logout():
            session_state["current_user_id"] = ""
            session_state["current_document_id"] = ""
            session_state["uploaded_documents"] = []
            
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                "", 
                "", 
                gr.update(value="<p style='color:blue'>You have been logged out.</p>"), # Set login_status
                [], 
                gr.update(choices=[], value=None),
                "Please log in to continue." 
            )

        logout_button.click(
            handle_logout,
            outputs=[
                login_view, app_view, user_display, login_username, login_password, 
                login_status, chat_history, document_selector, document_info
            ]
        )
        
        def handle_registration(username, password, confirm_password):
            if not password or not confirm_password:
                return gr.update(value="<p style='color:red'>Passwords cannot be empty.</p>")
            if password != confirm_password:
                return gr.update(value="<p style='color:red'>Passwords do not match.</p>")
            
            success, message = register_new_user(username, password)
            if success:
                return gr.update(value=f"<p style='color:green'>{message}</p>")
            else:
                return gr.update(value=f"<p style='color:red'>{message}</p>")

        register_button.click(
            handle_registration,
            inputs=[reg_username, reg_password, reg_confirm_password],
            outputs=[registration_status]
        )

        def handle_login(username, password):
            if not username or not password:
                return (
                                gr.update(value="<p style='color:red'>Username and password are required.</p>"),
                                gr.update(visible=True), gr.update(visible=False), "", gr.update(choices=[]), gr.update(value="Please log in.")
                            )
                    
            user_id = authenticate_user(username, password)
            if user_id:
                session_state["current_user_id"] = user_id
                doc_choices = []
                initial_doc_id = None
                doc_info_md = "No documents found for this user. Please upload one!"

                try:
                    response = requests.get(f"{BACKEND_URL}/documents", params={"user_id": user_id})
                    response.raise_for_status()
                    user_docs = response.json()
                    session_state["uploaded_documents"] = user_docs
                    
                    if user_docs:
                        doc_choices = [(doc['file_name'], doc['document_id']) for doc in user_docs]
                        initial_doc_id = user_docs[0]['document_id']
                        session_state["current_document_id"] = initial_doc_id
                        doc_info_md = format_document_info(initial_doc_id)
                    else:
                        session_state["current_document_id"] = None
                    
                except Exception as e:
                    return (
                                    gr.update(value=f"<p style='color:red'>Login successful, but failed to fetch documents: {e}</p>"),
                                    gr.update(visible=False), gr.update(visible=True), user_id, gr.update(choices=[]), gr.update(value="Failed to load documents.")
                                )

                return (
                    gr.update(value=f"<p style='color:green'>Welcome, {username}!</p>"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    user_id,
                    gr.update(choices=doc_choices, value=initial_doc_id),
                    gr.update(value=doc_info_md),
                    username
                )
            else:
                return (
                    gr.update(value="<p style='color:red'>Invalid credentials. Please try again.</p>"),
                    gr.update(visible=True), gr.update(visible=False), "", gr.update(choices=[]), gr.update(value="Please log in."), ""
                )
        
        login_button.click(
            handle_login,
            inputs=[login_username, login_password],
            outputs=[
                login_status, login_view, app_view, user_id_state, 
                document_selector, document_info, user_display
            ]
        )

        # Handle document selection
        def on_select_document(doc_id):
            if not doc_id:
                return gr.update(value="No document selected.")
            session_state["current_document_id"] = doc_id
            doc_details = format_document_info(doc_id)
            return gr.update(value=doc_details)

        document_selector.change(
            fn=on_select_document,
            inputs=[document_selector],
            outputs=[document_info]
        )

        def go_to_chat():
            user_id = session_state.get("current_user_id")
            docs = get_user_documents(user_id)
            doc_choices = [(doc['file_name'], doc['document_id']) for doc in docs]
            
            current_doc_id = session_state.get("current_document_id")
            doc_info_md = format_document_info(current_doc_id)

            return (
                gr.update(selected=0),
                gr.update(choices=doc_choices, value=current_doc_id),
                gr.update(value=doc_info_md),
                gr.update(visible=False), 
                gr.update(visible=False)
            )

        proceed_btn.click(
            go_to_chat, 
            outputs=[app_view, document_selector, document_info, success_section]
        )

        def refresh_doc_list_on_tab_select():
            user_id = session_state.get("current_user_id")
            if not user_id:
                return gr.update(choices=[], value=None)

            docs = get_user_documents(user_id)
            doc_choices = [(doc['file_name'], doc['document_id']) for doc in docs]
            
            # Check if current selection is still valid
            current_doc_id = session_state.get("current_document_id")
            valid_doc_ids = [doc['document_id'] for doc in docs]
            if current_doc_id not in valid_doc_ids:
                current_doc_id = valid_doc_ids[0] if valid_doc_ids else None
                session_state["current_document_id"] = current_doc_id

            return gr.update(choices=doc_choices, value=current_doc_id)

        chat_tab.select(
            refresh_doc_list_on_tab_select,
            outputs=document_selector
        )

        # NEW: Page filtering event handlers
        def on_document_change(doc_id):
            """Handle document selection change - show/hide page filter controls."""
            if not doc_id:
                return (
                    gr.update(visible=False),  # page_filter_group
                    gr.update(value="all"),    # page_mode
                    gr.update(value="", visible=False),  # page_numbers_input
                    gr.update(value="", visible=False)   # page_info_display
                )
            
            doc_info = get_document_info(doc_id)
            if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
                # Show page filter controls for file-based documents
                return (
                    gr.update(visible=True),   # page_filter_group
                    gr.update(value="all"),    # page_mode
                    gr.update(value="", visible=False),  # page_numbers_input
                    gr.update(value="📄 This document supports page-level filtering", visible=True)  # page_info_display
                )
            else:
                # Hide page filter controls for audio/URL/generated documents
                return (
                    gr.update(visible=False),  # page_filter_group
                    gr.update(value="all"),    # page_mode
                    gr.update(value="", visible=False),  # page_numbers_input
                    gr.update(value="", visible=False)   # page_info_display
                )

        def on_page_mode_change(mode):
            """Handle page mode change - show/hide page numbers input."""
            if mode == "specific":
                return (
                    gr.update(visible=True),   # page_numbers_input
                    gr.update(value="ℹ️ Enter page numbers (e.g., 1,3,5-7)", visible=True)  # page_info_display
                )
            else:
                return (
                    gr.update(visible=False),  # page_numbers_input
                    gr.update(value="📄 Using all pages", visible=True)  # page_info_display
                )

        document_selector.change(
            on_document_change,
            inputs=[document_selector],
            outputs=[page_filter_group, page_mode, page_numbers_input, page_info_display]
        )

        page_mode.change(
            on_page_mode_change,
            inputs=[page_mode],
            outputs=[page_numbers_input, page_info_display]
        )

        send_btn.click(
            send_message,
            inputs=[message_input, chat_history, page_mode, page_numbers_input],  # NEW: Include page filtering inputs
            outputs=[chat_history, message_input]
        )
        clear_btn.click(lambda: ([], ""), outputs=[chat_history, message_input])
        
        clear_btn.click(lambda: ([], session_state.update(chat_history=[])), outputs=[chat_history, message_input])
        
        def handle_summarize(page_mode: str = "all", page_numbers_input: str = ""):
            user_id = session_state.get("current_user_id")
            doc_id = session_state.get("current_document_id")
            if not user_id or not doc_id:
                yield "Please select a document first."
                return

            # NEW: Handle page filtering
            page_numbers = None
            if page_mode == "specific" and page_numbers_input.strip():
                doc_info = get_document_info(doc_id)
                if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
                    page_numbers = parse_page_numbers(page_numbers_input)
                    if not page_numbers:
                        yield "Error: Invalid page numbers format. Use format like '1,3,5-7'."
                        return

            page_info = ""
            if page_numbers:
                page_info = f" (Pages: {', '.join(map(str, page_numbers))})"
            
            yield f"⏳ Generating summary{page_info}..."
            
            full_response = ""
            try:
                request_data = {
                    "document_id": doc_id, 
                    "user_id": user_id,
                    "page_numbers": page_numbers  # NEW: Include page filtering
                }
                with requests.post(
                    f"{BACKEND_URL}/summarize", 
                    json=request_data, 
                    stream=True
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=512):
                        if chunk:
                            full_response += chunk.decode('utf-8')
                            yield full_response
                
                update_user_stat(user_id, "summaries")

            except Exception as e:
                yield f"Error generating summary: {str(e)}"

        summarize_btn.click(
            handle_summarize,
            inputs=[page_mode, page_numbers_input],  # NEW: Include page filtering inputs
            outputs=summary_output
        )

        def handle_flashcards(page_mode: str = "all", page_numbers_input: str = "", flashcards_limit: int = 10):
            user_id = session_state.get("current_user_id")
            doc_id = session_state.get("current_document_id")
            if not user_id or not doc_id:
                yield "Please select a document first."
                return

            # Validate flashcards limit
            if flashcards_limit < 1 or flashcards_limit > 50:
                yield "Error: Number of flashcards must be between 1 and 50."
                return

            # NEW: Handle page filtering
            page_numbers = None
            if page_mode == "specific" and page_numbers_input.strip():
                doc_info = get_document_info(doc_id)
                if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
                    page_numbers = parse_page_numbers(page_numbers_input)
                    if not page_numbers:
                        yield "Error: Invalid page numbers format. Use format like '1,3,5-7'."
                        return

            page_info = ""
            if page_numbers:
                page_info = f" (Pages: {', '.join(map(str, page_numbers))})"
            
            yield f"⏳ Generating {flashcards_limit} flashcards{page_info}..."
            
            full_json_response = ""
            try:
                request_data = {
                    "document_id": doc_id, 
                    "user_id": user_id, 
                    "limit": flashcards_limit,  # Use user-specified limit
                    "page_numbers": page_numbers  # NEW: Include page filtering
                }
                with requests.post(
                    f"{BACKEND_URL}/flashcards", 
                    json=request_data,
                    stream=True
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=512):
                        if chunk:
                            full_json_response += chunk.decode('utf-8')
                
                update_user_stat(user_id, "flashcards_generated")

                flashcards = json.loads(full_json_response)
                html_output = f"<h2>Flashcards ({len(flashcards)})</h2>"
                for i, card in enumerate(flashcards, 1):
                    question = card.get("question", "N/A")
                    answer = card.get("answer", "N/A")
                    html_output += f"""
                    <div class="flashcard" style="border: 1px solid #555; border-radius: 12px; padding: 20px; margin-bottom: 20px; background: linear-gradient(135deg, #2f2f2f 0%, #3c3c3c 100%); box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                        <div class="card-header" style="font-size: 16px; font-weight: bold; color: #ccc; margin-bottom: 10px; border-bottom: 1px solid #555; padding-bottom: 10px;">Flashcard {i}</div>
                        <div class="card-body">
                            <p style="margin: 0;"><strong style="color: #fff;">Question:</strong> <span style="color: #e0e0e0;">{question}</span></p>
                            <p style="margin: 0;"><strong style="color: #fff;">Answer:</strong> <span style="color: #e0e0e0;">{answer}</span></p>
                        </div>
                    </div>
                    """
                yield html_output

            except json.JSONDecodeError:
                yield f"Error: Could not decode flashcards from the server. Response: {full_json_response}"
            except Exception as e:
                yield f"Error generating flashcards: {str(e)}"

        flashcards_btn.click(
            handle_flashcards,
            inputs=[page_mode, page_numbers_input, flashcards_limit_input],  # NEW: Include page filtering inputs and flashcards limit
            outputs=flashcards_output
        )

        def handle_quiz(page_mode: str = "all", page_numbers_input: str = "", quiz_limit: int = 5, quiz_level: str = "easy", quiz_format: str = "multiple_choice"):
            user_id = session_state.get("current_user_id")
            doc_id = session_state.get("current_document_id")
            if not user_id or not doc_id:
                yield "Please select a document first."
                return
            
            # Validate quiz limit
            if quiz_limit < 1 or quiz_limit > 20:
                yield "Error: Number of questions must be between 1 and 20."
                return
            
            # NEW: Handle page filtering
            page_numbers = None
            if page_mode == "specific" and page_numbers_input.strip():
                doc_info = get_document_info(doc_id)
                if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
                    page_numbers = parse_page_numbers(page_numbers_input)
                    if not page_numbers:
                        yield "Error: Invalid page numbers format. Use format like '1,3,5-7'."
                        return

            page_info = ""
            if page_numbers:
                page_info = f" (Pages: {', '.join(map(str, page_numbers))})"
            
            # Capitalize level for display
            level_display = quiz_level.capitalize()
            
            # Format the quiz format for display
            format_display_map = {
                "multiple_choice": "Multiple Choice",
                "fill_in_the_blanks": "Fill in the Blanks", 
                "true_false": "True/False"
            }
            format_display = format_display_map.get(quiz_format, quiz_format.replace("_", " ").title())
            
            yield f"⏳ Generating {quiz_limit} {level_display} {format_display} questions{page_info}..."
            
            full_json_response = ""
            try:
                request_data = {
                    "document_id": doc_id, 
                    "user_id": user_id, 
                    "limit": quiz_limit,  # Use user-specified limit
                    "page_numbers": page_numbers,
                    "level": quiz_level,
                    "format": quiz_format  # NEW: Include quiz level
                }
                with requests.post(
                    f"{BACKEND_URL}/quiz", 
                    json=request_data,
                    stream=True
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=512):
                        if chunk:
                            full_json_response += chunk.decode('utf-8')
                
                update_user_stat(user_id, "quizzes_taken")

                quiz_questions = json.loads(full_json_response)
                html_output = f"<h2>{level_display} {format_display} Quiz ({len(quiz_questions)} Questions)</h2>"
                
                for i, q in enumerate(quiz_questions, 1):
                    question = q.get("question", "N/A")
                    answer = q.get("answer", "N/A")
                    
                    # Handle different quiz formats
                    if quiz_format == "multiple_choice":
                        options = q.get("options", [])
                        html_output += f"""
                        <div class="quiz-question" style="border: 1px solid #555; border-radius: 12px; padding: 20px; margin-bottom: 20px; background: linear-gradient(135deg, #2f2f2f 0%, #3c3c3c 100%); box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                            <h4 style="color: #fff; margin-bottom: 15px;">Question {i}: {question}</h4>
                            <ul style="list-style-type: none; padding-left: 0; color: #e0e0e0;">
                                {''.join([f"<li style='margin-bottom: 8px; padding: 8px; background: #1e1e1e; border-radius: 6px; border: 1px solid #3a3a3a;'>• {opt}</li>" for opt in options])}
                            </ul>
                            <p style="margin-top: 15px;"><strong style="color: #fff;">Answer:</strong> <span style="color: #6eff6e;">{answer}</span></p>
                        </div>
                        """
                    elif quiz_format == "fill_in_the_blanks":
                        html_output += f"""
                        <div class="quiz-question" style="border: 1px solid #555; border-radius: 12px; padding: 20px; margin-bottom: 20px; background: linear-gradient(135deg, #2f2f2f 0%, #3c3c3c 100%); box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                            <h4 style="color: #fff; margin-bottom: 15px;">Question {i}: Fill in the blank</h4>
                            <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #3a3a3a; margin: 10px 0;">
                                <p style="color: #e0e0e0; font-style: italic; line-height: 1.6;">{question}</p>
                            </div>
                            <p style="margin-top: 15px;"><strong style="color: #fff;">Answer:</strong> <span style="color: #6eff6e; background: #1e3a1e; padding: 4px 8px; border-radius: 4px; border: 1px solid #4a8a4a;">{answer}</span></p>
                        </div>
                        """
                    elif quiz_format == "true_false":
                        # Determine which option should be highlighted based on the answer
                        true_style = "background: #1e3a1e; border: 1px solid #4a8a4a; color: #6eff6e;" if answer.lower() == "true" else "background: #1e1e1e; border: 1px solid #3a3a3a; color: #e0e0e0;"
                        false_style = "background: #3a1e1e; border: 1px solid #8a4a4a; color: #ff6e6e;" if answer.lower() == "false" else "background: #1e1e1e; border: 1px solid #3a3a3a; color: #e0e0e0;"
                        
                        html_output += f"""
                        <div class="quiz-question" style="border: 1px solid #555; border-radius: 12px; padding: 20px; margin-bottom: 20px; background: linear-gradient(135deg, #2f2f2f 0%, #3c3c3c 100%); box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                            <h4 style="color: #fff; margin-bottom: 15px;">Question {i}: True or False</h4>
                            <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #3a3a3a; margin: 10px 0;">
                                <p style="color: #e0e0e0; line-height: 1.6;">{question}</p>
                            </div>
                            <div style="margin: 15px 0;">
                                <div style="display: inline-block; margin-right: 20px; padding: 8px 15px; border-radius: 6px; {false_style}">
                                    <span>False</span>
                                </div>
                                <div style="display: inline-block; padding: 8px 15px; border-radius: 6px; {true_style}">
                                    <span>True</span>
                                </div>
                            </div>
                            <p style="margin-top: 15px;"><strong style="color: #fff;">Correct Answer:</strong> <span style="color: #6eff6e; background: #1e3a1e; padding: 4px 8px; border-radius: 4px; border: 1px solid #4a8a4a; font-weight: bold;">{answer}</span></p>
                        </div>
                        """
                    else:
                        # Fallback for unknown formats
                        html_output += f"""
                        <div class="quiz-question" style="border: 1px solid #555; border-radius: 12px; padding: 20px; margin-bottom: 20px; background: linear-gradient(135deg, #2f2f2f 0%, #3c3c3c 100%); box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                            <h4 style="color: #fff; margin-bottom: 15px;">Question {i}: {question}</h4>
                            <p style="margin-top: 15px;"><strong style="color: #fff;">Answer:</strong> <span style="color: #6eff6e;">{answer}</span></p>
                        </div>
                        """
                
                yield html_output
            except json.JSONDecodeError:
                yield f"Error: Could not decode quiz from the server. Response: {full_json_response}"
            except Exception as e:
                yield f"Error generating quiz: {str(e)}"

        quiz_btn.click(
            handle_quiz,
            inputs=[page_mode, page_numbers_input, quiz_limit_input, quiz_level_input, quiz_format_input],  # NEW: Include page filtering inputs, quiz limit, quiz level, and quiz format
            outputs=quiz_output
        )

        def handle_podcast(page_mode: str = "all", page_numbers_input: str = ""):
            user_id = session_state.get("current_user_id")
            doc_id = session_state.get("current_document_id")
            if not user_id or not doc_id:
                yield "Please select a document first."
                return

            # NEW: Handle page filtering
            page_numbers = None
            if page_mode == "specific" and page_numbers_input.strip():
                doc_info = get_document_info(doc_id)
                if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
                    page_numbers = parse_page_numbers(page_numbers_input)
                    if not page_numbers:
                        yield "Error: Invalid page numbers format. Use format like '1,3,5-7'."
                        return

            page_info = ""
            if page_numbers:
                page_info = f" (Pages: {', '.join(map(str, page_numbers))})"
            
            yield f"⏳ Generating podcast script{page_info}..."
            
            full_response = ""
            try:
                request_data = {
                    "document_id": doc_id, 
                    "user_id": user_id,
                    "page_numbers": page_numbers  # NEW: Include page filtering
                }
                with requests.post(
                    f"{BACKEND_URL}/podcast", 
                    json=request_data, 
                    stream=True
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=512):
                        if chunk:
                            full_response += chunk.decode('utf-8')
                            yield full_response

            except Exception as e:
                yield f"Error generating podcast: {str(e)}"

        podcast_btn.click(
            handle_podcast,
            inputs=[page_mode, page_numbers_input],  # NEW: Include page filtering inputs
            outputs=podcast_output
        )

        def handle_voice_podcast(page_mode: str = "all", page_numbers_input: str = ""):
            user_id = session_state.get("current_user_id")
            doc_id = session_state.get("current_document_id")
            if not user_id or not doc_id:
                return None, "Please select a document first.", "❌ Error: No document selected", False

            # Handle page filtering
            page_numbers = None
            if page_mode == "specific" and page_numbers_input.strip():
                doc_info = get_document_info(doc_id)
                if doc_info and supports_page_filtering(doc_info.get('file_name', '')):
                    page_numbers = parse_page_numbers(page_numbers_input)
                    if not page_numbers:
                        return None, "Error: Invalid page numbers format. Use format like '1,3,5-7'.", "❌ Error: Invalid page format", False

            page_info = ""
            if page_numbers:
                page_info = f" (Pages: {', '.join(map(str, page_numbers))})"
            
            try:
                # Update log
                initial_log = f"🎙️ Generating voice podcast{page_info}...\n⏳ This may take a few minutes..."
                
                request_data = {
                    "document_id": doc_id,
                    "user_id": user_id,
                    "page_numbers": page_numbers,
                    "speaker1": None,  # Use default speakers
                    "speaker2": None,
                    "cfg_scale": 1.3
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/voice-podcast",
                    json=request_data,
                    timeout=300  # 5 minute timeout for voice generation
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("success", False):
                    # Create audio array from file path
                    audio_file_path = result.get("audio_file_path")
                    if audio_file_path and os.path.exists(audio_file_path):
                        import soundfile as sf
                        audio_data, sample_rate = sf.read(audio_file_path)
                        audio_output = (sample_rate, audio_data)
                    else:
                        audio_output = None
                    
                    success_log = result.get("log", "✅ Voice podcast generated successfully!")
                    return audio_output, success_log, success_log, True
                else:
                    error_msg = f"❌ Voice generation failed: {result.get('log', 'Unknown error')}"
                    return None, error_msg, error_msg, False

            except requests.exceptions.Timeout:
                error_msg = "❌ Voice generation timed out. Please try again with a shorter document."
                return None, error_msg, error_msg, False
            except Exception as e:
                error_msg = f"❌ Error generating voice podcast: {str(e)}"
                return None, error_msg, error_msg, False

        def update_voice_podcast_output(audio, log, log_display, success):
            # Show/hide the voice podcast output row based on success
            return {
                voice_podcast_output_row: gr.update(visible=success),
                voice_podcast_audio: audio,
                voice_podcast_log: log_display
            }

        voice_podcast_btn.click(
            handle_voice_podcast,
            inputs=[page_mode, page_numbers_input],
            outputs=[voice_podcast_audio, voice_podcast_log, voice_podcast_log, voice_podcast_output_row]
        ).then(
            lambda audio, log, log_display, success: gr.update(visible=success),
            inputs=[voice_podcast_audio, voice_podcast_log, voice_podcast_log, voice_podcast_output_row],
            outputs=voice_podcast_output_row
        )

    return demo

if __name__ == "__main__":
    main_app = create_main_app()
    main_app.launch(debug=True,share=True) 