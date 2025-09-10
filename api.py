import os
import tempfile
import uuid
import uvicorn
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from fastapi.responses import StreamingResponse, FileResponse
import json

from Qdrant import QdrantManager
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from chatbot_service import RAGChatbot
from media_processor import MediaProcessor
from vibevoice_service import VibeVoiceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Upload and Embedding API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
qdrant_manager = QdrantManager()
document_processor = DocumentProcessor()
vibevoice_service = VibeVoiceService()


# Try to use OpenAI first, fall back to local model if API key not available
try:
    embedding_service = EmbeddingService(use_openai=True)
    logger.info("Using OpenAI embeddings")
except ValueError:
    logger.warning("OpenAI API key not found, using local embeddings")
    embedding_service = EmbeddingService(use_openai=False)

# Collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Tutor-Documents")

# Initialize RAG Chatbot
try:
    chatbot = RAGChatbot()
    logger.info("RAG Chatbot initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize RAG Chatbot: {e}")
    chatbot = None

# Initialize VibeVoice Service
try:
    
    if vibevoice_service.is_available():
        logger.info("VibeVoice service initialized successfully")
    else:
        raise Exception("VibeVoice service not available")
except Exception as e:
    logger.warning(f"Failed to initialize VibeVoice service: {e}")
    logger.info("Attempting to use mock VibeVoice service for testing...")
    try:
        from mock_vibevoice_service import MockVibeVoiceService
        vibevoice_service = MockVibeVoiceService()
        logger.info("Mock VibeVoice service initialized successfully")
    except Exception as mock_error:
        logger.error(f"Failed to initialize mock VibeVoice service: {mock_error}")
        vibevoice_service = None

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    content: str  # Or a summary

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    limit: int = 10

class SearchResponse(BaseModel):
    success: bool
    results: List[dict]
    message: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    chat_history: List[List[str]] = [] 
    page_numbers: Optional[List[int]] = None  

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []
    success: bool
    context_used: Optional[int] = None

class SummarizeRequest(BaseModel):
    document_id: str
    user_id: str
    page_numbers: Optional[List[int]] = None 

class FlashcardRequest(BaseModel):
    document_id: str
    user_id: str
    limit: int = 10
    page_numbers: Optional[List[int]] = None 

class QuizRequest(BaseModel):
    document_id: str
    user_id: str
    limit: int = 10
    page_numbers: Optional[List[int]] = None 
    level: str = "easy",
    format: str = "multiple_choice"

class MindmapRequest(BaseModel):
    document_id: str
    user_id: str

class TopicRequest(BaseModel):
    topic: str
    user_id: str

class Flashcard(BaseModel):
    question: str
    answer: str

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

class PodcastRequest(BaseModel):
    document_id: str
    user_id: str
    page_numbers: Optional[List[int]] = None

class VoicePodcastRequest(BaseModel):
    document_id: str
    user_id: str
    page_numbers: Optional[List[int]] = None
    speaker1: Optional[str] = None  # Voice preset name for Speaker 1
    speaker2: Optional[str] = None  # Voice preset name for Speaker 2
    cfg_scale: float = 1.3

class VoicePodcastResponse(BaseModel):
    script: str
    audio_file_path: str
    duration: float
    log: str
    success: bool 



@app.on_event("startup")
async def startup_event():
    try:
        vector_size = embedding_service.get_embedding_dimension()
        await qdrant_manager.get_or_create_company_collection(COLLECTION_NAME, vector_size)
        logger.info(f"Collection {COLLECTION_NAME} ready with vector size {vector_size}")
    except Exception as e:
        logger.error(f"Failed to initialize collection: {str(e)}")

@app.post("/upload-document", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    chunk_size: int = Form(1000),
    overlap: int = Form(200),
    language: str = Form("auto"),
):
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        processor = DocumentProcessor()
        supported_formats = processor.supported_formats
        
        if file_extension not in supported_formats:
            supported_text = ', '.join(processor.text_formats)
            supported_audio = ', '.join(processor.audio_formats) if processor.audio_formats else 'None (install audio dependencies)'
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. "
                       f"Supported text formats: {supported_text}. "
                       f"Supported audio formats: {supported_audio}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            logger.info(f"Processing document: {file.filename}")
            
            timestamp = int(datetime.now().timestamp())
            filename_hash = abs(hash(file.filename)) % 10000  
            document_id = f"doc_{timestamp}_{filename_hash}"
            current_time = datetime.now().isoformat()
            
            documents = []
            total_pages = 1  
            
            if processor.is_file_format(file.filename):
                logger.info(f"Using page-based processing for file format")
                chunks_with_pages, total_pages = processor.process_document_with_pages(
                    temp_file_path, file.filename, chunk_size, overlap
                )
                
                if not chunks_with_pages:
                    raise HTTPException(status_code=400, detail="No readable content found in the document")
                
                chunk_texts = [chunk['text'] for chunk in chunks_with_pages]
                
                logger.info(f"Creating embeddings for {len(chunk_texts)} chunks from {total_pages} pages...")
                embeddings = embedding_service.create_embeddings(chunk_texts)
                
                for chunk_data, embedding in zip(chunks_with_pages, embeddings):
                    doc = {
                        "id": str(uuid.uuid4()),  
                        "vector": embedding,
                        "payload": {
                            "document_id": document_id,
                            "user_id": user_id,
                            "file_name": file.filename,
                            "created_at": current_time,
                            "text": chunk_data['text'],
                            "chunk_index": chunk_data['chunk_index'],
                            "page_number": chunk_data['page_number'],  
                            "total_pages": chunk_data['total_pages'],  
                            "total_chunks": len(chunks_with_pages),
                            "processing_type": "page_based"  
                        }
                    }
                    documents.append(doc)
                
                # Store concatenated text for response (for compatibility)
                text = "\n".join(chunk_texts)
                
            elif processor.is_audio_file(file.filename):
                # EXISTING: Audio file processing
                logger.info(f"Processing audio file with language: {language}")
                text = processor.process_document(temp_file_path, file.filename, language)
                
                # Get transcription metadata for additional info
                audio_metadata = processor.get_last_audio_metadata()
                if audio_metadata:
                    logger.info(f"Audio transcription completed with confidence: {audio_metadata.get('confidence_score', 'N/A')}")
                
                if not text.strip():
                    raise HTTPException(status_code=400, detail="No text content found in the audio")
                
                # Traditional chunking for audio
                chunks = processor.chunk_text(text, chunk_size, overlap)
                logger.info(f"Created {len(chunks)} chunks from audio transcription")
                
                # Create embeddings
                logger.info("Creating embeddings...")
                embeddings = embedding_service.create_embeddings(chunks)
                
                # Prepare documents without page information (audio doesn't have pages)
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    doc = {
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "document_id": document_id,
                            "user_id": user_id,
                            "file_name": file.filename,
                            "created_at": current_time,
                            "text": chunk,
                            "chunk_index": i,
                            "page_number": None,  # NEW: No page number for audio
                            "total_pages": None,  # NEW: No pages for audio
                            "total_chunks": len(chunks),
                            "processing_type": "traditional"  # NEW: Processing type marker
                        }
                    }
                    documents.append(doc)
            else:
                # EXISTING: Traditional text document processing (fallback)
                text = processor.process_document(temp_file_path, file.filename)
                
                if not text.strip():
                    raise HTTPException(status_code=400, detail="No text content found in the document")
                
                chunks = processor.chunk_text(text, chunk_size, overlap)
                logger.info(f"Created {len(chunks)} chunks from document")
                
                embeddings = embedding_service.create_embeddings(chunks)
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    doc = {
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "document_id": document_id,
                            "user_id": user_id,
                            "file_name": file.filename,
                            "created_at": current_time,
                            "text": chunk,
                            "chunk_index": i,
                            "page_number": None,  # NEW: No page number for fallback processing
                            "total_pages": None,  # NEW: No pages for fallback processing  
                            "total_chunks": len(chunks),
                            "processing_type": "traditional"
                        }
                    }
                    documents.append(doc)
            
            # Insert documents into Qdrant
            logger.info(f"Inserting {len(documents)} document chunks into Qdrant...")
            await qdrant_manager.insert_documents(COLLECTION_NAME, documents)
            
            return DocumentResponse(
                document_id=document_id,
                filename=file.filename,
                content=text
            )
            
        finally:
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-url", response_model=DocumentResponse)
async def process_url(
    url: str = Form(...),
    user_id: str = Form(...),
    chunk_size: int = Form(1000),
    overlap: int = Form(200),
    language: str = Form("auto")
):
    media_processor = MediaProcessor()
    document_processor = DocumentProcessor()
    
    # 1. Get video info to use as filename
    video_info = media_processor.get_video_info(url)
    if not video_info:
        raise HTTPException(status_code=400, detail="Invalid or unsupported URL.")
        
    file_name = f"{video_info.get('title', 'media_file')}.wav"
    
    # 2. Download audio from URL to a temporary file
    temp_audio_path = media_processor.download_audio(url)
    if not temp_audio_path:
        raise HTTPException(status_code=500, detail="Failed to download audio from the URL.")
    
    try:
        # 3. Process the downloaded audio file (same as upload flow)
        logger.info(f"Processing downloaded audio: {file_name}")
        text = document_processor.process_document(temp_audio_path, file_name, language)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in the media.")

        # 4. Chunk, embed, and store in Qdrant (same as upload flow)
        chunks = document_processor.chunk_text(text, chunk_size, overlap)
        embeddings = embedding_service.create_embeddings(chunks)

        timestamp = int(datetime.now().timestamp())
        filename_hash = abs(hash(file_name)) % 10000
        document_id = f"doc_{timestamp}_{filename_hash}"
        current_time = datetime.now().isoformat()

        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "document_id": document_id,
                    "user_id": user_id,
                    "file_name": file_name,
                    "source_url": url, # Add source URL to metadata
                    "created_at": current_time,
                    "text": chunk,
                    "chunk_index": i,
                    "page_number": None,  # NEW: No page numbers for URL processing
                    "total_pages": None,  # NEW: No pages for URL processing
                    "total_chunks": len(chunks),
                    "processing_type": "traditional"  # NEW: Processing type marker
                }
            })

        await qdrant_manager.insert_documents(COLLECTION_NAME, documents)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file_name,
            content=text
        )
    
    finally:
        # 5. Clean up the temporary audio file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
            logger.info(f"Cleaned up temporary file: {temp_audio_path}")

# @app.post("/search", response_model=SearchResponse)
# async def search_documents(request: SearchRequest):
#     try:
#         query_embedding = embedding_service.create_single_embedding(request.query)
        
#         # Build filter conditions based on provided parameters
#         filter_conditions = None
#         conditions = []
        
#         if request.user_id:
#             conditions.append({"key": "user_id", "match": {"value": request.user_id}})
        
#         if request.document_id:
#             conditions.append({"key": "document_id", "match": {"value": request.document_id}})
        
#         if conditions:
#             filter_conditions = {"must": conditions}
        
#         # Search in Qdrant
#         search_results = qdrant_manager.search_documents(
#             collection_name=COLLECTION_NAME,
#             query_vector=query_embedding,
#             limit=request.limit,
#             filter_conditions=filter_conditions
#         )
        
#         # Format results
#         results = []
#         for result in search_results:
#             results.append({
#                 "score": result.score,
#                 "document_id": result.payload.get("document_id"),
#                 "file_name": result.payload.get("file_name"),
#                 "text": result.payload.get("text"),
#                 "user_id": result.payload.get("user_id"),
#                 "created_at": result.payload.get("created_at"),
#                 "chunk_index": result.payload.get("chunk_index")
#             })
        
#         return SearchResponse(
#             success=True,
#             results=results,
#             message=f"Found {len(results)} results"
#         )
        
#     except Exception as e:
#         logger.error(f"Error searching documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Document Upload API"}

# @app.get("/collections/{collection_name}/verify")
# async def verify_collection_setup(collection_name: str):
#     """Verify that the collection is properly set up for page-based filtering."""
#     try:
#         verification_report = await qdrant_manager.verify_collection_indices(collection_name)
#         return verification_report
#     except Exception as e:
#         logger.error(f"Error verifying collection {collection_name}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

# @app.post("/collections/{collection_name}/migrate")
# async def migrate_collection_for_page_filtering(collection_name: str):
#     """Migrate an existing collection to support page-based filtering."""
#     try:
#         success = await qdrant_manager.migrate_collection_for_page_filtering(collection_name)
#         if success:
#             return {"success": True, "message": "Collection migration completed successfully"}
#         else:
#             return {"success": False, "message": "Migration completed with some issues. Check logs."}
#     except Exception as e:
#         logger.error(f"Error migrating collection {collection_name}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Migration error: {str(e)}")

# @app.get("/collections/{collection_name}/info")
# async def get_collection_info(collection_name: str):
#     try:
#         info = qdrant_manager.get_collection_info(collection_name)
#         return {"success": True, "info": info}
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))

@app.get("/documents")
async def get_all_documents(user_id: str = None, limit: int = 100, offset: int = 0):
    """
    Retrieves all documents for a specific user.
    If user_id is not provided, it might return public documents or an error.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        # Assuming get_all_documents_for_user returns a list of document metadata
        documents = await qdrant_manager.get_all_documents_for_user(
            collection_name=COLLECTION_NAME,
            user_id=user_id,
            limit=limit
        )
        return documents
    except Exception as e:
        logger.error(f"Error getting documents for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")

# @app.get("/documents/{document_id}/pages")
# async def get_document_page_info(document_id: str, user_id: str):
#     """Get page information for a specific document."""
#     try:
#         page_info = await qdrant_manager.get_document_page_info(COLLECTION_NAME, document_id, user_id)
#         return page_info
#     except Exception as e:
#         logger.error(f"Error retrieving page info for document {document_id}: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve document page information.")

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")
        
        async def stream_chat():
            async for chunk in chatbot.stream_rag_response(
                query=request.message,
                user_id=request.user_id,
                document_id=request.document_id,
                page_numbers=request.page_numbers,  # NEW: Pass page filtering
                chat_history=request.chat_history # Pass chat history
            ):
                yield chunk

        return StreamingResponse(stream_chat(), media_type="text/plain")
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        async def stream_summary():
            async for chunk in chatbot.stream_summary(
                document_id=request.document_id,
                user_id=request.user_id,
                page_numbers=request.page_numbers  # NEW: Pass page filtering
            ):
                yield chunk

        return StreamingResponse(stream_summary(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/flashcards")
async def generate_flashcards(request: FlashcardRequest):
    """
    Generates and streams flashcards for a given document.
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        async def stream_flashcards():
            async for chunk in chatbot.stream_flashcards(
                document_id=request.document_id,
                user_id=request.user_id,
                page_numbers=request.page_numbers,  # NEW: Pass page filtering
                limit=request.limit
            ):
                yield chunk

        return StreamingResponse(stream_flashcards(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in flashcards endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Flashcard generation error: {str(e)}")

@app.post("/quiz")
async def generate_quiz(request: QuizRequest):
    """
    Generates and streams a quiz for a given document.
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        async def stream_quiz():
            async for chunk in chatbot.stream_quiz(
                document_id=request.document_id,
                user_id=request.user_id,
                page_numbers=request.page_numbers,  # NEW: Pass page filtering
                limit=request.limit,
                level=request.level,
                format=request.format
            ):
                yield chunk

        return StreamingResponse(stream_quiz(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in quiz endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quiz generation error: {str(e)}")


@app.post("/mindmap")
async def generate_mindmap(request: MindmapRequest):
    """
    Generates and streams a mindmap for a given document.
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        async def stream_mindmap():
            async for chunk in chatbot.stream_mindmap(
                document_id=request.document_id,
                user_id=request.user_id,
            ):
                yield chunk

        return StreamingResponse(stream_mindmap(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in mindmap endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mindmap generation error: {str(e)}")
    
@app.post("/podcast")
async def generate_podcast(request: PodcastRequest):
    """
    Generates and streams a podcast script for a given document.
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        async def stream_podcast():
            async for chunk in chatbot.stream_podcast(
                document_id=request.document_id,
                user_id=request.user_id,
                page_numbers=request.page_numbers,
            ):
                yield chunk

        return StreamingResponse(stream_podcast(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in podcast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Podcast generation error: {str(e)}")

@app.post("/voice-podcast", response_model=VoicePodcastResponse)
async def generate_voice_podcast(request: VoicePodcastRequest):
    """
    Generates a podcast script and converts it to voice audio for a given document.
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")
        
        if not vibevoice_service or not vibevoice_service.is_available():
            raise HTTPException(
                status_code=503, 
                detail="VibeVoice service is not available. This feature requires the VibeVoice model and voice samples to be installed. Please check the logs for more details."
            )

        # Step 1: Generate the script
        logger.info(f"Generating podcast script for document: {request.document_id}")
        script = ""
        async for chunk in chatbot.stream_podcast(
            document_id=request.document_id,
            user_id=request.user_id,
            page_numbers=request.page_numbers,
        ):
            script += chunk

        if not script.strip():
            raise HTTPException(status_code=500, detail="Failed to generate podcast script")

        # Step 2: Generate voice audio
        logger.info("Converting script to voice podcast")
        audio_file_path, generation_log = vibevoice_service.generate_voice_podcast(
            script=script,
            speaker1=request.speaker1,
            speaker2=request.speaker2,
            cfg_scale=request.cfg_scale
        )

        if audio_file_path is None:
            raise HTTPException(status_code=500, detail=f"Voice generation failed: {generation_log}")

        # Step 3: Calculate duration from saved file
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_file_path)
            duration = len(audio_data) / sample_rate
        except Exception as e:
            logger.warning(f"Could not read audio file for duration: {e}")
            duration = 0.0

        return VoicePodcastResponse(
            script=script,
            audio_file_path=audio_file_path,
            duration=duration,
            log=generation_log,
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice podcast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice podcast generation error: {str(e)}")

@app.get("/download-voice-podcast/{filename}")
async def download_voice_podcast(filename: str):
    """
    Download generated voice podcast file
    """
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error downloading voice podcast: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download audio file")

@app.post("/generate-from-topic", response_model=DocumentResponse)
async def generate_from_topic(request: TopicRequest):
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        # 1. Generate learning content from the topic
        logger.info(f"Generating learning module for topic: {request.topic}")
        content = chatbot.generate_learning_module(request.topic)
        
        if not content or not content.strip():
            raise HTTPException(status_code=500, detail="Failed to generate content for the topic.")

        # 2. Process and embed the generated content (similar to file upload)
        # Create a filename from the course and topic
        file_name = f"Course: {request.topic.replace(' ', '_').lower()}"

        # Split text into chunks
        chunks = document_processor.chunk_text(content, chunk_size=1000, overlap=200)
        logger.info(f"Created {len(chunks)} chunks from generated content")

        # Create embeddings for all chunks
        logger.info("Creating embeddings for generated content...")
        embeddings = embedding_service.create_embeddings(chunks)

        # Prepare documents for insertion
        timestamp = int(datetime.now().timestamp())
        document_id = f"topic_{timestamp}_{abs(hash(request.topic)) % 10000}"
        current_time = datetime.now().isoformat()
        
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "document_id": document_id,
                    "user_id": request.user_id,
                    "file_name": file_name,
                    "created_at": current_time,
                    "text": chunk,
                    "chunk_index": i,
                    "page_number": None,  # NEW: No page numbers for generated content
                    "total_pages": None,  # NEW: No pages for generated content
                    "total_chunks": len(chunks),
                    "processing_type": "traditional",  # NEW: Processing type marker
                    "source": "generated_topic"
                }
            }
            documents.append(doc)
        
        # Insert documents into Qdrant
        logger.info(f"Inserting {len(documents)} document chunks into Qdrant...")
        await qdrant_manager.insert_documents(COLLECTION_NAME, documents)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file_name,
            content=content
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating from topic: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)