import os
from google import genai
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

from Qdrant import QdrantManager
from embedding_service import EmbeddingService

load_dotenv()
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.model_name = 'gemini-2.5-flash'
        
        # Initialize services
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.qdrant_manager = QdrantManager()
        
        # Initialize embedding service (preferably local for consistency)
        try:
            self.embedding_service = EmbeddingService(use_openai=False)
            logger.info("Using local embeddings for RAG")
        except Exception:
            self.embedding_service = EmbeddingService(use_openai=True)
            logger.info("Using OpenAI embeddings for RAG")
    
    async def search_relevant_documents(self, query: str, user_id: Optional[str] = None, 
                                document_id: Optional[str] = None, page_numbers: Optional[List[int]] = None, 
                                limit: int = 5) -> List[Dict]:
        """Search for relevant documents based on query with optional page filtering."""
        try:
            # Create embedding for the query
            query_embedding = self.embedding_service.create_single_embedding(query)
            
            # Build filter conditions
            filter_conditions = None
            must_conditions = []
            
            if user_id:
                must_conditions.append({"key": "user_id", "match": {"value": user_id}})
            
            if document_id:
                must_conditions.append({"key": "document_id", "match": {"value": document_id}})
            
            # NEW: Add page number filtering
            if page_numbers and len(page_numbers) > 0:
                if len(page_numbers) == 1:
                    # Single page filter - add to must conditions
                    must_conditions.append({"key": "page_number", "match": {"value": page_numbers[0]}})
                    filter_conditions = {"must": must_conditions}
                else:
                    # Multiple pages filter - use both must and should
                    should_conditions = []
                    for page_num in page_numbers:
                        should_conditions.append({"key": "page_number", "match": {"value": page_num}})
                    
                    # For multiple pages, we need must (user/doc) AND should (any of the pages)
                    must_conditions.append({"should": should_conditions})
                    filter_conditions = {"must": must_conditions}
            else:
                # No page filtering - use only must conditions
                if must_conditions:
                    filter_conditions = {"must": must_conditions}
            
            # Search in Qdrant
            search_results = await self.qdrant_manager.search_documents(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=10,
                filter_conditions=filter_conditions
            )
            
            # Format results with page information
            relevant_docs = []
            for result in search_results:
                relevant_docs.append({
                    "text": result.payload.get("text", ""),
                    "file_name": result.payload.get("file_name", ""),
                    "document_id": result.payload.get("document_id", ""),
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "page_number": result.payload.get("page_number"),  # NEW: Include page number
                    "total_pages": result.payload.get("total_pages"),  # NEW: Include total pages
                    "processing_type": result.payload.get("processing_type", "traditional")  # NEW: Processing type
                })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def stream_summary(self, document_id: str, user_id: str, page_numbers: Optional[List[int]] = None, max_context_length: int = 15000):
        """Fetches all document content and streams a summary from Gemini."""
        try:
            # 1. Fetch chunks for the document (with optional page filtering)
            if page_numbers and len(page_numbers) > 0:
                # Use page-filtered retrieval
                doc_points = await self.qdrant_manager.retrieve_chunks_by_pages(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    page_numbers=page_numbers,
                    limit=500
                )
            else:
                # Use all chunks (existing behavior)
                doc_points = await self.qdrant_manager.retrieve_all_chunks(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    limit=500
                )

            if not doc_points:
                if page_numbers:
                    yield f"Could not find content for pages {page_numbers} in the document to summarize."
                else:
                    yield "Could not find the document to summarize."
                return

            sorted_chunks = sorted(doc_points, key=lambda p: p.payload.get('chunk_index', 0))
            full_text = "\n".join([point.payload['text'] for point in sorted_chunks])
            
            # Add page information to the prompt if specific pages were requested
            page_info = ""
            if page_numbers and len(page_numbers) > 0:
                if len(page_numbers) == 1:
                    page_info = f"\n(Summary for page {page_numbers[0]} only)\n"
                else:
                    page_info = f"\n(Summary for pages {', '.join(map(str, page_numbers))} only)\n"

            if len(full_text) > max_context_length:
                full_text = full_text[:max_context_length] + "\n... (Content truncated for summary)"

            # 3. Create the summarization prompt.
            prompt = f"""You are a helpful AI assistant specialized in summarizing documents.
            Provide a clear, concise, and comprehensive summary of the following document.
            The summary should cover the key points, main arguments, and any important conclusions.
            Use bullet points for clarity where appropriate.

            Document Content:
            ---
            {full_text}
            ---

            Summary:"""

            # 4. Stream the response from Gemini.
            # Accumulate the full response instead of yielding chunks
            full_response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt],
            ):
                if chunk.text:
                    full_response += chunk.text
            yield full_response

        except Exception as e:
            logger.error(f"Error streaming summary: {str(e)}")
            yield f"Error generating summary: {str(e)}"

    async def stream_rag_response(self, query: str, user_id: Optional[str] = None,
                           document_id: Optional[str] = None, page_numbers: Optional[List[int]] = None,
                           max_context_length: int = 4000, chat_history: Optional[List[List[str]]] = None):
        """Stream RAG response from Gemini."""
        try:
            # Search for relevant documents
            relevant_docs = await self.search_relevant_documents(
                query=query, 
                user_id=user_id, 
                document_id=document_id,
                page_numbers=page_numbers,  # NEW: Pass page filtering
                limit=5
            )
            
            # Build context from relevant documents
            context_parts = []
            sources = []
            current_length = 0
            
            for doc in relevant_docs:
                doc_text = doc["text"]
                if current_length + len(doc_text) > max_context_length:
                    break
                
                context_parts.append(f"Document: {doc['file_name']}\nContent: {doc_text}")
                sources.append({
                    "file_name": doc["file_name"],
                    "document_id": doc["document_id"],
                    "score": doc["score"],
                    "chunk_index": doc["chunk_index"]
                })
                current_length += len(doc_text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create prompt for Gemini
            prompt_parts = []
            prompt_parts.append("You are an AI tutor assistant. Answer the user's question based on the provided context from uploaded documents.\n")

            if chat_history:
                prompt_parts.append("Here is the past conversation for context:")
                for human_message, ai_message in chat_history:
                    if human_message: 
                        prompt_parts.append(f"User: {human_message}")
                    if ai_message: 
                        prompt_parts.append(f"AI: {ai_message}")
                prompt_parts.append("\n---")

            if context:
                prompt_parts.append(f"\nContext from documents:\n{context}\n")
            else:
                prompt_parts.append("\nNo relevant document context was found. Try rephrasing your question or uploading more documents.\n")

            prompt_parts.append(f"""User Question: {query}
            Instructions:
            - If this is a general greeting (like "hello", "hi", "hey", "good morning", etc.), respond warmly and ask how you can help with their documents
            - For questions about documents, answer based primarily on the provided context
            - If the context doesn't contain enough information for a document-related question, suggest they rephrase or upload more documents
            - If the question relates to chat history, respond with a proper answer based on chat history
            - Be helpful, educational, and conversational in your response
            Answer:""")

            prompt = "\n".join(prompt_parts)
            
            # Stream the response from Gemini
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            ):
                yield chunk.text

        except Exception as e:
            logger.error(f"Error streaming RAG response: {str(e)}")
            yield f"Error generating RAG response: {str(e)}"

    async def stream_flashcards(self, document_id: str, user_id: str, page_numbers: Optional[List[int]] = None, limit: int = 10, max_context_length: int = 15000):
        """Fetches all document content and streams flashcards from Gemini."""
        try:
            # 1. Fetch chunks for the document (with optional page filtering)
            if page_numbers and len(page_numbers) > 0:
                # Use page-filtered retrieval
                doc_points = await self.qdrant_manager.retrieve_chunks_by_pages(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    page_numbers=page_numbers,
                    limit=1500
                )
            else:
                # Use all chunks (existing behavior)
                doc_points = await self.qdrant_manager.retrieve_all_chunks(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    limit=1500  # Adjust as needed
                )

            if not doc_points:
                if page_numbers:
                    yield f"Could not find content for pages {page_numbers} in the document to create flashcards from."
                else:
                    yield "Could not find the document to create flashcards from."
                return

            sorted_chunks = sorted(doc_points, key=lambda p: p.payload.get('chunk_index', 0))
            full_text = "\n".join([point.payload['text'] for point in sorted_chunks])
            
            # Add page information to the prompt if specific pages were requested
            page_info = ""
            if page_numbers and len(page_numbers) > 0:
                if len(page_numbers) == 1:
                    page_info = f"\n(Flashcards for page {page_numbers[0]} only)\n"
                else:
                    page_info = f"\n(Flashcards for pages {', '.join(map(str, page_numbers))} only)\n"

            if len(full_text) > max_context_length:
                full_text = full_text[:max_context_length] + "\n... (Content truncated for flashcards)"

            # 2. Create the flashcard generation prompt.
            prompt = f"""You are a helpful AI assistant designed to create educational flashcards from a document.
            Based on the document content provided, generate exactly {limit} flashcards. Each flashcard should consist of a clear question and a concise answer.

            You MUST output your response as a single, valid JSON array of objects. Each object should have two keys: "question" and "answer". Do not include any text or formatting outside of the JSON array.

            Example format:
            [
                {{
                    "question": "What is the capital of France?",
                    "answer": "The capital of France is Paris."
                }},
                {{
                    "question": "What is the formula for water?",
                    "answer": "H2O"
                }}
            ]

            Document Content:
            ---
            {full_text}
            ---

            Flashcards (JSON format):"""

            # 3. Stream the response from Gemini.
            # We will accumulate the full response and yield it at the end as a single JSON string.
            full_response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            ):
                if chunk.text:
                    full_response += chunk.text
            
            # Clean up the response to ensure it's a valid JSON array
            # This is a common step as LLMs can sometimes add markdown backticks
            cleaned_response = full_response.strip().replace("```json", "").replace("```", "").strip()
            
            yield cleaned_response

        except Exception as e:
            logger.error(f"Error streaming flashcards: {str(e)}")
            yield f'{{"error": "Error generating flashcards: {str(e)}"}}'

    async def stream_quiz(self, document_id: str, user_id: str, page_numbers: Optional[List[int]] = None, limit: int = 10, max_context_length: int = 15000, level: str = "easy", format: str = "multiple_choice"):
        """Fetches all document content and streams quiz questions from Gemini."""
        try:
            # 1. Fetch chunks for the document (with optional page filtering)
            if page_numbers and len(page_numbers) > 0:
                # Use page-filtered retrieval
                doc_points = await self.qdrant_manager.retrieve_chunks_by_pages(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    page_numbers=page_numbers,
                    limit=1500
                )
            else:
                # Use all chunks (existing behavior)
                doc_points = await self.qdrant_manager.retrieve_all_chunks(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    limit=1500  # Adjust as needed
                )

            if not doc_points:
                if page_numbers:
                    yield f"Could not find content for pages {page_numbers} in the document to create quiz from."
                else:
                    yield "Could not find the document to create quiz from."
                return

            sorted_chunks = sorted(doc_points, key=lambda p: p.payload.get('chunk_index', 0))
            full_text = "\n".join([point.payload['text'] for point in sorted_chunks])
            
            # Add page information to the prompt if specific pages were requested
            page_info = ""
            if page_numbers and len(page_numbers) > 0:
                if len(page_numbers) == 1:
                    page_info = f"\n(Quiz for page {page_numbers[0]} only)\n"
                else:
                    page_info = f"\n(Quiz for pages {', '.join(map(str, page_numbers))} only)\n"

            if len(full_text) > max_context_length:
                full_text = full_text[:max_context_length] + "\n... (Content truncated for quiz)"

            # 2. Create the quiz generation prompt based on the format
            if format == "multiple_choice":
                prompt = f"""You are a helpful AI assistant designed to create educational multiple choice quiz questions.
                Based on the **information and concepts presented in the following text**, generate exactly {limit} multiple choice quiz questions at a {level} level. Each question should have 4 options (A, B, C, D) and one correct answer.

                IMPORTANT: Ensure the questions are about the *subject matter* covered in the text, not about the document itself (e.g., avoid phrases like "According to the document", "The text states", "In this course structure").

                You MUST output your response as a single, valid JSON array of objects. Each object should have three keys: "question", "options", and "answer". The options should be an array of 4 strings, and the answer should be the letter (A, B, C, or D) of the correct option.

                Example format:
                [
                    {{
                        "question": "What is the capital of France?",
                        "options": ["London", "Paris", "Berlin", "Madrid"],
                        "answer": "B"
                    }},
                    {{
                        "question": "What is the chemical formula for water?",
                        "options": ["H2O", "CO2", "O2", "H2SO4"],
                        "answer": "A"
                    }}
                ]

                Document Content:
                ---
                {full_text}
                ---

                Quiz Questions (JSON format):"""
            elif format == "fill_in_the_blanks":
                prompt = f"""You are a helpful AI assistant designed to create fill-in-the-blank quiz questions.
                Based on the **information and concepts presented in the following text**, generate exactly {limit} fill-in-the-blank questions at a {level} level. Each question should have a statement with a blank to fill in.

                IMPORTANT: Ensure the questions are about the *subject matter* covered in the text, not about the document itself (e.g., avoid phrases like "According to the document", "The text states", "In this course structure").

                You MUST output your response as a single, valid JSON array of objects. Each object should have two keys: "question" and "answer". The question should contain a blank represented by "_____", and the answer should be the correct word or phrase.

                Example format:
                [
                    {{
                        "question": "The capital of France is _____.",
                        "answer": "Paris"
                    }},
                    {{
                        "question": "Water is made up of _____.",
                        "answer": "H2O"
                    }}
                ]

                Document Content:
                ---
                {full_text}
                ---

                Fill-in-the-Blank Questions (JSON format):"""
            elif format == "true_false":
                prompt = f"""You are a helpful AI assistant designed to create true/false quiz questions.
                Based on the **information and concepts presented in the following text**, generate exactly {limit} true/false questions at a {level} level. Each question should be a statement that can be answered with "True" or "False".

                IMPORTANT: Ensure the questions are about the *subject matter* covered in the text, not about the document itself (e.g., avoid phrases like "According to the document", "The text states", "In this course structure").

                You MUST output your response as a single, valid JSON array of objects. Each object should have two keys: "question" and "answer". The question should be a statement, and the answer should be either "True" or "False".

                Example format:
                [
                    {{
                        "question": "The capital of France is Paris.",
                        "answer": "True"
                    }},
                    {{
                        "question": "Water is a solid at room temperature.",
                        "answer": "False"
                    }}
                ]

                Document Content:
                ---
                {full_text}
                ---

                True/False Questions (JSON format):"""

            # 3. Stream the response from Gemini.
            # We will accumulate the full response and yield it at the end as a single JSON string.
            full_response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            ):
                if chunk.text:
                    full_response += chunk.text
            
            # Clean up the response to ensure it's a valid JSON array
            # This is a common step as LLMs can sometimes add markdown backticks
            cleaned_response = full_response.strip().replace("```json", "").replace("```", "").strip()
            
            yield cleaned_response

        except Exception as e:
            logger.error(f"Error streaming quiz: {str(e)}")
            yield f'{{"error": "Error generating quiz: {str(e)}"}}'

    async def stream_mindmap(self, document_id: str, user_id: str, max_context_length: int = 15000):
        """Fetches all document content and streams mindmap from Model."""
        try:
            # 1. Fetch all chunks for the document.
            doc_points = await self.qdrant_manager.retrieve_all_chunks(
                collection_name=self.collection_name,
                document_id=document_id,
                user_id=user_id,
                limit=1500
            )

            if not doc_points:
                yield "Could not find the document to create mindmap from."
                return

            sorted_chunks = sorted(doc_points, key=lambda p: p.payload.get('chunk_index', 0))
            full_text = "\n".join([point.payload['text'] for point in sorted_chunks])

            if len(full_text) > max_context_length:
                full_text = full_text[:max_context_length] + "\n... (Content truncated for quiz)"

            # 2. Create the quiz generation prompt.
            prompt = f"""Create a mind map from the uploaded document in Markdown format for Markmap.js visualization.

            IMPORTANT: Follow this EXACT structure:

            # [Main Topic]

            ## [Primary Branch 1]
            ### [Subtopic A]
            - Detail 1
            - Detail 2
            - Detail 3

            ### [Subtopic B]
            - Key Point 1
            - Key Point 2

            ## [Primary Branch 2]
            ### [Subtopic C]
            - Concept 1
            - Concept 2

            RULES:
            1. Use # for the central topic (only ONE # heading)
            2. Use ## for main branches (4-6 maximum)
            3. Use ### for subtopics (2-4 per branch)
            4. Use - for bullet points (details/examples)
            5. Keep headings concise (1-4 words)
            6. Use bullet points for specific details
            7. Maximum 4 levels deep (# ## ### -)
            8. No code blocks, just raw Markdown
            9. No bold/italic formatting in headings
            10. Output ONLY the Markdown, no explanations

            Focus on creating educational value with:
            - Key concepts and definitions
            - Practical applications
            - Real-world examples
            - Important formulas or processes

            Document content:
            ---
            {full_text}
            ---

            Mindmap (Markdown format):"""

            # 3. Stream the response from Gemini.
            # We will accumulate the full response and yield it at the end as a single JSON string.
            full_response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            ):
                if chunk.text:
                    full_response += chunk.text
            
            yield full_response.strip()

        except Exception as e:
            logger.error(f"Error streaming mindmap: {str(e)}")
            yield f'{{"error": "Error generating mindmap: {str(e)}"}}'
    
    async def stream_podcast(self, document_id: str, user_id: str, page_numbers: Optional[List[int]] = None, max_context_length: int = 15000):
        """Fetches all document content and streams a podcast from Gemini."""
        try:
            # 1. Fetch chunks for the document (with optional page filtering)
            if page_numbers and len(page_numbers) > 0:
                # Use page-filtered retrieval
                doc_points = await self.qdrant_manager.retrieve_chunks_by_pages(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    page_numbers=page_numbers,
                    limit=500
                )
            else:
                # Use all chunks (existing behavior)
                doc_points = await self.qdrant_manager.retrieve_all_chunks(
                    collection_name=self.collection_name,
                    document_id=document_id,
                    user_id=user_id,
                    limit=500
                )

            if not doc_points:
                if page_numbers:
                    yield f"Could not find content for pages {page_numbers} in the document to create podcast from."
                else:
                    yield "Could not find the document to create podcast from."
                return

            sorted_chunks = sorted(doc_points, key=lambda p: p.payload.get('chunk_index', 0))
            full_text = "\n".join([point.payload['text'] for point in sorted_chunks])
            
            # Add page information to the prompt if specific pages were requested
            page_info = ""
            if page_numbers and len(page_numbers) > 0:
                if len(page_numbers) == 1:
                    page_info = f"\n(Podcast for page {page_numbers[0]} only)\n"
                else:
                    page_info = f"\n(Podcast for pages {', '.join(map(str, page_numbers))} only)\n"

            if len(full_text) > max_context_length:
                full_text = full_text[:max_context_length] + "\n... (Content truncated for podcast)"

            # 3. Create the podcast generation prompt.
            prompt = f"""You are a helpful AI assistant specialized in creating conversational podcast scripts from documents.

            Create an engaging podcast script in the following format:
            - Use "Speaker 1:" and "Speaker 2:" for dialogue (do NOT use placeholder names like [Speaker 1's Name])
            - Make it conversational and natural, like two people discussing the content
            - Cover the key points, main arguments, and important conclusions
            - Use a friendly, engaging tone with natural speech patterns
            - Include transitions between topics
            - Make it sound like a real conversation between knowledgeable hosts
            - Each speaker should contribute meaningfully to the discussion
            - Use examples and analogies to make complex topics accessible
            - Start directly with the conversation, no introductions or setup

            Document Content:
            ---
            {full_text}
            ---

            Podcast Script (in conversational dialogue format):"""

            # 4. Stream the response from Gemini.
            # Accumulate the full response instead of yielding chunks
            full_response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt],
            ):
                if chunk.text:
                    full_response += chunk.text
            yield full_response

        except Exception as e:
            logger.error(f"Error streaming podcast: {str(e)}")
            yield f"Error generating podcast: {str(e)}"    

    def generate_learning_module(self, topic: str) -> str:
        """Generates a learning module on a given topic."""
        # prompt = f"""
        # You are an expert educator and content creator.
        # Your task is to generate a comprehensive, well-structured learning module on the topic: '{topic}'.
        # The module should be written as a single piece of text, suitable for a student who is new to the subject.
        # Please ensure the content is accurate, clear, and engaging. Include the following sections:
        # 1. An introduction to the topic.
        # 2. Key concepts with detailed explanations.
        # 3. Examples to illustrate the concepts.
        # 4. A summary of the main points.

        # Generate the content now.
        # """

        prompt = f"""
    You are an expert educational designer. Create a comprehensive course structure for: "{topic}"
        
    Requirements:
    1. Analyze the topic's nature (theoretical/practical, complexity, domain)
    2. Determine optimal learning sequence
    3. Identify natural breakpoints and modules
    4. Suggest appropriate content types for each section
    5. Include assessment strategies
    6. Consider prerequisite knowledge
    
    Provide:
    - Course outline with 5-8 main modules
    - Learning objectives for each module
    - Estimated time for each section
    - Content type recommendations (theory, examples, practice, projects)
    - Assessment methods
    - Prerequisites and dependencies
        """

        response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
        return response.text

chatbot_service = RAGChatbot()
 