import os
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PayloadSchemaType
import logging
from dotenv import load_dotenv
from typing import List
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
# QDRANT_URL = "https://cc102304-2c06-4d51-9dee-d436f4413549.us-west-1-0.aws.cloud.qdrant.io"
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cHs27o6erIf1BQHCdTxE4L4qZg4vCdrp51oNNNghjWM"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

class QdrantManager:
    def __init__(self):
        self.qdrant_client = AsyncQdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("Connected to Qdrant")

    async def get_or_create_company_collection(self, collection_name: str, vector_size: int = 384) -> str:
        try:
            # Check if collection already exists
            try:
                collection_info = await self.qdrant_client.get_collection(collection_name)
                print(f"Collection {collection_name} already exists")
                return collection_name
            except Exception:
                # Collection doesn't exist, create it
                print(f"Creating new collection: {collection_name}")

            await self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0,
                ),
            )
            
            # Create payload indices for efficient filtering
            payload_indices = {
                "document_id": PayloadSchemaType.KEYWORD,      # Essential for document filtering
                "user_id": PayloadSchemaType.KEYWORD,          # Essential for user isolation
                "file_name": PayloadSchemaType.KEYWORD,        # For document identification
                "created_at": PayloadSchemaType.KEYWORD,       # For sorting by date
                "chunk_index": PayloadSchemaType.INTEGER,      # For ordering chunks (numeric)
                "total_chunks": PayloadSchemaType.INTEGER,     # For metadata (numeric)
                "page_number": PayloadSchemaType.INTEGER,      # NEW: Essential for page filtering (numeric)
                "total_pages": PayloadSchemaType.INTEGER,      # NEW: For page range validation (numeric)
                "processing_type": PayloadSchemaType.KEYWORD,  # NEW: To distinguish processing types
                "text": PayloadSchemaType.TEXT,                # For full-text search (if needed)
            }
            
            for field_name, schema_type in payload_indices.items():
                try:
                    await self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=schema_type
                    )
                    logger.info(f"Created index for field: {field_name}")
                except Exception as e:
                    logger.warning(f"Failed to create index for {field_name}: {e}")
                    
            print(f"Successfully created collection '{collection_name}' with {len(payload_indices)} indexed fields")
            logger.info(f"Indexed fields: {list(payload_indices.keys())}")
            return collection_name
            
        except Exception as e:
            error_msg = f"Failed to create collection {collection_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def insert_documents(self, collection_name: str, documents: list, batch_size: int = 100) -> bool:
        try:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                if not batch_docs:
                    continue
                
                points = [
                    models.PointStruct(
                        id=doc["id"],
                        vector=doc["vector"],
                        payload=doc["payload"]
                    ) for doc in batch_docs
                ]

                await self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )

            print(f"Successfully inserted {len(documents)} documents into {collection_name}")
            return True

        except Exception as e:
            error_msg = f"Failed to insert documents into {collection_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def search_documents(self, collection_name: str, query_vector: list, limit: int = 5, filter_conditions: dict = None) -> list:
        try:
            query_filter = None
            if filter_conditions:
                must_conditions = []
                should_conditions = []
                
                # Handle must conditions
                if "must" in filter_conditions:
                    for condition in filter_conditions["must"]:
                        if "key" in condition and "match" in condition:
                            field_condition = models.FieldCondition(
                                key=condition["key"],
                                match=models.MatchValue(value=condition["match"]["value"])
                            )
                            must_conditions.append(field_condition)
                        elif "should" in condition:
                            # Handle nested should conditions (for page filtering)
                            for should_condition in condition["should"]:
                                if "key" in should_condition and "match" in should_condition:
                                    field_condition = models.FieldCondition(
                                        key=should_condition["key"],
                                        match=models.MatchValue(value=should_condition["match"]["value"])
                                    )
                                    should_conditions.append(field_condition)
                
                # Build the filter
                if must_conditions and should_conditions:
                    query_filter = models.Filter(must=must_conditions, should=should_conditions)
                elif must_conditions:
                    query_filter = models.Filter(must=must_conditions)
                elif should_conditions:
                    query_filter = models.Filter(should=should_conditions)
            
            search_result = await self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                search_params=models.SearchParams(hnsw_ef=128, exact=False),
                limit=limit,
            )
            
            return search_result.points
            
        except Exception as e:
            error_msg = f"Failed to search in collection {collection_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def retrieve_all_chunks(self, collection_name: str, document_id: str, user_id: str, limit: int = 2000) -> List[any]:
        """
        Retrieves all document chunks for a specific document_id using scrolling.
        """
        try:
            logger.info(f"Retrieving all chunks for document_id: {document_id}")
            
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )

            points, _ = await self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # No need for vectors here
            )
            
            logger.info(f"Retrieved {len(points)} chunks for document_id: {document_id}")
            return points

        except Exception as e:
            logger.error(f"Error retrieving all chunks from Qdrant: {str(e)}")
            return []

    async def retrieve_chunks_by_pages(self, collection_name: str, document_id: str, user_id: str, 
                                     page_numbers: List[int], limit: int = 2000) -> List[any]:
        """
        Retrieves document chunks for specific pages only.
        """
        try:
            logger.info(f"Retrieving chunks for document_id: {document_id}, pages: {page_numbers}")
            
            # Build base filter conditions (document_id and user_id)
            must_conditions = [
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
            
            # Create page filter conditions
            if len(page_numbers) == 1:
                # Single page filter - add directly to must conditions
                must_conditions.append(
                    models.FieldCondition(
                        key="page_number",
                        match=models.MatchValue(value=page_numbers[0])
                    )
                )
                scroll_filter = models.Filter(must=must_conditions)
            else:
                # Multiple pages filter - use should (OR) condition for pages
                page_conditions = []
                for page_num in page_numbers:
                    page_conditions.append(
                        models.FieldCondition(
                            key="page_number",
                            match=models.MatchValue(value=page_num)
                        )
                    )
                
                # Create filter with must (AND) for document/user and should (OR) for pages
                scroll_filter = models.Filter(
                    must=must_conditions,
                    should=page_conditions
                )

            points, _ = await self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # No need for vectors here
            )
            
            logger.info(f"Retrieved {len(points)} chunks for document_id: {document_id}, pages: {page_numbers}")
            return points

        except Exception as e:
            logger.error(f"Error retrieving chunks by pages from Qdrant: {str(e)}")
            return []

    async def get_all_documents_for_user(self, collection_name: str, user_id: str, limit: int = 100) -> List[dict]:
        """
        Retrieves a unique list of all documents for a given user.
        """
        try:
            # We need to scroll through all points to find unique document_ids
            all_points, _ = await self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
                    ]
                ),
                limit=2000, # Adjust this limit based on expected number of chunks
                with_payload=True,
                with_vectors=False
            )
            
            # Use a dictionary to store the latest version of each document
            unique_documents = {}
            for point in all_points:
                doc_id = point.payload.get("document_id")
                if doc_id not in unique_documents:
                    unique_documents[doc_id] = {
                        "document_id": doc_id,
                        "file_name": point.payload.get("file_name"),
                        "created_at": point.payload.get("created_at"),
                        "user_id": user_id,
                        "total_pages": point.payload.get("total_pages"),  # NEW: Include page info
                        "processing_type": point.payload.get("processing_type", "traditional")  # NEW: Include processing type
                    }

            # Return the list of unique documents, sorted by creation date
            sorted_docs = sorted(unique_documents.values(), key=lambda d: d.get('created_at', ''), reverse=True)
            print(f"Length of Sorted documents: {len(sorted_docs)}")
            return sorted_docs[:limit]

        except Exception as e:
            logger.error(f"Error retrieving documents for user {user_id}: {str(e)}")
            return []

    async def get_document_page_info(self, collection_name: str, document_id: str, user_id: str) -> dict:
        """
        Gets page information for a specific document.
        Returns total pages and available page numbers.
        """
        try:
            logger.info(f"Getting page info for document_id: {document_id}")
            
            # Get all chunks for this document
            all_points, _ = await self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
                    ]
                ),
                limit=2000,
                with_payload=True,
                with_vectors=False
            )
            
            if not all_points:
                return {"total_pages": 0, "available_pages": [], "processing_type": "unknown"}
            
            # Extract page information
            page_numbers = set()
            total_pages = 0
            processing_type = "traditional"
            
            for point in all_points:
                page_num = point.payload.get("page_number")
                if page_num is not None:
                    page_numbers.add(int(page_num))
                
                total_pages = max(total_pages, point.payload.get("total_pages", 0))
                processing_type = point.payload.get("processing_type", processing_type)
            
            return {
                "total_pages": total_pages,
                "available_pages": sorted(list(page_numbers)) if page_numbers else [],
                "processing_type": processing_type,
                "supports_page_filtering": processing_type == "page_based"
            }
            
        except Exception as e:
            logger.error(f"Error getting page info for document {document_id}: {str(e)}")
            return {"total_pages": 0, "available_pages": [], "processing_type": "unknown", "supports_page_filtering": False}

    # async def verify_collection_indices(self, collection_name: str) -> dict:
    #     """
    #     Verify that the collection has all required indices for page-based filtering.
    #     Returns a report of missing indices and suggestions.
    #     """
    #     try:
    #         logger.info(f"Verifying indices for collection: {collection_name}")
            
    #         # Get collection info
    #         collection_info = await self.qdrant_client.get_collection(collection_name)
            
    #         # Required indices for optimal page-based filtering
    #         required_indices = {
    #             "document_id": PayloadSchemaType.KEYWORD,
    #             "user_id": PayloadSchemaType.KEYWORD,
    #             "page_number": PayloadSchemaType.INTEGER,  # Critical for page filtering (numeric)
    #             "processing_type": PayloadSchemaType.KEYWORD,
    #             "chunk_index": PayloadSchemaType.INTEGER,  # Numeric for ordering
    #             "total_pages": PayloadSchemaType.INTEGER,  # Numeric for page validation
    #         }
            
    #         # Optional but recommended indices
    #         recommended_indices = {
    #             "file_name": PayloadSchemaType.KEYWORD,
    #             "created_at": PayloadSchemaType.KEYWORD,
    #             "total_chunks": PayloadSchemaType.INTEGER,  # Numeric for chunk count
    #             "text": PayloadSchemaType.TEXT,
    #         }
            
    #         verification_report = {
    #             "collection_exists": True,
    #             "vector_size": collection_info.config.params.vectors.size,
    #             "points_count": collection_info.points_count,
    #             "required_indices_status": {},
    #             "recommended_indices_status": {},
    #             "missing_critical_indices": [],
    #             "recommendations": []
    #         }
            
    #         # Note: Qdrant doesn't provide direct API to list indices,
    #         # so we'll make a simple test query to verify index effectiveness
    #         try:
    #             # Test page_number index (most critical)
    #             test_filter = models.Filter(
    #                 must=[
    #                     models.FieldCondition(
    #                         key="page_number",
    #                         match=models.MatchValue(value=1)
    #                     )
    #                 ]
    #             )
                
    #             # This will work efficiently only if page_number is indexed
    #             await self.qdrant_client.scroll(
    #                 collection_name=collection_name,
    #                 scroll_filter=test_filter,
    #                 limit=1,
    #                 with_payload=False,
    #                 with_vectors=False
    #             )
                
    #             verification_report["required_indices_status"]["page_number"] = "✅ Working"
                
    #         except Exception as e:
    #             verification_report["required_indices_status"]["page_number"] = f"❌ Error: {str(e)}"
    #             verification_report["missing_critical_indices"].append("page_number")
            
    #         # Add recommendations
    #         if verification_report["missing_critical_indices"]:
    #             verification_report["recommendations"].append(
    #                 "Critical: Re-create collection with proper page_number indexing"
    #             )
            
    #         verification_report["recommendations"].extend([
    #             "All documents should have 'page_number' field (null for non-page content)",
    #             "Use page-based processing for PDF, DOCX, TXT files",
    #             "Ensure 'processing_type' field distinguishes page-based vs traditional processing"
    #         ])
            
    #         return verification_report
            
    #     except Exception as e:
    #         logger.error(f"Error verifying collection indices: {str(e)}")
    #         return {
    #             "collection_exists": False,
    #             "error": str(e),
    #             "recommendations": ["Create collection with proper indices"]
    #         }

    # async def migrate_collection_for_page_filtering(self, collection_name: str) -> bool:
    #     """
    #     Migrate an existing collection to support page-based filtering.
    #     This adds missing indices if possible.
    #     """
    #     try:
    #         logger.info(f"Migrating collection {collection_name} for page filtering...")
            
    #         # Try to add missing indices
    #         new_indices = {
    #             "page_number": PayloadSchemaType.INTEGER,      
    #             "total_pages": PayloadSchemaType.INTEGER,      
    #             "processing_type": PayloadSchemaType.KEYWORD,  
    #             "chunk_index": PayloadSchemaType.INTEGER,      
    #         }
            
    #         success_count = 0
    #         for field_name, schema_type in new_indices.items():
    #             try:
    #                 await self.qdrant_client.create_payload_index(
    #                     collection_name=collection_name,
    #                     field_name=field_name,
    #                     field_schema=schema_type
    #                 )
    #                 logger.info(f"Added index for field: {field_name}")
    #                 success_count += 1
    #             except Exception as e:
    #                 logger.warning(f"Could not add index for {field_name} (may already exist): {e}")
            
    #         logger.info(f"Migration completed. Added {success_count} new indices.")
    #         return success_count > 0
            
    #     except Exception as e:
    #         logger.error(f"Error migrating collection: {str(e)}")
    #         return False

# Example usage
async def main():
    try:
        qdrant_manager = QdrantManager()
        collection_name = "Tutor-Documents"
        result = await qdrant_manager.get_or_create_company_collection(collection_name)
        print(f"Collection name: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())