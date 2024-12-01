import os
from typing import List, Optional, Generator
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import shutil
from PIL import Image
from io import BytesIO
import base64

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings,
)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.core.schema import ImageNode

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from ell import init, simple
from ell import Message as EllMessage
from openai import OpenAI

# Load environment variables
load_dotenv()

class RAGError(Exception):
    """Base exception class for RAG-related errors"""
    pass

class DocumentLoadError(RAGError):
    """Raised when documents cannot be loaded"""
    pass

class RAGAgent:
    def __init__(
        self,
        input_dir: str = "./input",
        output_dir: str = "./out",
        chunk_size: int = 4096,
        chunk_overlap: int = 20
    ):
        self.input_dir = input_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Vector store path
        self.vector_store_path = self.output_dir / "vector_store"

        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize Claude 3.5 Sonnet for both RAG and chat
        self.llm = Anthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            max_tokens=1024
        )

        # Initialize Claude 3.5 for multimodal understanding
        self.multimodal_llm = AnthropicMultiModal(
            model="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
            max_tokens=1024
        )

        # Text splitter for chunking
        self.node_parser = SimpleNodeParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.node_parser

        # Initialize BERTopic
        self.topic_model = None

        init()  # Initialize ell

        # Initialize OpenAI client for GPT-4V
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.setup_rag()

    def _process_pdf_with_images(self, pdf_path: str) -> List[Document]:
        """Process PDF and extract both text and images with descriptions"""
        from pdf2image import convert_from_path
        import pytesseract
        import fitz  # PyMuPDF

        documents = []
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            print(f"📄 Processing page {page_num + 1}/{len(pdf_document)} of {pdf_path}")
            
            # Extract text
            text = page.get_text()
            if text.strip():
                doc = Document(
                    text=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "type": "text"
                    }
                )
                documents.append(doc)
            
            # Extract images
            image_list = page.get_images()
            if image_list:
                print(f"🖼️  Found {len(image_list)} images on page {page_num + 1}")
                
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(BytesIO(image_bytes))
                    
                    # Convert PIL Image to base64 string
                    buffered = BytesIO()
                    image.save(buffered, format=image.format or 'PNG')
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Create image node with base64 string
                    image_node = ImageNode(
                        image=img_str,
                        metadata={
                            "source": pdf_path,
                            "page": page_num,
                            "image_index": img_index,
                            "type": "image"
                        }
                    )
                    
                    # Get image description using GPT-4V
                    try:
                        description = self._get_image_description(image_node)
                        if description:
                            # Create document with image description
                            doc = Document(
                                text=description,
                                metadata={
                                    "source": pdf_path,
                                    "page": page_num,
                                    "image_index": img_index,
                                    "type": "image_description"
                                }
                            )
                            documents.append(doc)
                            print(f"✅ Processed image {img_index + 1} on page {page_num + 1}")
                    except Exception as e:
                        print(f"⚠️ Error getting description for image {img_index} on page {page_num}: {str(e)}")
                        continue
                    
                except Exception as e:
                    print(f"⚠️ Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
        
        return documents

    def _get_image_description(self, image_node: ImageNode) -> str:
        """Get description of image using GPT-4V"""
        try:
            # Resize image to 150 DPI
            image_bytes = base64.b64decode(image_node.image)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert RGBA to RGB if needed
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate new size for 150 DPI
            dpi = 150
            width = int(image.width * dpi / 72)  # Convert from default 72 DPI
            height = int(image.height * dpi / 72)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to JPEG
            buffered = BytesIO()
            image.save(buffered, format='JPEG', quality=61)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Call GPT-4o-mini
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in great detail. No intro, no explanations, only plain text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=512,
                temperature=0.1
            )
            
            # Extract the description from the response
            description = response.choices[0].message.content
            return description.strip()

        except Exception as e:
            print(f"⚠️ Error getting image description: {str(e)}")
            return ""

    def _create_new_index(self) -> None:
        """Create new vector index from documents"""
        try:
            print("\n📚 Loading documents from input directory...")
            
            all_documents = []
            for pdf_file in Path(self.input_dir).glob("*.pdf"):
                print(f"Processing {pdf_file}...")
                documents = self._process_pdf_with_images(str(pdf_file))
                all_documents.extend(documents)

            if not all_documents:
                raise DocumentLoadError(f"No documents found in {self.input_dir}")

            print(f"📄 Found {len(all_documents)} documents to process")

            # Create vector index first
            print("\n🔍 Creating vector index...")
            self.index = VectorStoreIndex.from_documents(
                all_documents
            )

            # Now process with BERTopic after index is created
            print("\n🧠 Processing documents with BERTopic...")
            print("This may take a few minutes depending on the document size...")
            processed_documents = self._process_documents_with_bertopic(all_documents)

            # Generate questions for each document
            print("\n❓ Generating questions for each document...")
            processed_documents = self._generate_questions(processed_documents)

            # Update the index with processed documents
            print("\n📝 Updating index with processed documents...")
            self.index = VectorStoreIndex.from_documents(
                processed_documents
            )

            # Persist the index
            print("💾 Saving vector store to disk...")
            self.index.storage_context.persist(
                persist_dir=str(self.vector_store_path)
            )
            print("✅ Index creation complete!")

        except Exception as e:
            raise DocumentLoadError(f"Failed to create index: {str(e)}")

    def setup_rag(self) -> None:
        """Initialize RAG system with persistence"""
        try:
            import shutil  # Move import to the top of the method
            from datetime import datetime

            print("\n🔧 Initializing RAG system...")
            # Check if vector store exists
            if self.vector_store_path.exists():
                print("📂 Found existing vector store.")

                # Check embedding dimension consistency
                if self._check_embedding_dimension():
                    print("✅ Embedding dimensions match. Loading index from disk...")
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(self.vector_store_path)
                    )
                    self.index = load_index_from_storage(
                        storage_context=storage_context
                    )
                    print("✅ Successfully loaded existing index")
                else:
                    print("⚠️ Embedding dimensions do not match. Rebuilding the index...")
                    # Create backup with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = self.output_dir / f"vector_store_backup_{timestamp}"
                    
                    # Remove old backup if it exists
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    
                    # Move current vector store to backup
                    shutil.move(str(self.vector_store_path), str(backup_path))
                    print(f"📦 Backed up existing index to {backup_path}")
                    
                    self._create_new_index()
            else:
                print("🆕 No existing index found, creating new one...")
                self._create_new_index()

            # Create query engine
            print("🔄 Configuring query engine...")
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5
            )
            print("✨ RAG system ready! You can start asking questions.\n")

        except Exception as e:
            raise RAGError(f"Failed to setup RAG system: {str(e)}")

    def _check_embedding_dimension(self) -> bool:
        """Check if the stored embeddings match the current embedding dimension"""
        try:
            # Load a sample embedding from the index
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.vector_store_path)
            )
            index = load_index_from_storage(storage_context=storage_context)
            doc_id = next(iter(index.docstore.docs))
            stored_embedding = index.docstore.get_node(doc_id).get_embedding()
            if stored_embedding is None:
                return False
            stored_dim = len(stored_embedding)
            current_dim = len(self.embed_model.get_text_embedding("test"))
            return stored_dim == current_dim
        except Exception as e:
            print(f"⚠️ Error checking embedding dimensions: {str(e)}")
            return False

    def _initialize_bertopic(self):
        """Initialize BERTopic model"""
        if not hasattr(self, 'index'):
            print("⚠️ Warning: Index not initialized yet")
            return

        # Get document count for dynamic configuration
        doc_count = len(list(self.index.docstore.docs.values()))
        print(f"📊 Configuring BERTopic for {doc_count} documents...")

        # For very small document sets, use minimal configuration
        if doc_count < 5:
            print("⚠️ Small document set detected, using minimal topic configuration...")
            from sklearn.cluster import KMeans
            
            self.topic_model = BERTopic(
                embedding_model=self.embed_model.get_text_embedding,
                min_topic_size=1,
                n_gram_range=(1, 1),
                calculate_probabilities=True,
                verbose=False,
                hdbscan_model=KMeans(n_clusters=max(2, doc_count // 2))
            )
        else:
            # For larger document sets, use standard configuration
            n_neighbors = max(2, min(doc_count - 1, 15))
            n_components = max(2, min(doc_count - 1, 5))
            
            from hdbscan import HDBSCAN
            
            self.topic_model = BERTopic(
                embedding_model=self.embed_model.get_text_embedding,
                umap_model=UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                ),
                vectorizer_model=CountVectorizer(
                    ngram_range=(1, 1),
                    stop_words="english",
                    min_df=1
                ),
                hdbscan_model=HDBSCAN(
                    min_cluster_size=2,
                    min_samples=1,
                    cluster_selection_method='eom',
                    prediction_data=True
                ),
                min_topic_size=1,
                nr_topics="auto",
                calculate_probabilities=True,
                verbose=False
            )

    def _process_documents_with_bertopic(self, documents: List[Document]) -> List[Document]:
        """Process documents using BERTopic and enrich metadata"""
        # Prepare documents text
        docs_text = [doc.get_text() for doc in documents]

        # Initialize BERTopic model
        print("⚙️  Initializing BERTopic model...")
        self._initialize_bertopic()

        print("📊 Fitting BERTopic model to documents...")
        topics, probabilities = self.topic_model.fit_transform(docs_text)
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        print(f"\n🏷️  Discovered {len(topic_info)} topics in documents")

        # Enrich documents with topic metadata
        print("✍️  Enriching documents with topic information...")
        processed_documents = []
        for doc, topic, prob in zip(documents, topics, probabilities):
            doc_extra_info = doc.extra_info or {}
            doc_extra_info['topic'] = topic
            doc_extra_info['topic_probability'] = prob.tolist()
            processed_doc = Document(
                text=doc.get_text(),
                metadata=doc.metadata,
                extra_info=doc_extra_info
            )
            processed_documents.append(processed_doc)

        return processed_documents

    def _generate_questions(self, documents: List[Document]) -> List[Document]:
        """Generate 3 questions per document using ell and enrich document metadata"""
        @simple(model="claude-3-5-sonnet-20241022", max_tokens=512)
        def generate_questions(SUMMARY: str) -> str:
            """You are a helpful assistant generating questions based on the provided SUMMARY."""
            prompt = (
                "When the page has enough information that makes sense and you can understand it, "
                "generate 3 specific questions that will definitely have answers in document content."
                "\n1. Generate 3 specific, helpful questions that you as an expert would have asked looking at SUMMARY. "
                "Seek key information to comprehend the SUMMARY fully and process it faster. "
                "\n2. Be concise and factual. NEVER invent facts and data."
                "\n3. Use Canadian English."
                "\n4. Focus on the most relevant aspects of the case."
                "\n5. Use US date format (MM/DD/YYYY) and 12h format for time."
                "\n6. Always use **bold** markdown notation for essential words and phrases of your answer."
                "\n7. Never say 'Based on the medical reports,' or anything similar. Just say the question."
                "\n8. Try not to repeat QUESTION HISTORY, always suggest more tailored questions to the user."
                "\n9. ALWAYS ask questions based on context: each question must DEFINITELY HAVE an answer on this page. "
                "Do NOT ask questions that would get back as 'The document does not mention...', "
                "'The document does not provide...'"
                "\n10. Skip questions part if page is BLANK."
                "\n\nInclude 3 questions that a person would ask and would DEFINITELY have an answer on this page:"
                "\n- Question 1"
                "\n- Question 2"
                "\n- Question 3"
                f"\n\nSUMMARY:\n{SUMMARY}"
            )
            return prompt

        processed_documents = []
        for doc in documents:
            try:
                questions = generate_questions(SUMMARY=doc.get_text())
                # Clean up the generated questions
                questions = questions.strip()
                doc_extra_info = doc.extra_info or {}
                doc_extra_info['questions'] = questions
                enriched_doc = Document(
                    text=doc.get_text(),
                    metadata=doc.metadata,
                    extra_info=doc_extra_info
                )
                processed_documents.append(enriched_doc)
                print(f"✅ Generated questions for document {doc.doc_id}")
            except Exception as e:
                print(f"⚠️ Failed to generate questions for document: {str(e)}")
                processed_documents.append(doc)
        return processed_documents

    def chat(self, query: str) -> str:
        """Process a query using the RAG system"""
        try:
            # Infer query topic
            query_topic = self._infer_query_topic(query)
            print(f"\n🔍 Analyzing query topic: {query_topic}")
            
            # Get topic description if available
            if self.topic_model:
                topic_words = self.topic_model.get_topic(query_topic)
                if topic_words:
                    print(f"📑 Related terms: {', '.join([word for word, _ in topic_words[:3]])}")

            print("🤔 Searching for relevant information...")
            response = self.query_engine.query(query)
            return response.response

        except Exception as e:
            raise RAGError(f"Query failed: {str(e)}")

    def _infer_query_topic(self, query: str) -> int:
        """Infer topic of the query using BERTopic"""
        topics, _ = self.topic_model.transform([query])
        return topics[0]

def chat_loop() -> None:
    """Interactive chat loop with improved error handling"""
    try:
        print("\n🚀 Initializing RAG Chat System...")
        rag_agent = RAGAgent()
        print("\n💡 Chat initialized! Type 'quit' to exit.")
        print("📝 You can ask questions about the documents in the input directory.")
        
        while True:
            query = input("\n👤 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print("\n🤖 Assistant: ", end='')
            response = rag_agent.chat(query)
            print(response)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except RAGError as e:
        print(f"\n❌ RAG Error: {str(e)}")
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {str(e)}")

if __name__ == "__main__":
    chat_loop()