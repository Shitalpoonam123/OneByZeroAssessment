# OneByZeroAssessment

## Objective:
Develop a Retrieval-Augmented Generation (RAG) based AI system capable of
answering questions about yourself. The system should handle inquiries in English
and manage follow-up questions effectively.

## Files added
htmlTemplates.py: A file containing HTML templates and CSS used for styling the chat interface.
.env: An environment file with placeholders for sensitive data (e.g., API keys).
README.md: Instructions on setting up and running the application, along with a brief project overview.
requirements.txt: Lists all the dependencies required to run the project.
docs: Pdf files supported.
app.py: streamlit RAG application code.

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment: python -m venv env
3. Activate the virtual environment:
4. Windows: .\env\Scripts\activate
5. MacOS/Linux: source env/bin/activate
6. Install dependencies: pip install -r requirements.txt
7. Set up environment variables by copying .env.example to .env and filling in the required information.
8. Run the Streamlit application: streamlit run src/main.py

## System Architecture
The system is designed as a conversational AI application that leverages Azure OpenAI services for processing PDF documents and responding to user queries. The architecture consists of:

* Frontend: Built using Streamlit, allowing users to upload PDFs and interact with the AI.
* Backend: Includes text extraction, chunking, and vectorization using FAISS (Facebook AI Similarity Search). The system retrieves relevant information from PDFs using Azure OpenAI's embedding and chat models.
* Memory: The system maintains conversational context using a ConversationBufferMemory to handle follow-up questions effectively.

## Data Ingestion Process
1. PDF Upload: Users can upload one or multiple PDF files via the Streamlit interface.
2. Text Extraction: Text is extracted from the uploaded PDFs using PyPDF2.
3. Text Chunking: The extracted text is split into manageable chunks using the CharacterTextSplitter to ensure effective retrieval and embedding.
4. Vectorization: The text chunks are converted into vectors using Azure OpenAI's embedding model, and stored in a FAISS vector store for efficient retrieval.
RAG Integration
5. Retrieval: The system retrieves relevant text chunks from the FAISS vector store based on the user's query.
6. Augmentation: The retrieved information is passed to the Azure OpenAI chat model to generate a contextually accurate response.
7. Generation: The final response is generated by the AI model and returned to the user, considering both the retrieved information and conversational context.


## Evaluation Methods
To evaluate the quality of the application, consider the following methods:

A) Response Accuracy: Test the AI's responses for accuracy by comparing them with known information in the PDFs.

B) Handling of Follow-up Questions: Assess the AI’s ability to maintain context across multiple interactions.

C) User Experience: Gather user feedback on the ease of use and overall experience with the system.

D) Latency: Measure the time taken for responses to ensure the system operates within acceptable performance limits.


## Demo Video

