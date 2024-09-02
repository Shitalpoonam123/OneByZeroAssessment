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
