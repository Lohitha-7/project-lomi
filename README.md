# Project Lomi

A full-stack AI career companion project with a FastAPI backend and a frontend interface.

## Project structure

- backend/ - FastAPI backend service
- frontend/ - static frontend files
- .gitignore - ignores local environment and cache files

## Tech stack

- Python
- FastAPI
- Groq API
- HTML / frontend assets

## Setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment if needed.
3. Install backend dependencies.
4. Add your Groq API key to a local `.env` file in the backend directory.

Example:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

## Run backend

```bash
cd backend
python -m uvicorn app.main:app --reload
```

## Run frontend

Open the HTML file in the frontend folder in a browser or serve it using a simple local static server.

## Notes

- Never commit your `.env` file.
- Keep generated files such as `__pycache__` and local virtual environments ignored.

## Repository status

This project is configured for GitHub and already connected to the remote repository.
