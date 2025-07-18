"""Main server application."""
import sys
import uvicorn
from fastapi import FastAPI, Request

from app.config import Config
from app.logging_config import setup_logging
from app.middleware import log_requests
from app.routes import create_message, count_tokens, root
from app.models import MessagesRequest, TokenCountRequest

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI()

# Add middleware
app.middleware("http")(log_requests)

# Add routes
app.post("/v1/messages")(create_message)
app.post("/v1/messages/count_tokens")(count_tokens)
app.get("/")(root)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level=Config.LOG_LEVEL)