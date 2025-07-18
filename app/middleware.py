"""Middleware for request logging and processing."""
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    """Middleware to log basic request details."""
    method = request.method
    path = request.url.path
    
    logger.debug(f"Request: {method} {path}")
    
    response = await call_next(request)
    
    return response