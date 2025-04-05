"""
Utility functions for LLM operations with centralized logging and error handling.
"""

import logging
import datetime
from langchain_core.messages import AIMessage
from app.config import settings

logger = logging.getLogger(__name__)

def invoke_llm(llm, messages, function_name="unknown", **kwargs):
    """
    Wrapper function for LLM calls with centralized logging and error handling.
    
    Args:
        llm: The LLM instance to use
        messages: List of messages to send to the LLM
        function_name: Name of the calling function for logging
        **kwargs: Additional arguments to pass to the LLM invoke method
        
    Returns:
        The LLM response or a default error response
    """
    try:
        logger.info(f"LLM call from {function_name} with {len(messages)} messages")
        start_time = datetime.datetime.now()
        
        response = llm.invoke(messages, **kwargs)
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"LLM call from {function_name} completed in {duration:.2f} seconds")
        
        return response
    except Exception as e:
        logger.error(f"Error in LLM call from {function_name}: {str(e)}", exc_info=True)
        # Return a default error message that can be handled by the calling function
        return AIMessage(content=f"Error in LLM processing: {str(e)}")
