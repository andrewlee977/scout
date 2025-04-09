from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    NEWS_API_KEY: str
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str
    # GOOGLE_APPLICATION_CREDENTIALS: str
    
    # Application Settings
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAINTENANCE_MODE: bool = True
    
    # Singleton LLM instance
    _llm: ChatOpenAI | None = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Returns singleton LLM instance"""
        if self._llm is None:
            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return self._llm

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings() 