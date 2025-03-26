from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    NEWS_API_KEY: str
    OPENAI_API_KEY: str
    # TTS_API_KEY: str
    GOOGLE_APPLICATION_CREDENTIALS: str
    
    # Application Settings
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings() 