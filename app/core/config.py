from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    UPLOAD_DIR: str
    OPENAI_API_KEY: str
    ROUTER_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
