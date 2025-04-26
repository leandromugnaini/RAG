from pydantic import BaseModel, Field


class DocumentsResponse(BaseModel):
    message: str = Field(example="Documents processed successfully")
    documents_indexed: int
    total_chunks: int
