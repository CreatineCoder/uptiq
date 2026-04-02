from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Handles splitting full documents into smaller chunks for vector database indexing.
    """
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        # We use character length as a proxy for tokens (approx 4 chars per token).
        # So chunk_size=2000 ~ 500 tokens. chunk_overlap=200 ~ 50 tokens.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, text: str) -> list[str]:
        """Split a single text document into a list of string chunks."""
        if not text or text.strip() == "":
            return []
        return self.splitter.split_text(text)
