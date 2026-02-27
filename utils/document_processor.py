import time
from typing import List, Tuple
from io import BytesIO
from pypdf import PdfReader


class DocumentProcessor:
    """Utility class for processing PDF documents"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """
        Extract text content from a PDF file
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            str: Extracted text content
        """
        try:
            # Handle both file path and uploaded file objects
            if hasattr(pdf_file, 'read'):
                pdf_reader = PdfReader(BytesIO(pdf_file.read()))
            else:
                pdf_reader = PdfReader(pdf_file)
            
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks of specified size
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum tokens per chunk (approximate)
            
        Returns:
            List[str]: List of text chunks
        """
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add period back if it was removed by split
            sentence = sentence + ". " if sentence else ""
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > chunk_size * 4:  # Approximate token to char ratio
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    @staticmethod
    def validate_pdf(pdf_file) -> Tuple[bool, str]:
        """
        Validate PDF file
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if hasattr(pdf_file, 'read'):
                pdf_reader = PdfReader(BytesIO(pdf_file.read()))
                pdf_file.seek(0)  # Reset file pointer
            else:
                pdf_reader = PdfReader(pdf_file)
            
            if len(pdf_reader.pages) == 0:
                return False, "PDF file appears to be empty"
                
            return True, ""
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"