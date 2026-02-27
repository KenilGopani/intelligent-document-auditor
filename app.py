import streamlit as st
import time
from rag_engines.vector_engine import VectorRAGEngine
from rag_engines.tree_engine import TreeRAGEngine
from utils.document_processor import DocumentProcessor
from utils.performance_tracker import PerformanceTracker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'vector_engine' not in st.session_state:
    st.session_state.vector_engine = None
if 'tree_engine' not in st.session_state:
    st.session_state.tree_engine = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""

def main():
    st.set_page_config(
        page_title="Intelligent Document Auditor",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Intelligent Document Auditor")
    st.markdown("""
    Compare Traditional Vector-based RAG vs Vectorless Tree-based RAG using local Ollama models.
    Upload a PDF document and see how each approach retrieves and processes information differently.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Ollama model selection
        model_name = st.text_input(
            "Ollama Model Name",
            value="llama3.1",
            help="Enter the Ollama model to use for processing"
        )
        
        # PDF uploader
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            # Validate PDF
            is_valid, error_msg = DocumentProcessor.validate_pdf(uploaded_file)
            if not is_valid:
                st.error(f"❌ Invalid PDF: {error_msg}")
                return
            
            # Process document
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Extract text
                        text_content = DocumentProcessor.extract_text_from_pdf(uploaded_file)
                        st.session_state.document_text = text_content
                        
                        # Initialize engines
                        st.session_state.vector_engine = VectorRAGEngine(model_name=model_name)
                        st.session_state.tree_engine = TreeRAGEngine(model_name=model_name)
                        
                        # Build indices
                        documents = [text_content]
                        
                        # Build Vector index
                        with st.spinner("Building Vector Index..."):
                            vector_build_time = st.session_state.vector_engine.build_index(documents)
                            st.success(f"✅ Vector Index built in {vector_build_time:.2f}s")
                        
                        # Build Tree index  
                        with st.spinner("Building Tree Index..."):
                            tree_build_time = st.session_state.tree_engine.build_index(documents)
                            st.success(f"✅ Tree Index built in {tree_build_time:.2f}s")
                        
                        st.session_state.document_processed = True
                        st.success("🎉 Document processing complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}")
        
        # Display performance stats if processed
        if st.session_state.document_processed:
            st.divider()
            st.subheader("Performance Stats")
            
            vector_time = st.session_state.vector_engine.get_build_time()
            tree_time = st.session_state.tree_engine.get_build_time()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vector Index Build Time", f"{vector_time:.2f}s")
            with col2:
                st.metric("Tree Index Build Time", f"{tree_time:.2f}s")
            
            # Show tree structure info
            tree_info = st.session_state.tree_engine.get_tree_structure_info()
            if "num_nodes" in tree_info:
                st.info(f"Tree Nodes: {tree_info['num_nodes']}")

    # Main content area
    if not st.session_state.document_processed:
        st.info("👈 Please upload a PDF document and click 'Process Document' to get started")
        return
    
    # Query interface
    st.divider()
    st.subheader("Document Analysis")
    
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'Summarize the key risks in Section 4'",
        key="query_input"
    )
    
    # Trace toggle
    show_trace = st.toggle("Show Source Traces", value=False)
    
    if query and st.button("Analyze Document", type="primary"):
        if not st.session_state.vector_engine.is_initialized() or not st.session_state.tree_engine.is_initialized():
            st.error("Engines not properly initialized")
            return
        
        with st.spinner("Analyzing with both engines..."):
            try:
                # Track performance
                perf_tracker = PerformanceTracker()
                
                # Query Vector engine
                with perf_tracker.track_time("vector_query"):
                    vector_result = st.session_state.vector_engine.query(query)
                
                # Query Tree engine
                with perf_tracker.track_time("tree_query"):
                    tree_result = st.session_state.tree_engine.query(query)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                # Vector Engine Results
                with col1:
                    st.subheader("📊 Vector-Based RAG")
                    st.caption(f"Response Time: {perf_tracker.format_time(vector_result.execution_time)}")
                    
                    st.markdown("**Response:**")
                    st.write(vector_result.response)
                    
                    if show_trace:
                        st.markdown("**Retrieved Context:**")
                        with st.expander("View Sources"):
                            st.text_area(
                                "Source Content",
                                value=vector_result.get_source_text(),
                                height=200,
                                key="vector_sources"
                            )
                
                # Tree Engine Results
                with col2:
                    st.subheader("🌳 Tree-Based RAG")
                    st.caption(f"Response Time: {perf_tracker.format_time(tree_result.execution_time)}")
                    
                    st.markdown("**Response:**")
                    st.write(tree_result.response)
                    
                    if show_trace:
                        st.markdown("**Retrieved Context:**")
                        with st.expander("View Sources"):
                            st.text_area(
                                "Source Content",
                                value=tree_result.get_source_text(),
                                height=200,
                                key="tree_sources"
                            )
                
                # Performance comparison
                st.divider()
                st.subheader("⚡ Performance Comparison")
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.metric(
                        "Vector Response Time", 
                        perf_tracker.format_time(vector_result.execution_time)
                    )
                
                with perf_col2:
                    st.metric(
                        "Tree Response Time", 
                        perf_tracker.format_time(tree_result.execution_time)
                    )
                
                with perf_col3:
                    time_diff = abs(vector_result.execution_time - tree_result.execution_time)
                    faster_engine = "Vector" if vector_result.execution_time < tree_result.execution_time else "Tree"
                    st.metric(
                        "Time Difference",
                        perf_tracker.format_time(time_diff),
                        f"{faster_engine} is faster"
                    )
                
                # Key differences highlight
                st.info("""
                **Key Differences:**
                - **Vector RAG**: Uses embedding similarity to find relevant chunks (faster, but may miss context)
                - **Tree RAG**: Uses LLM to understand document structure and hierarchy (slower, but more contextual)
                """)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()