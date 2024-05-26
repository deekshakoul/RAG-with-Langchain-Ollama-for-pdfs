import streamlit as st
from rag_system import RAGSystem

rag_system = RAGSystem()
# st.title("RAG-based AMA System")

def main():
    st.title("Ask me anything!")
    question = st.text_input("Ask your question:")
    if st.button("Ask"):
        response = rag_system.answer_query(question)
        st.write(response)

if __name__ == "__main__":
    main()
