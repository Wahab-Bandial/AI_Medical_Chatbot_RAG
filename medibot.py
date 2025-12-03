import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Do not try to make up an answer.
        Do not provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # --- CLEAN RESPONSE WITH FORMATTED SOURCES ---
            clean_result = result.strip()

            # Extract unique, clean source references
            seen = set()
            sources = []
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "Unknown page")
                filename = os.path.basename(source)
                label = f"{filename} (p. {page})"
                if label not in seen:
                    seen.add(label)
                    sources.append(label)

            # Append sources only if available
            if sources:
                clean_result += "\n\n**Sources:**\n- " + "\n- ".join(sources)

            result_to_show = clean_result

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()