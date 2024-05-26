from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


class RAGSystem:
    def __init__(self, data_dir_path = "data", db_path = "chroma") -> None:
        print("inside rag system")
        self.data_directory = data_dir_path
        self.db_path = db_path
        self.model_name = "llama3"
        self.llm_model = "llama3"
        self._setup_collection() 
        self.model = Ollama(model=self.llm_model)
        self.prompt_template = """
                                Use the following pieces of context to answer the question at the end. 
                                If you don't know the answer, just say that you are unsure.
                                Don't try to make up an answer.

                                {context}

                                Question: {question}
                                Answer:
                                """

    def _setup_collection(self):
        pages = self._load_documents()
        chunks = self._document_splitter(pages)
        chunks = self._get_chunk_ids(chunks)  
        vectordb = self._initialize_vectorDB()
        present_in_db = vectordb.get()
        ids_in_db = present_in_db["ids"]
        print(f"Number of existing ids in db: {len(ids_in_db)}")
        # add chunks to db - check if they already exist
        chunks_to_add = [i for i in chunks if i.metadata.get("chunk_id") not in ids_in_db]
        if len(chunks_to_add) > 0:
            vectordb.add_documents(chunks_to_add, ids = [i.metadata["chunk_id"] for i in chunks_to_add])
            print(f"added to db: {len(chunks_to_add)} records")
            vectordb.persist()
        else:
            print("No records to add")

    def _get_chunk_ids(self, chunks):
        ''''
        for same page number: x
            source_x_0
            source_x_1
            source_x_2
        for same source but page number: x+1
            source_x+1_0
            source_x+1_1
            source_x+1_2
        '''
        prev_page_id = None
        for i in chunks:
            src = i.metadata.get("source")
            page = i.metadata.get("page")
            curr_page_id = f"{src}_{page}"
            if curr_page_id == prev_page_id:
                curr_chunk_index += 1
            else:
                curr_chunk_index = 0
            # final id of chunk
            curr_chunk_id = f"{curr_page_id}_{curr_chunk_index}"
            prev_page_id = curr_page_id
            i.metadata["chunk_id"] = curr_chunk_id
        return chunks        
    
    def _retrieve_context_from_query(self, query_text):
        vectordb = self._initialize_vectorDB()
        context = vectordb.similarity_search_with_score(query_text, k=1)
        return context
    
    def _get_prompt(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        print(f" ***** CONTEXT ******{context} \n")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

    def answer_query(self, query_text):
        prompt = self._get_prompt(query_text)
        response_text = self.model.invoke(prompt)
        formatted_response = f"Response: {response_text}\n"
        return formatted_response

    def _load_documents(self):
        loader = PyPDFDirectoryLoader(self.data_directory)
        pages = loader.load()
        return pages
    
    def _document_splitter(self, documents):
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
        )
        return splitter.split_documents(documents)
    
    def _get_embedding_func(self):
        embeddings = OllamaEmbeddings(model=self.model_name)
        return embeddings
    
    def _initialize_vectorDB(self):
        return Chroma(
            persist_directory = self.db_path,
            embedding_function = self._get_embedding_func(),
        )

