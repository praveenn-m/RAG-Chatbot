from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document  
  
texts = [  
    Document(page_content='The quarterly revenue for Acme Corp was .2 million in 2024.', metadata={'source':'doc1'}),  
    Document(page_content='The company policy states that approvals must be signed by the finance lead.', metadata={'source':'doc2'}),  
]  
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  
splits = text_splitter.split_documents(texts)  
vstore = Chroma.from_documents(documents=splits, embedding=embeddings)  
results = vstore.similarity_search('What was Acme Corp revenue in 2024?', k=2)  
print('RESULTS', len(results))  
for r in results:  
    print(r.page_content)  
print('done')  
