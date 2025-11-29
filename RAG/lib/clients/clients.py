from typing import TypedDict


import chromadb


from processors.processor import DataProcessor
from models.embedder import Embedder

class DBClient:
    def __init__(self, path: str = "./chroma_db", collection_name: str = "rf_constitution"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=path)
        self.processor = DataProcessor()
        self.embedder_obj = Embedder()
        try:
            
            
            self.collection = self.client.get_collection(name=self.collection_name, embedding_function=Embedder())
            
            
            print(f"Collection {self.collection_name} has been found") # change to logger
           
        except (chromadb.errors.NotFoundError, ValueError):
            self.collection = self.client.create_collection(name=self.collection_name,
                                                            embedding_function=Embedder())
            self.extract_data()
            
            print(f"Collection {self.collection_name} has not been found") # change to logger
            print(f"Collection {self.collection_name} has been created") # change to logger

    def embedder(self, input):
        return self.embedder_obj.encode(texts=input)
    
    def load_data(self, path):
        try:
            with open(path, "r", encoding="UTF-8") as f:
                text = f.read()

        except (FileExistsError, FileNotFoundError) as e:
            print(f"Problem with loading a data: {e}") # change to logger
            raise
        chunks = self.processor.process(text=text) # change to logger
        print(f"Loaded {len(chunks)} chunks.")

        return chunks

    def extract_data(self):
        chunks = self.load_data("data/raw/text.txt")
        
        if not chunks:
            print(f"No chunks to load") 

        documents = []
        metadatas = []
        ids = []

        for i, document in enumerate(chunks):
            documents.append(document["text"])
            metadatas.append(document["metadata"])

            doc_part = document["metadata"]["document_part"]
            
            if doc_part == "Preamble":
                ids.append(f"preamble")

            elif doc_part == "Section":
                section_num = document["metadata"]["section"]
                ids.append(f"section_{section_num}")

            elif doc_part == "Chapter":
                chapter_num = document["metadata"]["chapter"]
                ids.append(f"chapter_{chapter_num}")
            
            elif doc_part == "Article":
                article_num = document["metadata"]["article"]
                ids.append(f"article_{article_num}")
            
            else:
                ids.append(i)
            
        try:


            batch_size = 50  # Adjust this based on your GPU memory
            total_added = 0
        
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                
                batch_documents = documents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                self.collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                total_added += len(batch_documents)
                print(f"Added batch {i//batch_size + 1}: {len(batch_documents)} documents (total: {total_added})")
            # self.collection.add(
            #     documents=documents,
            #     metadatas=metadatas,
            #     ids=ids
            # )

            print(f"Successfully added {len(chunks)} in database.") 
            
        
        except Exception:
            print(f"Something went wrong during adding chunks in database")# change to logger
            raise
        
    def search(self, query: str, n_results: int, where: dict = None) -> chromadb.QueryResult | None:

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        
        except Exception as e:
            print(f"Something went wrong during database search {e}") # change to logger
            return None

    def inspect_collection(self):
        """Debug method to see what's in the collection"""
        try:
            # Get a sample of documents
            sample = self.collection.peek(limit=10)
            print("Sample documents in collection:")
            for i, (doc, metadata, doc_id) in enumerate(zip(sample['documents'], sample['metadatas'], sample['ids'])):
                print(f"ID: {doc_id}")
                print(f"Metadata: {metadata}")
                print(f"Text preview: {doc[:100]}...")
                print("-" * 50)
            
            # Count total documents
            count = self.collection.count()
            print(f"Total documents in collection: {count}")
            
        except Exception as e:
            print(f"Error inspecting collection: {e}")


#Отдельные сервисы