from models import Embedder, LLMTest
from clients import DBClient
from processors import DataProcessor



class RAG:
    def __init__(self, path: str = "./chroma_db"):
        self.db_client = DBClient(path=path)
        self.llm = LLMTest()
        self.processor = DataProcessor()
        self.embedder = Embedder()
        #self.db_client.inspect_collection()
    def run_rag(self):

        print("System is ready")
        while True:
            promt = input("Enter your promt(str or [str]) or 'q' to quit:").strip()
            if promt == 'q':
                break
            if not promt:
                continue


    
            mode = input("Run LLM with RAG?(yes/no)")

            if mode.lower() == "yes":
                search_res = self.db_client.search(query=promt, n_results=3)

                if search_res and search_res["documents"]:
                    context = "\n\n".join(search_res["documents"][0])
                    
                    rag_promt = self.llm.rag_generate(prompt=promt, context=context)
                    print(f"Promt: {promt} ")
                    ans = self.llm.generate(prompt=rag_promt)
                    print(f"Answer: {ans}")

                    for i, (doc, metadata) in enumerate(zip(search_res["documents"][0], search_res["metadatas"][0])):
                        print(f"Source {i + 1}")
                        print((doc, metadata))
                else:
                    print("No relevant documents")

            elif mode.lower() == 'no':
                promt = f"{promt}. Это вопрос. Ответ: "
                print(self.llm.generate(prompt=promt))

            else:
                print("Invalid answer. Enter yes/no.")