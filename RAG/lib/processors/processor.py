import re
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self):
        pass
    
    def process(self, text: str) -> List[Dict[str, Any]]:
        # НЕ чистим текст здесь - это ломает разбивку!
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # parts: [pre, "Статья 1 ...", body1, "Статья 2 ...", body2, ...]
        parts = re.split(r"\n(Статья\s+\d+[^\n]*)\n", text)
        chunks = []

        # преамбула
        preamble = parts[0].strip()
        if preamble:
            chunks.append({
                "text": preamble,
                "metadata": {
                    "document": "RF Constitution",
                    "document_part": "Preamble",
                    "section": "",
                    "chapter": "", 
                    "article": ""
                }
            })

        # пары (заголовок, тело)
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
            title = parts[i].strip()
            body = parts[i + 1].strip()
            if not title and not body:
                continue
            
            # Извлекаем номер статьи
            article_match = re.search(r'Статья\s+(\d+(?:\.\d+)?)', title)
            article_num = article_match.group(1) if article_match else ""
            
            chunk_text = (title + "\n\n" + body).strip()
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document": "RF Constitution",
                    "document_part": "Article",
                    "section": "",
                    "chapter": "",
                    "article": article_num
                }
            })
            with open("data/processed/chunks.txt", 'w') as f:
                f.write(str(chunks))
        return chunks

    def clean_data(self, text: str) -> str:
        """Очистка текста - вызывается ДО process в DBClient"""
        # Здесь чистим, но не ломаем структуру
        text = re.sub(r'\n+', '\n', text)  # Заменяем множественные переносы на одинарные
        text = re.sub(r'[ \t]+', ' ', text)  # Заменяем множественные пробелы/табы на один
        return text.strip()