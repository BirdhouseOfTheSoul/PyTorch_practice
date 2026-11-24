#!/usr/bin/env python3

import re
from pathlib import Path


from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate




def main() -> None:
    source_path = Path(__file__).with_name("chicago-il-1.txt")
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    sections = parse_sections(text)
    print(f"Parsed {len(sections)} sections from the municipal code.")
    #for i in range(0,10):
    #    print("--------------------")
    #    print(sections[i])
    #    print("--------------------")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(sections)

    print(f"Parsed {len(sections)} sections into {len(docs)} chunks.")



    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "mps"}, show_progress=True)

    #query_embedding = embeddings.embed_query("What is a sentence embedding?")
    #doc_embeddings = embeddings.embed_documents([doc.page_content for doc in docs[0:100]])

    docsearch = Chroma.from_documents(docs[0:1000], embeddings)  # store the embedding in docsearch using Chromadb
    print('documents loaded into ChromaDB')

    #results = docsearch.similarity_search("Can i have a nuclear weapon in Chicago?", k=3)
    #for doc in results:
    #    print("--------------------")
    #    print(doc.page_content, doc.metadata)
    #    print("--------------------")

  
    template = """ 
        Answer the question as truthfully as possible using the provided context, 
        and if the answer is not contained within the text below, say "I don't know.

        {query}
        
        {context}
    """

    langchain_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=template,
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-4-hydrogen",
        temperature=0.7,
    )

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Perform a similarity search to find relevant documents
        results = docsearch.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        # Format the prompt with the query and context
        prompt = langchain_prompt.format(query=query, context=context)

        # Get the response from the LLM
        response = llm.invoke(prompt)
        print(response.content[0]['text'])
    







#written by copilot
CHAPTER_HEADER_RE = re.compile(
    r"^CHAPTER\s+(?P<chapter>\d+(?:\.\d+)*-\d+(?:\.\d+)*)\s*$",
    re.IGNORECASE,
)
SECTION_HEADER_RE = re.compile(
    r"^\s*(?P<section>\d+(?:\.\d+)*-\d+(?:\.\d+)*-\d{3,}(?:\.\d+)*)"
    r"[ \t\u00a0]+(?P<section_title>\S.*)$"
)


#written by copilot
def parse_sections(text: str) -> list[Document]: 
    lines = text.splitlines()
    chapters = [
        (index, match.group("chapter"))
        for index, line in enumerate(lines)
        if (match := CHAPTER_HEADER_RE.match(line.strip())) is not None
    ]

    documents = []
    for chapter_index, (chapter_start, chapter_number) in enumerate(chapters):
        chapter_end = (
            chapters[chapter_index + 1][0]
            if chapter_index + 1 < len(chapters)
            else len(lines)
        )
        headings = []
        for line_index in range(chapter_start + 1, chapter_end):
            match = SECTION_HEADER_RE.match(lines[line_index])
            if (
                match is not None
                and match.group("section").rsplit("-", 1)[0] == chapter_number
                and match.group("section_title")[0].isupper()
            ):
                headings.append((line_index, match))

        if not headings:
            continue

        first_section = headings[0][1].group("section")
        body_start = next(
            (
                index
                for index, match in headings[1:]
                if match.group("section") == first_section
            ),
            None,
        )
        if body_start is None:
            continue

        section_headings = [
            (index, match) for index, match in headings if index >= body_start
        ]
        for heading_index, (start, match) in enumerate(section_headings):
            end = (
                section_headings[heading_index + 1][0]
                if heading_index + 1 < len(section_headings)
                else chapter_end
            )
            section_number = match.group("section")
            title_number, chapter_part, *_ = section_number.split("-")
            page_content = "\n".join(lines[start:end]).strip()
            if page_content:
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "section": section_number,
                            "section_title": match.group("section_title").strip(),
                            "chapter": f"{title_number}-{chapter_part}",
                            "title": title_number,
                            "source_line": start + 1,
                        },
                    )
                )

    if not documents:
        raise ValueError("No municipal code sections were found in the input file.")
    return documents



if __name__ == "__main__":
    main()




