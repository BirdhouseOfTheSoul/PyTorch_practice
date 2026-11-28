 #!/usr/bin/env python3

import re
from pathlib import Path


from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.prompts import PromptTemplate

import gradio as gr
from gradio.components import Textbox
import argparse


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-file", help="Path to the batch file containing queries", required=False, action="store")
    args = parser.parse_args()
    batch_file = args.batch_file

    source_path = Path(__file__).with_name("chicago-il-1.txt")
    text = source_path.read_text(encoding="utf-8", errors="ignore")

    #parse the text file using the co-pilot parser function
    sections = parse_sections(text)
    print(f"Parsed {len(sections)} sections from the municipal code.")


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

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(docs[1:1000])  # Add documents to the vector store
    print('documents loaded into vector store')

    #results = docsearch.similarity_search("Can i have a nuclear weapon in Chicago?", k=3)
    #for doc in results:
    #    print("--------------------")
    #    print(doc.page_content, doc.metadata)
    #    print("--------------------")


    agent = create_agent(
        model="google_genai:gemini-4-hydrogen",
        system_prompt="You are a helpful assistant that answers questions about the Chicago Municipal Code.",
    )

  
    template = """ 
        Answer the question using the provided context, and cite the section number and/or title number of the relevant section(s) in your answer.
        If the answer is not contained within the text below, say "I don't know."

        {query}
        
        {context}
    """

    langchain_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=template,
    )
    
    


    def qa_interface(query):

        # Perform a similarity search to find relevant documents
        results = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.metadata.get("section", "Unknown Section") + ': ' + doc.metadata.get("section_title", "Unknown Title") for doc in results])

        print(f"Query: {query}\n")
        print(f"Context sections: {context}\n")

        # Format the prompt with the query and context
        prompt = langchain_prompt.format(query=query, context=context)

        # Get the response from the LLM
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        response_text = response['messages'][-1].content[0]['text']  # Extract the text from the response
        print(response_text)
        return response_text # Return the output text from the response

    


    if batch_file:
        
        # Read queries from the batch file and process them
        with open(batch_file, 'r') as f:
            queries = f.readlines()

        for query in queries:
            query = query.strip()
            if query:
                answer = qa_interface(query)
                print(f"Answer: {answer}\n")


    else:    

        demo=gr.Interface(fn=qa_interface, 
                     inputs=Textbox(lines=2), 
                     outputs=Textbox(lines=10), 
                     title="Chicago Municipal Code Q&A", 
                     description="Ask questions about the Chicago Municipal Code.")

        demo.launch(server_name="127.0.0.1", server_port= 7860)
    






#written by copilot, not reviewed yet
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




