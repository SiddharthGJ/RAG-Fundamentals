from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama



PERSIST_DIRECTORY = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

llm = ChatOllama(
    model="llama3",
    temperature=0.2
)

chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")


    if chat_history:
        rewrite_messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question "
                    "so that it is standalone and searchable. "
                    "Return ONLY the rewritten question."
                )
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        search_question = llm.invoke(rewrite_messages).content.strip()
        print(f"🔍 Searching for: {search_question}")

    else:
        search_question = user_question
        print(f"🔍 Searching for: {search_question}")


    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = "\n".join(doc.page_content.split("\n")[:2])
        print(f"  Doc {i}: {preview}...")



    combined_input = f"""
Based on the following documents, please answer this question:

Question:
{user_question}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in docs])}

Rules:
- Use ONLY the documents.
- Answer clearly and cleanly.
- Do NOT copy text verbatim.
- If the answer is not found, say:
  "I don't have enough information to answer that question based on the provided documents."
"""

    answer_messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions "
                "based only on the provided documents and conversation history."
            )
        ),
        HumanMessage(content=combined_input)
    ]


    response = llm.invoke(answer_messages)
    answer = response.content

    print("\n--- 🤖 Answer ---")
    print(answer)



    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer



def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye boss 👋")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
