import os
import asyncio
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
import openai
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )
    
openai.api_key = OPENAI_API_KEY

PERSIST_DIR = "./query-engine-storage"
DATA_DIR = "./data"

async def load_llama_index() -> VectorStoreIndex:
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")

    if not os.path.exists(PERSIST_DIR):
        print("📂 Persist directory not found. Creating a new index...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"✅ Index created and persisted to '{PERSIST_DIR}'.")
    else:
        print(f"📂 Loading existing index from '{PERSIST_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("✅ Index loaded successfully.")

    return index

def get_chat_completion_sync(query_res: str, user_query: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": query_res},
                {"role": "user", "content": user_query},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❗ An error occurred while contacting OpenAI: {str(e)}"

async def get_chat_completion(query_res: str, user_query: str, executor: ThreadPoolExecutor) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_chat_completion_sync, query_res, user_query)

async def query_info(index: VectorStoreIndex, query: str) -> str:
    try:
        query_engine = index.as_query_engine(use_async=True)
        res = await query_engine.aquery(query)
        print("📄 Query result:", res)
        return str(res)
    except Exception as e:
        print(f"❗ An error occurred during querying: {str(e)}")
        return f"An error occurred while querying the index: {str(e)}"

async def handle_query(index: VectorStoreIndex, user_query: str, executor: ThreadPoolExecutor) -> str:
    index_response = await query_info(index, user_query)

    if not index_response:
        return "❗ I couldn't retrieve any information from the index."

    chat_response = await get_chat_completion(index_response, user_query, executor)
    return chat_response

async def main():
    index = await load_llama_index()
    executor = ThreadPoolExecutor(max_workers=5)

    print("\n✅ The system is ready. You can start asking questions.")
    print("Type 'exit' or 'quit' to end the session.\n")

    try:
        while True:
            user_input = input("🧑 You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("❗ Please enter a valid query.")
                continue

            response = await handle_query(index, user_input, executor)
            print(f"🤖 ChatGPT: {response}\n")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        executor.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())
