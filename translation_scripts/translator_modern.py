import json
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

from prompt import user_message, sys_message

load_dotenv()


class TranslationWorker:
    def __init__(self, api_key, base_url, model, system_prompt, output_folder):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.output_folder = output_folder

    def translate_document(self, document):
        text_data = document["content"]
        text_title = document["title"]
        text_id = document["id"]

        user_input = user_message.format(
            text_title=text_title,
            text_data=text_data
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input},
                ],
                stream=True,
                temperature=0.35,
                top_p=1.0
            )

            complete_response = ""
            for chunk in response:
                if chunk.choices[0].finish_reason is not None:
                    break
                delta = chunk.choices[0].delta.content
                if delta:
                    complete_response += delta

            self.save_files(text_id, text_title, text_data, complete_response)

            return {
                "id": text_id,
                "input": user_input,
                "output": complete_response
            }

        except Exception as e:
            print(f"Error processing {text_id}: {e}")
            return None

    def save_files(self, text_id, text_title, text_data, translation):
        os.makedirs(self.output_folder, exist_ok=True)

        original_path = os.path.join(self.output_folder, f"{text_id}.md")
        translated_path = os.path.join(self.output_folder, f"{text_id}_translated.md")

        with open(original_path, "w", encoding="utf-8") as f:
            f.write(f"# {text_title}\n\n{text_data}\n")

        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(translation)


class ContinuousTranslator:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/gpt-4.1-mini"
        self.output_folder = "./output_new42/"
        self.docs_json = "docs.json"
        self.output_json = "output_new42.json"
        self.system_prompt = sys_message
        self.document_queue = queue.Queue()
        self.results_lock = threading.Lock()
        self.completed_count = 0
        self.total_count = 0
        self.shutdown_event = threading.Event()

    def load_documents(self):
        with open(self.docs_json, "r", encoding="utf-8") as f:
            documents = json.load(f)

        completed = []
        if os.path.exists(self.output_json):
            with open(self.output_json, "r", encoding="utf-8") as f:
                completed = json.load(f)

        completed_ids = {entry["id"] for entry in completed}
        pending = [doc for doc in documents if doc["id"] not in completed_ids]

        return pending, completed

    def save_result(self, result, all_results):
        if result:
            with self.results_lock:
                all_results.append(result)
                self.completed_count += 1
                print(f"Completed {result['id']} - Progress: {self.completed_count}/{self.total_count}")

                # Atomic file write
                temp_path = "output_tmp.json"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, self.output_json)

    def worker_thread(self, worker, all_results):
        """Worker thread that continuously processes documents from the queue"""
        while not self.shutdown_event.is_set():
            try:
                # Get document with timeout to check shutdown periodically
                document = self.document_queue.get(timeout=1)

                if document is None:  # Poison pill
                    break

                result = worker.translate_document(document)
                self.save_result(result, all_results)
                self.document_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                self.document_queue.task_done()

    def translate_continuous(self):
        pending_documents, existing_results = self.load_documents()

        if not pending_documents:
            print("No pending documents to translate.")
            return existing_results

        self.total_count = len(pending_documents)
        print(f"Starting translation of {self.total_count} documents with {self.max_workers} workers")

        all_results = existing_results.copy()

        # Add all pending documents to queue
        for doc in pending_documents:
            self.document_queue.put(doc)

        # Create workers and start worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            workers = [
                TranslationWorker(
                    self.api_key,
                    self.base_url,
                    self.model,
                    self.system_prompt,
                    self.output_folder
                ) for _ in range(self.max_workers)
            ]

            # Start worker threads
            futures = []
            for worker in workers:
                future = executor.submit(self.worker_thread, worker, all_results)
                futures.append(future)

            # Wait for all documents to be processed
            self.document_queue.join()

            # Send poison pills to stop workers
            for _ in range(self.max_workers):
                self.document_queue.put(None)

            # Wait for workers to finish
            for future in futures:
                future.result()

        print(f"Translation completed. Total documents processed: {len(all_results)}")
        return all_results

    def shutdown(self):
        """Gracefully shutdown the translator"""
        self.shutdown_event.set()


def main():
    translator = ContinuousTranslator(max_workers=20)

    try:
        translator.translate_continuous()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        translator.shutdown()


if __name__ == "__main__":
    main()