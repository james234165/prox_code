import argparse
import os.path
import json
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import anthropic
from tqdm import tqdm
from typing import List, Dict, Sequence
from anthropic import AnthropicVertex
from datasets import load_dataset


class ClaudeTranslator:

    def __init__(self, project_id: str, region: str, model_type: str, max_retries: int = 100, sleep_sec: int = 2):
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key is None:
            self.client = AnthropicVertex(project_id=project_id, region=region)
            print(f"Create an AnthropicVertex client by\n"
                  f"- project_id={project_id}\n"
                  f"- region={region}\n"
                  f"- model_type={model_type}")
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
            print(f"Create an Anthropic client by\n"
                  f"- api_key={api_key}\n"
                  f"- model_type={model_type}")

        self.project_id = project_id
        self.region = region

        self.model_type = model_type
        self.max_retries = max_retries
        self.sleep_sec = sleep_sec

    @property
    def category(self):
        assert hasattr(self, "_category")
        return self._category

    @category.setter
    def category(self, value):
        if value == "other":
            value = "professional"
        self._category = value

    def get_completion(self, prompt: str, system_message: str) -> str:
        response = self.client.messages.create(
            model=self.model_type,
            max_tokens=4096,
            system=system_message,
            temperature=0.4,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content

    def initial_translation(self, source_lang: str, target_lang: str, source_text: str) -> str:
        system_message = f"You are a {self.category} translator specializing in accurate translation of technical and academic content from {source_lang} to {target_lang}."
        
        translation_prompt = f"""Your task is to translate assessment questions in the {self.category} field while:
1. Preserving technical accuracy and terminology
2. Ensuring cultural appropriateness for {target_lang} speakers
3. Keeping terminology consistent throughout questions and options
4. Preserving all LaTeX notation, mathematical formulas, and programming code exactly as they appear (do not translate content inside LaTeX delimiters or code blocks, including variable names, function names, and comments)
5. Preserving all currency symbols ($) exactly as they appear in the original text, without converting to local currency units
6. For units of measurement: Use the conventional translations in the target language while preserving the exact numerical values and relationships
7. Preserving any special formatting or emphasis in the original text

Please translate the following {self.category} assessment question and options:
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

Output:
Only provide the {target_lang} translation for the above text. Do not include any explanations or text apart from the translation. Different options are separated by newline characters(\n).
The number of options in the output must match the input exactly. Do not skip or combine any options.
Return the translation in the following JSON format, with keys "question" and "options", where the value of "options" is a dictionary with keys option1, option2, option3, etc. 
All JSON keys must remain in English exactly as shown and only translate the content inside square brackets:

<TRANSLATION>
{{
    "question": "[translation of question]",
    "options": {{
        "option1": "[translation of option1]",
        "option2": "[translation of option2]",
        "option3": "[translation of option3]",
        ...
    }}
}}
</TRANSLATION>"""
        
        return self.get_completion(translation_prompt, system_message)

    def reflect_on_translation(self, source_lang: str, target_lang: str, source_text: str, initial_trans: str) -> str:
        system_message = f"You are a {self.category} translation expert, specializing in translation from {source_lang} to {target_lang}."
        
        reflection_prompt = f"""Task Description:
Carefully review the source text and its translation from {source_lang} to {target_lang}, and then provide constructive suggestions in English.

Requirements:
1. Do not add, remove, or explain any information.
2. Make sure retain the original format for specialized information, e.g., anonymous information.
3. Identify any instances where proper nouns remain untranslated or where the translation contains unnecessary explanations, parenthetical original terms, or additions from {source_lang}.
4. Examine whether any technical terms, subject-specific concepts, or other specialized vocabulary have been left in {source_lang} instead of using their established {target_lang} equivalents.
5. Verify that currency symbols, mathematical operators, and measurement units remain exactly as they appear in {source_lang} text. These symbols should not be converted to their written form in {target_lang}.
6. Check that no additional symbols or written representations have been added to options where they did not exist in {source_lang} text.

Input:
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<INITIAL_TRANSLATION>
{initial_trans}
</INITIAL_TRANSLATION>

Output:
<SUGGESTIONS>
[Your suggestions here]
</SUGGESTIONS>"""
        
        return self.get_completion(reflection_prompt, system_message)

    def improve_translation(self, source_lang: str, target_lang: str, source_text: str, initial_trans: str, reflection: str) -> str:
        system_message = f"You are a {self.category} translation expert, specializing in translation from {source_lang} to {target_lang}."
        
        improvement_prompt = f"""Task Description:
Carefully review and edit the {self.category} translation from {source_lang} to {target_lang}, incorporating the expert feedback.

Requirements:
1. Do not explain any information.
2. Strictly keep the single quotes in the original text and do not add new single and double quotes.
3. Remove unnecessary explanations or original terms from {source_lang} if present in the translation.

Input:
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<INITIAL_TRANSLATION>
{initial_trans}
</INITIAL_TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Output:
Only provide the improved translation. Do not include any explanations or text apart from the translation.
Different options are separated by newline characters(\n).
The number of options in the output must match the input exactly. Do not skip or combine any options.
Return the translation in the following JSON format, with keys "question" and "options", where the value of "options" is a dictionary with keys option1, option2, option3, etc. All JSON keys must remain in English exactly as shown and only translate the content inside square brackets:

<IMPROVED_TRANSLATION>
{{
    "question": "[improved translation of question]",
    "options": {{
        "option1": "[improved translation of option1]",
        "option2": "[improved translation of option2]",
        "option3": "[improved translation of option3]",
        ...
    }}
}}
</IMPROVED_TRANSLATION>"""
        
        return self.get_completion(improvement_prompt, system_message)

    def translate_text(self, question: str, options: str, source_lang: str, target_lang: str) -> str:
        source_text = "<QUESTION>\n"+question+"\n</QUESTION>\n"+"<OPTIONS>\n"+options+"\n</OPTIONS>"

        initial_trans = None
        for attempt in range(self.max_retries):
            try:
                initial_trans = self.initial_translation(source_lang, target_lang, source_text)
            except anthropic.APITimeoutError:
                print(f"Timeout occurred during initial_translation. Retrying {attempt + 1}/{self.max_retries}......")
                time.sleep(self.sleep_sec)
            except Exception as e:
                print(f"An error occurred during initial_translation: {str(e)}. Retrying {attempt + 1}/{self.max_retries}......")
                time.sleep(self.sleep_sec)
            else:
                break
        if initial_trans is None:
            raise RuntimeError(f"Max retries reached. Unable to get a complete response. Skip initial_translation......")

        reflection = None
        for attempt in range(self.max_retries):
            try:
                reflection = self.reflect_on_translation(source_lang, target_lang, source_text, initial_trans[0].text)
            except anthropic.APITimeoutError:
                print(f"Timeout occurred during reflect_on_translation. Retrying {attempt + 1}/{self.max_retries}......")
                time.sleep(self.sleep_sec)
            except Exception as e:
                print(f"An error occurred during reflect_on_translation: {str(e)}. Retrying {attempt + 1}/{self.max_retries}......")
                time.sleep(self.sleep_sec)
            else:
                break
        if reflection is None:
            raise RuntimeError(f"Max retries reached. Unable to get a complete response. Skip reflect_on_translation......")

        trans = None
        for attempt in range(self.max_retries):
            try:
                trans = self.improve_translation(source_lang, target_lang, source_text, initial_trans[0].text, reflection[0].text)
            except anthropic.APITimeoutError:
                print(f"Timeout occurred during improve_translation. Retrying {attempt + 1}/{self.max_retries}......")
                time.sleep(self.sleep_sec)
            except Exception as e:
                print(f"An error occurred during improve_translation: {str(e)}. Retrying {attempt + 1}/{self.max_retries}......")
                time.sleep(self.sleep_sec)
            else:
                break
        if trans is None:
            raise RuntimeError(f"Max retries reached. Unable to get a complete response. Skip improve_translation......")

        return trans[0].text

    def batch_translate(self, question_list: List[str], options_list: List[str], source_lang: str, target_lang: str) -> List[dict]:
        total = len(question_list)
        return [self.translate_text(question, options, source_lang, target_lang) for question, options in tqdm(zip(question_list, options_list), total=total, desc="Translating")]


def process_chunk(ds_chunk: Sequence[Dict], translator_chunk: ClaudeTranslator, args, pbar, result_writer, writer_lock):
    chunk_failed_ids = []
    for doc in ds_chunk:
        doc_id = doc["question_id"]
        question = doc["question"]
        options = ""
        for opt_idx in range(10):
            opt = doc[f"option_{opt_idx}"]
            if opt is not None:
                options += f"option{opt_idx + 1}. {opt}\n"
        options = options.strip()

        category = doc["category"]
        translator_chunk.category = category
        print(f"current category: {translator_chunk.category}")

        try:
            translated_text = translator_chunk.translate_text(question, options, "English", args.target_lang)
        except Exception as e:
            chunk_failed_ids.append(doc_id)
            print(f"An error occurred for id={doc_id}: {str(e)}. Skip id={doc_id}......")
            continue

        with writer_lock:
            result_writer.write(
                json.dumps(
                    dict(
                        question_id=doc_id,
                        question=doc["question"],
                        options=[doc[f"option_{opt_idx}"] for opt_idx in range(10) if doc[f"option_{opt_idx}"] is not None],
                        answer=doc["answer"],
                        answer_index=doc["answer_index"],
                        cot_content=doc["cot_content"],
                        category=category,
                        src=doc["src"],
                        translated_text=translated_text,
                        question_id_src=doc["question_id_src"],
                    ),
                    ensure_ascii=False
                ) + "\n"
            )
            result_writer.flush()
            pbar.update(1)

    return chunk_failed_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="claude-3-5-sonnet-v2@20241022")
    parser.add_argument('--output_dir', type=str, default="./results/translator")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--target_lang', type=str, default="Japanese")
    parser.add_argument('--region', type=str, default=None)
    parser.add_argument('--region_ratio', type=str, default="1")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--max_retries', type=int, default=100)
    parser.add_argument('--n_thread', type=int, default=4)
    parser.add_argument('--sleep_sec', type=int, default=2)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    ds = load_dataset("TIGER-Lab/MMLU-Pro", "en", split=args.split)
    result_save_path = os.path.join(args.output_dir, "mmlu_pro", args.split, f"{args.target_lang}_{args.model_type}.jsonl")
    if args.debug:
        result_save_path = result_save_path.replace(".jsonl", "_debug.jsonl")
        args.overwrite = True
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if args.overwrite:
        print(f"Start from scratch in {result_save_path}")
    else:
        if os.path.exists(result_save_path):
            with open(result_save_path, "r") as f:
                for line in f.readlines():
                    line_dict = json.loads(line.strip())
                    finish_ids.append(line_dict["question_id"])
            print(f"Resume from {len(finish_ids)} finished samples in {result_save_path}")
        else:
            print(f"Start from scratch in {result_save_path}")

    ds = [item for item in ds if item["question_id"] not in finish_ids]
    result_writer = open(result_save_path, "w" if args.overwrite else "a")  # a for resuming
    writer_lock = Lock()

    if args.debug:
        ds = ds[:args.n_thread * 3]
    pbar = tqdm(total=len(ds), desc=f"{args.model_type} infer on {args.target_lang} for mmlu-pro")

    if args.n_thread == 1:
        assert ":" not in args.region
        translator = ClaudeTranslator(project_id=None, region=args.region, model_type=args.model_type, max_retries=args.max_retries, sleep_sec=args.sleep_sec)
        failed_ids = process_chunk(
            ds_chunk=ds, translator_chunk=translator, args=args, pbar=pbar,
            result_writer=result_writer, writer_lock=writer_lock
        )
    else:
        region_list = args.region.split(":")
        region_ratio_list = [int(reg_ratio) for reg_ratio in args.region_ratio.split(":")]
        assert len(region_list) == len(region_ratio_list)

        region_num_list = [round(reg_ratio * (args.n_thread / sum(region_ratio_list))) for reg_ratio in region_ratio_list]
        thread_region_list = []
        for reg_idx in range(len(region_num_list)):
            thread_region_list.extend([region_list[reg_idx] for _ in range(region_num_list[reg_idx])])
        thread_region_list = thread_region_list[:args.n_thread]

        if len(thread_region_list) < args.n_thread:
            thread_region_list += region_list[0] * (args.n_thread - len(thread_region_list))

        chunks = [ds[i::args.n_thread] for i in range(args.n_thread)]
        translators = [ClaudeTranslator(project_id=None, region=thread_reg, model_type=args.model_type, max_retries=args.max_retries, sleep_sec=args.sleep_sec)
                       for thread_reg in thread_region_list]
        print("Thread regions:", [trans.region for trans in translators])

        with ThreadPoolExecutor(max_workers=args.n_thread) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(
                    process_chunk,
                    chunk,
                    translator,
                    args,
                    pbar,
                    result_writer,
                    writer_lock
                ): i for i, (chunk, translator) in enumerate(zip(chunks, translators))
            }

            # Collect results as they complete
            failed_ids = []
            for future in as_completed(future_to_chunk):
                chunk_failed_ids = future.result()
                failed_ids.extend(chunk_failed_ids)

    result_writer.close()
    pbar.close()
    with open(result_save_path, "r") as f:
        result_data_list = [json.loads(line.strip()) for line in f.readlines()]

    if args.debug:
        result_json_path = result_save_path.replace(".jsonl", ".json")
        with open(result_json_path, "w") as f:
            json.dump(result_data_list, f, indent=4)
        print(f"Results have been saved to {result_json_path}")

    if len(failed_ids) > 0:
        print(
            f"There are {len(failed_ids)} failed samples in this job. "
            "Please resume by running the same command without the '--overwrite' argument."
        )

