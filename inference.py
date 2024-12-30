import os
import re
import json
import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

import title


MODEL_PATH = "/home/llm_models"


def preprocess_text_for_llm(input_text):
    """
    Preprocesses text data for LLM by simplifying and structuring it.

    Args:
        input_text (str): Raw input text.

    Returns:
        str: Preprocessed text suitable for LLM input.
    """
    # Step 1: Remove unnecessary line breaks
    text = re.sub(r"\n+", " ", input_text)

    # Step 2: Simplify formatting (remove redundant symbols and special cases)
    text = re.sub(r"\*\*|`|\"|\(.*?\)|—>", "", text)

    # Step 3: Structure the text
    sections = []
    current_section = ""

    for line in text.split(". "):
        line = line.strip()
        if line.startswith("###"):
            # Append previous section if exists
            if current_section:
                sections.append(current_section.strip())
            # Start a new section
            current_section = f"\n{line.strip('#').strip()}\n"
        else:
            current_section += f"{line.strip()} "

    # Append the last section
    if current_section:
        sections.append(current_section.strip())

    # Step 4: Final formatting of structured sections
    processed_text = "\n".join(sections).strip()

    # Step 5: Add bullet points and reformat lists for LLM readability
    processed_text = re.sub(r"- ", "\u2022 ", processed_text)

    return processed_text


# 테스트 데이터 로드
def load_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    return data


def load_model(model_name: str, max_model_len: int):
    model_path = os.path.join(MODEL_PATH, model_name)
    engine_args = {
        "model": model_path,
        "max_model_len": max_model_len,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.8,
    }
    model = LLM(**engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def get_dictionary(tokenizer, targets):
    """target(대상) 텍스트에 대해서 토큰 길이 사전 생성"""
    # 토큰 사전에 등록
    token_dictionary = {
        target: tokenizer.encode(target) for target in list(set(targets))
    }
    # 모든 토큰 사전 요소들에 대해 첫 요소가 모두 같을 경우 첫 요소 제거
    first_elements = [tokens[0] for tokens in token_dictionary.values()]
    if len(set(first_elements)) == 1:
        len_dictionary = {
            target: len(tokens) - 1 for target, tokens in token_dictionary.items()
        }
    else:
        len_dictionary = {
            target: len(tokens) for target, tokens in token_dictionary.items()
        }

    return len_dictionary


def inference(subject, model, tokenizer, prompt):
    # STEP 1: 생성 옵션 설정
    sampling_params = SamplingParams(
        seed=42,
        temperature=1,
        top_p=1,
        top_k=50,
        max_tokens=1,  # target에 대한 로그 확률만 계산할 것이기 때문에 생성은 필요 없음
        prompt_logprobs=1,  # 프롬프트의 로그 확률을 계산하기 위한 옵션
    )

    # STEP 2: target 토큰화해서 토큰 길이 사전 생성 (e.g. {"대한민국": 2, "한국": 1})
    targets = title.TITLE[subject]
    dictionary = get_dictionary(tokenizer, targets)

    # STEP 3: 필요한 자료구조 초기화 (e.g. 추론 결과 저장 딕셔너리)
    response_dict = {"subject": subject, "input": prompt}

    # STEP 4: 추론 수행
    try:
        for target in targets:
            # STEP 4-1. 추론
            output = model.generate(
                prompts=f"학습 내용:  {prompt}\n 주제: {target}",  # target에 대한 로그 확률을 계산할 것이기 때문에 context + target 전달
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            prompt_logprobs = output[0].prompt_logprobs[
                -dictionary[target] :
            ]  # target 토큰 길이만큼 자르기

            # STEP 4-2. target에 해당하는 토큰의 로그 확률 합 계산
            logprob = 0
            for logprob_dict in prompt_logprobs:
                # 하나의 prompt_logprobs의 element(dictionary)에 두 개의 item이 있을 경우, 하나는 prompt에 이미 존재하는 token, 하나는 가장 logprob가 높은 토큰
                # e.g., {239012: Logprob(logprob=-inf, rank=3, decoded_token='같'), 236039: Logprob(logprob=-0.2840934097766876, rank=1, decoded_token='다')}
                min_prob = min(logprob_dict.items(), key=lambda x: x[1].logprob)
                logprob += float(min_prob[1].logprob)

            response_dict[target] = logprob

    except Exception as e:
        pass

    return response_dict


def chunk_sentences_with_length_limit(text, max_length):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks[0]) == 0:
        chunks.pop(0)
    return chunks


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    # 모델 로드
    model_name = "Qwen/Qwen2.5-7B"
    model, tokenizer = load_model(model_name, 6000)

    # 테스트 데이터 로드
    test_data = load_data("raw_data.txt")
    # test_data = preprocess_text_for_llm(test_data)
    test_dataset = chunk_sentences_with_length_limit(test_data, len(test_data) / 5)
    print(len(test_data) / 5)
    total_result = {}
    idx = 0
    for data in tqdm(test_dataset):
        # 추론 수행
        result = inference("네트워크", model, tokenizer, data)
        total_result[idx] = result
        idx += 1

    # for name in list(total_result[0].keys())[2:]:
    #     print(name)
    #     total_result["avg"] = np.mean(
    #         [result[name] for result in total_result.values()]
    #     )

    with open(f"{model_name.split('/')[-1]}.json", "w", encoding="utf-8") as file:
        json.dump(total_result, file, ensure_ascii=False, indent=4)
