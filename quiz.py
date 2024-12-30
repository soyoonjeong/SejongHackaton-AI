import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/home/llm_models"


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


def inference(model, subject):
    # STEP 1: 생성 옵션 설정
    sampling_params = SamplingParams(
        seed=42,
        temperature=1,
        top_p=1,
        top_k=50,
        max_tokens=1024,  # target에 대한 로그 확률만 계산할 것이기 때문에 생성은 필요 없음
        prompt_logprobs=1,  # 프롬프트의 로그 확률을 계산하기 위한 옵션
    )

    output = model.generate(
        prompts=subject
        + "에 대한 오지선다 문제를 제작해줘. {'질문':'','정답':'', '해설':''} 형태로 출력해줘.",  # target에 대한 로그 확률을 계산할 것이기 때문에 context + target 전달
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print(output[0].outputs[0])
    text = output[0].outputs[0].text


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B"
    model, _ = load_model(model_name, 6000)

    text = inference(model, "네트워크 코어")
    print(text)
