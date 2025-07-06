from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM
from transformers import __version__

if __name__ == "__main__":
    model_name = "./Qwen_3B"
    tokenizer_name = "./Qwen_3B"

    print(f"Transformers version: {__version__}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print(type(config))
    print(config)

    model = Qwen2ForCausalLM(config)

    # 打印模型参数以验证初始化
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    print(type(model))
    print(model)

    model.save_pretrained(
        save_directory=model_name,
        safe_serialization=True  # 确保使用 Safetensors 格式
    )
