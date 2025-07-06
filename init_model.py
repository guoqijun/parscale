from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import __version__
from ParScale_Qwen_3B_P2.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM

if __name__ == "__main__":
    model_name = "./ParScale_Qwen_3B_P2"
    tokenizer_name = "./ParScale_Qwen_3B_P2"

    print(f"Transformers version: {__version__}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print(type(config))
    print(config)

    model = Qwen2ParScaleForCausalLM(config)

    # 打印模型参数以验证初始化
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    print(type(model))
    print(model)

    model.save_pretrained(
        save_directory=model_name,
        safe_serialization=True  # 确保使用 Safetensors 格式
    )
