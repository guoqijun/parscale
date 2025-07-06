import torch
from transformers import AutoModelForCausalLM, AutoConfig
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

    model_name_qwen = "./Qwen2.5-3B-Instruct/"
    model_qwen = AutoModelForCausalLM.from_pretrained(
        model_name_qwen,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    for name_parscale, param_par in model.named_parameters():
        for name_qwen, param_qwen in model_qwen.named_parameters():
            if name_parscale == name_qwen:
                param_par.data.copy_(param_qwen.data)
                print("拷贝:", name_qwen, name_parscale)

    # 打印模型参数以验证初始化
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    print(type(model))
    print(model)

    model.save_pretrained(
        save_directory=model_name,
        safe_serialization=True  # 确保使用 Safetensors 格式
    )
