# Qwen 特殊 Token 添加脚本

快速为 Qwen/Qwen2.5-VL-3B-Instruct (或兼容模型) 添加新的特殊 token，并保存成可直接加载的本地目录。

## 运行

```bash

fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id playground/Pretrained_models/Qwen2.5-VL-3B-Instruct \
  --tokens-file ${fast_token_list} \
  --save-dir playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --init-strategy normal
  
```

`tokens.txt` 示例：
```
<loc_x>
<loc_y>
<bbox_start>
<bbox_end>
```

## 参数
- --model-id: HF Hub HF 或 已存在的本地模型目录
- --save-dir: 输出目录
- --tokens-file
- --init-strategy: avg / normal / zero
- --as-special / --no-as-special: 作为 special token 还是普通 token
- --padding-side: left / right
- --device: cpu / cuda / mps / auto

## 结果
保存目录包含：
- config.json / model.safetensors / tokenizer.*
- added_token_id_map.json (记录新增 token 到 id 的映射)

## 加载
```python
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
tok = AutoTokenizer.from_pretrained("./qwen_vl_with_spatial", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./qwen_vl_with_spatial", torch_dtype="auto", trust_remote_code=True)
print(tok.convert_tokens_to_ids("<loc_x>"))
```
