# SatQuery AI — GeoChat BigEarthNet LoRA Adapter
**SIH26167: Multimodal Agentic Remote Sensing Platform (ISRO / SAC)**

## Overview
This directory contains the Parameter-Efficient Fine-Tuning (PEFT) LoRA adapter weights for **GeoChat-7B**, adapted to multi-sensor remote sensing imagery using **BigEarthNet-v2**.

## Training Configuration
- **Base Architecture**: `MBZUAI/GeoChat-7B` (LLaVA-derived remote sensing VLM)
- **Adaptation Method**: QLoRA (NF4 4-bit NormalFloat quantization)
- **Target Modules**: Query, Key, Value, and Output projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA Parameters**:
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Dropout: 0.05
  - Trainable Parameters: ~18.4M (0.26% of total 7B parameter space)
- **Dataset**: BigEarthNet-v2 (Co-registered Sentinel-1 SAR VV/VH + Sentinel-2 MSI 12-band patches with CORINE land-cover text annotations)
- **Objective**: Multisensor cross-modal alignment and remote sensing Visual Question Answering (RSVQA).

## SIH Problem Statement Compliance
Satisfies the mandatory requirement:
> *"At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet.txt or any open source training data."*

Run inference with:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("MBZUAI/GeoChat", load_in_4bit=True)
model = PeftModel.from_pretrained(model, "models/geochat_lora_bigearthnet")
```
