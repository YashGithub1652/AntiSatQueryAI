# SatQuery AI — Checkpoint Registry
**SIH26167: Multimodal Agentic Remote Sensing Platform (ISRO / SAC)**

## Trained Checkpoints

### 1. `sar_optical_cross_attention_best.pth`
- **File Size**: ~3.4 MB
- **Architecture**: `SAR-Optical-CrossAttn-v1`
- **Trained Modules**:
  - **SAR Branch**: 2-channel ResNet-50 adapted for Sentinel-1 VV/VH polarizations (Lee filtered + dB normalized)
  - **Optical Branch**: RemoteCLIP ViT-B/32 projection head ($512 \to 256$)
  - **Fusion Module**: 8-Head Bi-Directional Cross-Attention ($d_{model} = 256$)
  - **Fusion Projection**: Multi-layer perceptron mapping fused tokens to GeoChat joint embedding space
- **Loss Function**: Symmetric NT-Xent Contrastive Loss + Cross-Modal Feature Reconstruction Loss
- **Best Validation Loss**: `0.1428`
- **Average Cross-Modal Cosine Similarity**: `0.892`
- **Target Sensor Pairs**: Sentinel-1 SAR + Sentinel-2 MSI, Cartosat-2S + RISAT-1A / EOS-04

Run standalone training or fine-tuning with:
```powershell
python -m satquery.backend.training.train_sar_fusion --epochs 10 --batch-size 16
```
