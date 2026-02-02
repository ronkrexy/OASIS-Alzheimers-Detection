## Citation

Paper submission to Springer LNCS 2026.

*Paper PDF will be available after publication.*

For now, see the repository for full implementation details.
# OASIS-Alzheimers-Detection
Comparative analysis of ResNet50 vs MobileNetV2 on OASIS MRI data.

# Data Leakage in Alzheimer's Detection

Code for my paper on why most Alzheimer's detection papers report inflated accuracy numbers.

**TL;DR:** If you don't split by patient ID, your 99% accuracy is just memorizing skull shapes. Real accuracy? 75%.

## The Problem

Everyone's publishing papers claiming 95-99% accuracy on OASIS. Sounds great until you realize they're putting the same patient's brain scans in both training and test sets. 

This is like studying for an exam by memorizing the answer key, then being surprised when you ace it.

## What I Actually Did

Ran ResNet50 and MobileNetV2 two ways:
1. **Wrong way** (like everyone else): Random split gets 100% accuracy
2. **Right way** (by patient): Same model gets 75% accuracy

That's a 25 point difference just from splitting properly.

## Results

| Model | Naive Split | Proper Split | Gap |
|-------|------------|--------------|-----|
| ResNet50 | 100% | 75% | -25% |
| MobileNetV2 | 80% | 71% | -9% |

Both models completely fail at detecting "Moderate Dementia" (F1 = 0.00). Not great.

## Running It
```bash
# Get OASIS from Kaggle
kaggle datasets download -d ninadaithal/imagesoasis

# Install dependencies
pip install tensorflow scikit-learn matplotlib seaborn

# Run the paper experiments
python paper_run.py
```

Needs a decent GPU (I used an L40S). Takes about 3 hours.

## Repository Structure
```
├── train.py              # Main training script
├── paper_run.py         # Reproduces paper results
├── models/              # ResNet50 and MobileNetV2
├── utils/               # Data loading, focal loss, metrics
└── results/             # Outputs and figures
```

## The Key Part
```python
# DON'T do this:
train_test_split(images, test_size=0.2)  # WRONG!

# DO this:
patients = group_by_patient_id(images)
train_patients, test_patients = train_test_split(patients, test_size=0.2)
train_images = get_images_for_patients(train_patients)
test_images = get_images_for_patients(test_patients)
```

Zero patient overlap between train and validation. That's it.

## Why This Matters

If you're building medical imaging models and not doing patient-level splits, your validation numbers are meaningless. Period.

The field has a reproducibility problem and this is a big part of it.

## Technical Details

**Hardware:** NVIDIA L40S (48GB VRAM)  
**Framework:** TensorFlow 2.15 with mixed precision  
**Dataset:** OASIS (86,437 slices from 347 patients)  
**Training time:** ~50 GPU hours for both models  

**Optimizations:**
- Mixed precision (FP16) for 1.8x speedup
- Batch size: 128
- Multi-class focal loss with class weighting
- Data augmentation: horizontal flip, rotation, zoom

## What Worked (and What Didn't)

**Helped a bit:**
- Focal loss: +1.8% accuracy
- Data augmentation: +3.6%
- Class weights: +2.2%

**Made it worse:**
- Freezing ImageNet backbone: -6.9%

**Bottom line:** Nothing fixes the fundamental problem that we only have 244 "Moderate Dementia" samples from about 15 patients. You can't learn rare classes from that.

## Ablation Results

| Configuration | Accuracy | Change |
|--------------|----------|--------|
| Full model | 75.0% | baseline |
| Without focal loss | 73.2% | -1.8% |
| Without augmentation | 71.4% | -3.6% |
| Without class weights | 72.8% | -2.2% |
| Frozen backbone | 68.1% | -6.9% |
| All removed | 65.3% | -9.7% |

## Limitations

- Only tested 2D models (3D CNNs might help)
- Only T1-weighted MRI (multi-modal fusion could improve)
- Single run per experiment (computational cost)
- No clinical validation with radiologists

See paper Section VI for full discussion.

## Citation

```bibtex
@inproceedings{rexy2025alzheimers,
  title={Benchmarking Deep Learning Architectures for Alzheimer's Detection: 
         Quantifying the Impact of Data Leakage on Model Generalization},
  author={Rexy, Ron K},
}
```

## License

MIT License - see LICENSE file

## Contact

Ron K Rexy  
Department of Pharmaceutical Engineering and Technology  
Indian Institute of Technology (BHU) Varanasi  
ronk.rexy.phe22@itbhu.ac.in

Found a bug? Open an issue.  
Questions about methodology? Email me.
