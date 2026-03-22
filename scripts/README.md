# Agent Safety Monitor

A trace-based safety classifier trained on real malicious PyPI packages to detect credential exfiltration in AI agent tool sequences.

## The Idea

Malware authors have already solved the hardest problem in agent safety research: labeling real exfiltration attacks. This project repurposes malicious PyPI packages as ground-truth behavioral data for training a safety monitor, exploiting the structural analogy between supply chain attacks and misaligned agent tool usage.

A malicious package that reads `AWS_SECRET_ACCESS_KEY`, encodes it, and posts it to an external server executes the same behavioral sequence as a misaligned AI agent doing the same through tool calls. The monitor learns to detect both.

## Results

| System | FNR | FPR | Cost-Weighted |
|---|---|---|---|
| Rule-based (regex) | 0.15 | 1.00 | 40 |
| Fine-tuned Phi-3-mini | 0.00 | 0.10 | 1 |

Evaluated on 30 hand-written test examples isolated before training. The fine-tuned model caught every attack and produced one false alarm. The rule-based system flagged every safe action as unsafe.

## Dataset

- **Source:** DataDog's malicious-software-packages-dataset (2,339 packages, 2023-2024)
- **Extracted:** 339 clean exfiltration traces (85.5% exclusion rate -- static analysis only)
- **Safe examples:** Deployment-realistic behavioral templates (boto3, twine, httpx, CI/CD patterns)
- **Test set:** 30 hand-written examples, never seen during training

## Pipeline
```
DataDog malicious packages
        |
        v
AST static analysis (src/ast_extractor.py)
        |
        v
Filter: keep interpretable traces only
        |
        v
Labeled dataset (SAFE / UNSAFE)
        |
        v
QLoRA fine-tune Phi-3-mini
Autoresearch loop (54 configs, 8 hours)
        |
        v
Evaluation: FNR, FPR, cost-weighted score
```

## Autoresearch

We adapt Karpathy's autoresearch pattern from language model pretraining to classification fine-tuning. The loop varies LoRA rank, learning rate, epochs, and threshold across 54 configurations overnight. Best config: rank=8, lr=5e-4, epochs=2, threshold=0.3.

54/54 runs completed. 38/54 achieved perfect validation scores. Total compute cost: ~$18 on Lambda Labs GH200.

## Quickstart
```bash
git clone https://github.com/selimsevim/agent-safety-monitor
cd agent-safety-monitor
pip install -r requirements.txt

# Extract traces from DataDog dataset
git clone --filter=blob:none --sparse \
  https://github.com/DataDog/malicious-software-packages-dataset \
  data/sources/datadog
cd data/sources/datadog && git sparse-checkout set samples/pypi/malicious_intent && cd ../../..

python scripts/build_dataset.py
python scripts/augment.py

# Train
python scripts/train.py \
  --train data/processed/train_augmented.jsonl \
  --val data/processed/val.jsonl \
  --output models/adapters/phi3-safety-v1 \
  --lora-rank 8 \
  --epochs 2 \
  --lr 5e-4

# Evaluate
python scripts/evaluate.py \
  --test data/processed/test.jsonl \
  --adapter models/adapters/phi3-safety-v1 \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --output results/eval_output.json \
  --threshold 0.3

# Overnight hyperparameter search
nohup python scripts/autoresearch.py \
  --train data/processed/train_augmented.jsonl \
  --val data/processed/val.jsonl \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --output models/adapters \
  --budget 8 > results/nohup.log 2>&1 &
```

## Repository Structure
```
├── data/processed/          # Extracted and labeled traces
├── scripts/
│   ├── build_dataset.py     # AST extraction from malicious packages
│   ├── augment.py           # Tool aliasing + step reordering
│   ├── train.py             # QLoRA fine-tuning
│   ├── evaluate.py          # FNR, FPR, cost-weighted metrics
│   └── autoresearch.py      # Overnight hyperparameter search loop
├── src/
│   └── ast_extractor.py     # Core AST parser and tool vocabulary mapper
├── models/adapters/         # Saved LoRA adapters
├── results/                 # Evaluation outputs and autoresearch logs
└── demo/
    └── app.py               # Gradio demo
```

## Known Limitations

- Safe training examples are template-generated, not extracted from real benign packages
- Static analysis excludes 85.5% of packages (dynamic imports, obfuscation, binary droppers)
- Fixed tool vocabulary -- custom APIs outside the 8 canonical tools are not covered
- Not robust against adaptive adversaries who know the classifier

## Dataset Note

The research design targets the Backstabber's Knife Collection as the primary source. Access requires institutional email verification. This prototype uses DataDog's dataset, which is larger and more recent. The extraction pipeline is source-agnostic -- BKC integration requires no code changes.

## Citation
```bibtex
@misc{sevim2026supplychain,
  title={Supply Chain Malware as Ground Truth: Training a Trace-Based Agent Safety Monitor on Real-World Credential Exfiltration Patterns},
  author={Sevim, Selim},
  year={2026},
  note={AI Control Hackathon 2026, Apart Research x Redwood Research}
}
```

## Acknowledgements

Built for the Apart Research AI Control Hackathon 2026, co-organized with Redwood Research.
Dataset: DataDog Security Labs malicious-software-packages-dataset.
Base model: Microsoft Phi-3-mini-4k-instruct.
Autoresearch pattern: Karpathy (2026).
