# ASR (Automatic Speech Recognition) demo for Shan language


## Models
- [Finetuned](https://huggingface.co/NorHsangPha/wav2vec2-large-mms-1b-shan) on 600 samples [Shan-ASR-Nova](https://huggingface.co/datasets/NorHsangPha/Shan-ASR-Nova) datasets
- [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all)


## Finetune model
My fine-tuned model aims to address some pronunciation and missing vocabulary issues with newer Shan consonants like 'ၾ'.
However, small, noisy, and low-quality datasets do not yield significant improvements over the original model.

## Usage
```bash
# install requirements
pip install -r requirements.txt
```

```bash
# run
python app.py

# or gradio debug mode
gradio app.py
```
