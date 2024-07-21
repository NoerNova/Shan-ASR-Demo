import os
import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

ASR_SAMPLING_RATE = 16_000


def transcribe(model_name: str, audio_source=None, microphone=None, file_upload=None):
    if type(microphone) is dict:
        microphone = microphone["name"]

    audio_fp = (
        file_upload if "upload" in str(audio_source or "").lower() else microphone
    )

    if audio_fp is None:
        return "ERROR: You have to either use the microphone or upload an audio file"

    audio_samples = librosa.load(audio_fp, sr=ASR_SAMPLING_RATE, mono=True)[0]

    model_id = {
        "original": "facebook/mms-1b-all",
        "finetune": "NorHsangPha/wav2vec2-large-mms-1b-shan",
    }[model_name]

    auth_token = os.environ.get("TOKEN_READ_SECRET") or True

    if model_name == "original":
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.set_target_lang("shn")
        model.load_adapter("shn")
    elif model_name == "finetune":
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id, target_lang="shn", ignore_mismatched_sizes=True, token=auth_token
        )
        processor = AutoProcessor.from_pretrained(model_id, token=auth_token)
    else:
        return "ERROR: Wrong model name, or model not available please restart."

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    inputs = processor(
        audio_samples, sampling_rate=ASR_SAMPLING_RATE, return_tensors="pt"
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    return transcription


ASR_EXAMPLES = [
    ["finetune", "Upload audio", None, "upload/sample1.wav"],
    ["finetune", "Upload audio", None, "upload/sample2.wav"],
    ["original", "Upload audio", None, "upload/sample1.wav"],
    ["original", "Upload audio", None, "upload/sample2.wav"],
]
