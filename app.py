import gradio as gr
from asr import transcribe, ASR_EXAMPLES

mms_select_source_trans = gr.Radio(
    ["Record from Mic", "Upload audio"],
    label="Audio input",
    value="Record from Mic",
)
mms_mic_source_trans = gr.Audio(
    sources=["microphone"], type="filepath", label="Use mic"
)
mms_upload_source_trans = gr.Audio(
    sources=["upload"], type="filepath", label="Upload file", visible=False
)

mms_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Dropdown(
            [
                "original",
                "finetune",
            ],
            label="Model",
            value="finetune",
        ),
        mms_select_source_trans,
        mms_mic_source_trans,
        mms_upload_source_trans,
    ],
    outputs="text",
    examples=ASR_EXAMPLES,
    title="Auto Speech Recognition Demo",
    description=(
        "Transcribe audio from a microphone or input file in your desired language."
    ),
    allow_flagging="never",
)

with gr.Blocks() as demo:
    mms_transcribe.render()
    mms_select_source_trans.change(
        lambda x: [
            gr.update(visible=True if x == "Record from Mic" else False),
            gr.update(visible=True if x == "Upload audio" else False),
        ],
        inputs=[mms_select_source_trans],
        outputs=[mms_mic_source_trans, mms_upload_source_trans],
        queue=False,
    )

demo.launch()
