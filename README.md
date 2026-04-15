# 🎛️ Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing

[![arXiv](https://img.shields.io/badge/arXiv-2604.10708-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2604.10708)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=Github&style=flat-square)](https://zeyuet.github.io/Audio-Omni/)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/HKUSTAudio/Audio-Omni)
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/HKUSTAudio/AudioEdit)
[![🤗 Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Zeyue7/Audio-Omni)

---

**Official repository for "Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing" (SIGGRAPH 2026).**

## ✨ Teaser

<p align="center">
  <img src="https://github.com/user-attachments/assets/be999a58-3e3b-45e8-abde-e449c647da56" alt="An overview of the Audio-Omni framework and its capabilities.">
</p>
<p align="center"><em>An overview of the Audio-Omni framework and its capabilities.</em></p>

---

## ✨ Abstract

Recent progress in multimodal models has spurred rapid advances in audio understanding, generation, and editing. However, these capabilities are typically addressed by specialized models, leaving the development of a truly unified framework that can seamlessly integrate all three tasks underexplored.

We introduce **Audio-Omni**, the first end-to-end framework to unify understanding, generation, and editing across general sound, music, and speech domains. Our architecture synergizes a frozen Multimodal Large Language Model (Qwen2.5-Omni) for high-level reasoning with a trainable Diffusion Transformer for high-fidelity synthesis.

To overcome the critical data scarcity in audio editing, we construct a large-scale, high-quality dataset for audio editing tasks. Audio-Omni demonstrates remarkable emergent abilities inherited from the MLLM, enabling sophisticated audio manipulation through natural language instructions.

## ✨ Method

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c46567-7f91-4e31-b077-8e325de665b7" alt="The Audio-Omni Framework.">
</p>
<p align="center"><em>The Audio-Omni Framework.</em></p>

---

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU
- FFmpeg and libsndfile

### Install

```bash
git clone https://github.com/Audio-Omni/Audio-Omni.git
cd Audio-Omni

conda create -n audio-omni python=3.11 -y
conda activate audio-omni

pip install -e .
conda install -c conda-forge ffmpeg libsndfile
```

Additional packages (install separately if needed):

```bash
pip install flash-attn --no-build-isolation   # optional, faster attention
pip install "qwen-omni-utils[decord]"          # Qwen2.5-Omni utilities
```

### Download Model

```bash
huggingface-cli download HKUSTAudio/Audio-Omni --local-dir model/
```

Or via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="HKUSTAudio/Audio-Omni", local_dir="model/")
```

After downloading:
```
model/
├── Audio-Omni.json              # Model configuration
├── model.ckpt                   # Model checkpoint
└── synchformer_state_dict.pth   # Synchformer checkpoint (for V2A/V2M)
```

---

## 🤗 Gradio Demo

```bash
bash infer_demo.sh
# or directly:
CUDA_VISIBLE_DEVICES=0 python3 run_gradio.py \
    --model-config model/Audio-Omni.json \
    --ckpt-path model/model.ckpt \
    --server-port 7777
```

The demo will be available at `http://localhost:7777`.

---

## 🎯 Supported Tasks

Audio-Omni supports **understanding**, **generation**, and **editing** in a single model:

| Task | Type | Text Prompt | Audio Input | Video Input | Voice Prompt |
|:-----|:-----|:------------|:------------|:------------|:-------------|
| **Understanding** | Understanding | Question about the audio/video | Optional | Optional | — |
| **Text-to-Audio** (T2A) | Generation | `"A clock ticking."` | — | — | — |
| **Text-to-Music** (T2M) | Generation | `"Compose a bright jazz swing instrumental..."` | — | — | — |
| **Video-to-Audio** (V2A) | Generation | — | — | `example.mp4` | — |
| **Video-to-Music** (V2M) | Generation | — | — | `example.mp4` | — |
| **Text-to-Speech** (TTS) | Generation | `"Hello, welcome to Audio-Omni."` | — | — | Optional |
| **Voice Conversion** (VC) | Generation | Transcript of target speech | — | — | `ref_voice.wav` |
| **Add** | Editing | — | `source.wav` | — | — |
| **Remove** | Editing | — | `source.wav` | — | — |
| **Extract** | Editing | — | `source.wav` | — | — |
| **Style Transfer** | Editing | — | `source.wav` | — | — |

---

## 🖥️ Python API

### Load Model

```python
from audio_omni import AudioOmni
import torchaudio

model = AudioOmni("model/Audio-Omni.json", "model/model.ckpt")
```

### Understanding

Freely combine text, audio, and video inputs — omit any that are not needed:

```python
# Audio understanding
response = model.understand("Describe the sounds in this audio.", audio="example/example.wav")

# Video understanding
response = model.understand("What is happening in this video?", video="example/example.mp4")

# Audio + Video
response = model.understand("Does the audio match the video?", audio="example/example.wav", video="example/example.mp4")

# Text-only
response = model.understand("What instruments are commonly used in jazz music?")
```

### Text-to-Audio

```python
audio = model.generate("T2A", prompt="A clock ticking.")
torchaudio.save("output.wav", audio, model.sample_rate)

# Text-to-Music
audio = model.generate("T2M", prompt="Compose a bright jazz swing instrumental with walking bass, brushed drums, and a lively horn melody.")
torchaudio.save("output_music.wav", audio, model.sample_rate)

# Video-to-Audio / Video-to-Music
audio = model.generate("V2A", video_path="example/example.mp4")
audio = model.generate("V2M", video_path="example/example.mp4")
```

### Text-to-Speech

```python
audio = model.generate("TTS", prompt="Hello, welcome to Audio-Omni.")
torchaudio.save("tts_output.wav", audio, model.sample_rate)

# With voice cloning
audio = model.generate(
    "TTS",
    prompt="Hello, welcome to Audio-Omni.",
    voice_prompt_path="ref_voice.wav",
    voice_ref_text="This is the reference transcript.",
)
```

### Audio Editing

```python
# Add a sound
audio = model.edit("Add", "example/edit/add/add.mp3", desc="skateboarding")
torchaudio.save("output_add.wav", audio, model.sample_rate)

# Remove a sound
audio = model.edit("Remove", "example/edit/remove/remove.mp3", desc="female singing")
torchaudio.save("output_remove.wav", audio, model.sample_rate)

# Extract a sound
audio = model.edit("Extract", "example/edit/extract/extract.mp3", desc="wood thrush calling")
torchaudio.save("output_extract.wav", audio, model.sample_rate)

# Style transfer
audio = model.edit("Style Transfer", "example/edit/transfer/example.mp3",
                   source_category="playing electric guitar", target_category="playing saxophone")
torchaudio.save("output_transfer.wav", audio, model.sample_rate)
```

---

## 📁 Project Structure

```
Audio-Omni/
├── audio_omni/                 # Main package
│   ├── api.py                  # High-level Python API (AudioOmni class)
│   ├── prompts.py              # Prompt templates for all tasks
│   ├── models/                 # Model implementations
│   ├── interface/              # Gradio UI
│   ├── inference/              # Generation & sampling
│   └── data/                   # Data utilities
├── model/                      # Model config & checkpoint
├── output/                     # Generated outputs
├── docs/                       # Documentation
└── README.md
```

---

## 📝 Citation

If you find our work useful, please cite:

```bibtex
@article{tian2026audioomni,
  title={Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing},
  author={Tian, Zeyue and Yang, Binxin and Liu, Zhaoyang and Zhang, Jiexuan and Yuan, Ruibin and Yin, Hubery and Chen, Qifeng and Li, Chen and Lv, Jing and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:submit/7470507},
  year={2026}
}
```

## 📭 Contact

If you have any comments or questions, feel free to contact:
- **Zeyue Tian**: ztianad@connect.ust.hk

---

## 📄 License

The code repository is released under the [CC BY-NC 4.0 License](LICENSE).

**Note**: Model weights are for research use only. Commercial use requires authorization from the authors.

---

## 🙏 Acknowledgments

We thank [AudioX](https://github.com/ZeyueT/AudioX), [VidMuse](https://github.com/ZeyueT/VidMuse), [MMAudio](https://github.com/hkchengrex/MMAudio), [F5-TTS](https://github.com/swivid/f5-tts), and [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) for their valuable contributions.

---

**⭐ Star us on GitHub if you like our project!**

