#!/bin/bash
set -e

ENV_PREFIX="/aifs4su/data/tianzeyue/conda_envs/AudioOmni"
PROJECT_DIR="/aifs4su/data/tianzeyue/project/Audio-Omni/AudioX-sync"
LOG_FILE="${PROJECT_DIR}/setup_env.log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=========================================="
echo "开始创建 AudioOmni 环境"
echo "时间: $(date)"
echo "=========================================="

# Step 1: 创建 conda 环境
echo "[1/5] 创建 conda 环境 (Python 3.11) ..."
if [ -d "$ENV_PREFIX" ]; then
    echo "环境已存在，跳过创建"
else
    conda create --prefix "$ENV_PREFIX" python=3.11 -y
fi

# Step 2: 激活环境
echo "[2/5] 激活环境 ..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
echo "Python: $(python --version)"
echo "pip: $(pip --version)"

# Step 3: 安装 PyTorch 2.6 (CUDA 12.4)
echo "[3/5] 安装 PyTorch 2.6.0 + torchaudio + torchvision ..."
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Step 4: 安装其他依赖
echo "[4/5] 安装项目依赖 ..."

pip install \
    "transformers>=4.45.0" \
    huggingface_hub \
    safetensors \
    accelerate \
    "einops>=0.7.0" \
    "einops-exts>=0.0.4" \
    soundfile \
    librosa \
    tqdm \
    decord \
    opencv-python \
    sentencepiece \
    omegaconf \
    timm \
    "pytorch_lightning>=2.1.0" \
    "ema-pytorch>=0.2.3" \
    wandb \
    torchmetrics \
    pandas \
    webdataset \
    "alias-free-torch==0.0.6" \
    "local-attention>=1.8.6" \
    "vector-quantize-pytorch>=1.9.14" \
    "x-transformers<1.27.0" \
    pedalboard \
    "prefigure==0.0.9" \
    importlib-resources \
    PyWavelets \
    scipy \
    numpy \
    packaging \
    jieba \
    pypinyin \
    "gradio>=4.0.0" \
    argbind \
    Pillow \
    matplotlib \
    requests \
    s3fs \
    scikit-learn \
    "qwen-omni-utils[decord]"

# MMAudio (sync feature extraction)
pip install mmaudio || echo "WARNING: mmaudio 安装失败，继续..."

# 以下包可能有兼容性问题，逐个安装并容错
pip install "k-diffusion==0.1.1" || pip install k-diffusion || echo "WARNING: k-diffusion 安装失败，继续..."
pip install "aeiou==0.0.20" || pip install aeiou || echo "WARNING: aeiou 安装失败，继续..."
pip install "auraloss==0.4.0" || pip install auraloss || echo "WARNING: auraloss 安装失败，继续..."
pip install "descript-audio-codec==1.0.0" || pip install descript-audio-codec || echo "WARNING: descript-audio-codec 安装失败，继续..."
pip install "encodec==0.1.1" || pip install encodec || echo "WARNING: encodec 安装失败，继续..."
pip install "laion-clap==1.1.4" || pip install laion-clap || echo "WARNING: laion-clap 安装失败，继续..."
pip install "v-diffusion-pytorch==0.0.2" || pip install v-diffusion-pytorch || echo "WARNING: v-diffusion-pytorch 安装失败，继续..."
pip install descript-audiotools || pip install audiotools || echo "WARNING: audiotools 安装失败，继续..."
pip install deepspeed || echo "WARNING: deepspeed 安装失败，继续..."

# protobuf: wandb 需要 >=4.21, descript-audiotools 要求 <3.20 会降级导致冲突，这里升回来
pip install "protobuf>=4.21,<7" || echo "WARNING: protobuf 升级失败，继续..."

# libsndfile (soundfile 的系统依赖，通过 conda 安装)
echo "[4.1/5] 安装 libsndfile (via conda) ..."
conda install -y -c conda-forge libsndfile

# Flash Attention (可选，推荐，需要 nvcc)
echo "[4.2/5] 安装 flash-attn (可选) ..."
if command -v nvcc &>/dev/null; then
    export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
    export TMPDIR="${TMPDIR:-/aifs4su/data/tianzeyue/tmp}"
    mkdir -p "$TMPDIR"
    pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn 编译失败，将使用标准 attention"
else
    echo "未找到 nvcc，安装 CUDA toolkit 后再装 flash-attn ..."
    conda install -y -c nvidia cuda-toolkit=12.4
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    export TMPDIR="${TMPDIR:-/aifs4su/data/tianzeyue/tmp}"
    mkdir -p "$TMPDIR"
    pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn 编译失败，将使用标准 attention"
fi

# Step 5: 安装项目本身 (editable mode)
echo "[5/5] 安装项目 (editable mode) ..."
cd "$PROJECT_DIR"
pip install -e . --no-deps

echo ""
echo "=========================================="
echo "环境创建完成！"
echo "时间: $(date)"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_PREFIX"
echo "  cd $PROJECT_DIR"
echo "  bash infer_demo.sh"
echo ""

# 验证关键包
echo "验证关键包版本:"
python -c "
import torch; print(f'  torch: {torch.__version__}')
import torchaudio; print(f'  torchaudio: {torchaudio.__version__}')
import torchvision; print(f'  torchvision: {torchvision.__version__}')
import transformers; print(f'  transformers: {transformers.__version__}')
import gradio; print(f'  gradio: {gradio.__version__}')
import einops; print(f'  einops: {einops.__version__}')
import decord; print('  decord: OK')
import safetensors; print(f'  safetensors: {safetensors.__version__}')
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print()
print('所有关键包验证通过!')
"

echo ""
echo "冻结完整包列表到 requirements_frozen.txt ..."
pip freeze > "${PROJECT_DIR}/requirements_frozen.txt"
echo "Done! 查看 ${LOG_FILE} 获取完整日志"
