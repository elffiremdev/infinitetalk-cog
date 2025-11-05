# predict.py
import os
import json
import shutil
import importlib.util
from pathlib import Path
from typing import Optional

from cog import BasePredictor, Input, Path as CogPath
from huggingface_hub import snapshot_download, hf_hub_download

WEIGHTS_DIR = Path("/src/weights")
CODE_DIR = Path("/src/InfiniteTalk")

WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAV2VEC_ID = "TencentGameMate/chinese-wav2vec2-base"
INFINITETALK_ID = "MeiGen-AI/InfiniteTalk"

class Predictor(BasePredictor):
    def setup(self):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1) Orijinal kodu klonla
        if not CODE_DIR.exists():
            os.system(f"git clone --depth=1 https://github.com/MeiGen-AI/InfiniteTalk.git {CODE_DIR}")

        # 2) Ağırlıkları indir (gerekirse HF token kullan: HUGGINGFACE_TOKEN)
        token = os.getenv("HUGGINGFACE_TOKEN", None)

        snapshot_download(
            repo_id=WAN_MODEL_ID,
            local_dir=WEIGHTS_DIR / "Wan2.1-I2V-14B-480P",
            local_dir_use_symlinks=False,
            token=token,
        )

        snapshot_download(
            repo_id=WAV2VEC_ID,
            local_dir=WEIGHTS_DIR / "chinese-wav2vec2-base",
            local_dir_use_symlinks=False,
            token=token,
        )

        snapshot_download(
            repo_id=INFINITETALK_ID,
            local_dir=WEIGHTS_DIR / "InfiniteTalk",
            local_dir_use_symlinks=False,
            token=token,
        )

        # wav2vec2 için bazı revizyonlarda ek safetensors gerekebilir:
        # hf_hub_download(repo_id=WAV2VEC_ID, filename="model.safetensors", revision="refs/pr/1",
        #                 local_dir=WEIGHTS_DIR / "chinese-wav2vec2-base", token=token)

        # 3) generate_infinitetalk.py modül olarak içe aktar
        self.generate_module = self._import_generate_module()

    def _import_generate_module(self):
        gen_path = CODE_DIR / "generate_infinitetalk.py"
        spec = importlib.util.spec_from_file_location("generate_infinitetalk", gen_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    def _write_input_json(
        self,
        image: Optional[CogPath],
        video: Optional[CogPath],
        audio: CogPath,
        workdir: Path,
    ) -> Path:
        data = {"audio_path": str(audio)}
        if image is not None:
            data["image_path"] = str(image)
        if video is not None:
            data["video_path"] = str(video)

        jpath = workdir / "input.json"
        with open(jpath, "w") as f:
            json.dump(data, f)
        return jpath

    def predict(
        self,
        mode: str = Input(
            description="Generation mode",
            choices=["streaming", "clip"],
            default="streaming",
        ),
        size: str = Input(
            description="Output resolution preset",
            choices=["infinitetalk-480", "infinitetalk-720"],
            default="infinitetalk-480",
        ),
        image: Optional[CogPath] = Input(
            description="Image file for image-to-video (PNG/JPG). Leave empty if using video-to-video.",
            default=None,
        ),
        video: Optional[CogPath] = Input(
            description="Video file for video-to-video (MP4/MOV). Leave empty if using image-to-video.",
            default=None,
        ),
        audio: CogPath = Input(
            description="Audio file (WAV/MP3/M4A/OGG/FLAC)",
        ),
        sample_steps: int = Input(description="Sampling steps", default=40, ge=1, le=100),
        motion_frame: int = Input(description="Motion frame window", default=9, ge=1, le=32),
        sample_text_guide_scale: float = Input(description="Text CFG (no LoRA: ~5, with LoRA: ~1)", default=5.0),
        sample_audio_guide_scale: float = Input(description="Audio CFG (no LoRA: ~4, with LoRA: ~2)", default=4.0),
        max_frame_num: int = Input(description="Max frames (~25 fps; 1000≈40s)", default=1000, ge=50, le=3600),
        use_teacache: bool = Input(description="Enable TeaCache acceleration", default=False),
        use_apg: bool = Input(description="Enable APG", default=False),
    ) -> CogPath:
        if (image is None) and (video is None):
            raise ValueError("Provide either an image (for I2V) or a video (for V2V).")

        workdir = Path("/src/run")
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        input_json = self._write_input_json(image, video, audio, workdir)
        out_prefix = workdir / "infinitetalk_res"

        args = [
            "--ckpt_dir", str(WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir", str(WEIGHTS_DIR / "chinese-wav2vec2-base"),
            "--infinitetalk_dir", str(WEIGHTS_DIR / "InfiniteTalk" / "single" / "infinitetalk.safetensors"),
            "--input_json", str(input_json),
            "--size", size,
            "--sample_steps", str(sample_steps),
            "--mode", mode,
            "--motion_frame", str(motion_frame),
            "--save_file", str(out_prefix),
            "--max_frame_num", str(max_frame_num),
            "--sample_text_guide_scale", str(sample_text_guide_scale),
            "--sample_audio_guide_scale", str(sample_audio_guide_scale),
        ]
        if use_teacache:
            args += ["--use_teacache"]
        if use_apg:
            args += ["--use_apg"]

        # Modül doğrudan çağrılabiliyorsa (tercihimiz):
        if hasattr(self.generate_module, "main"):
            self.generate_module.main(args)  # type: ignore
        else:
            # fallback: subprocess
            import subprocess
            cmd = ["python", str(CODE_DIR / "generate_infinitetalk.py")] + args
            subprocess.run(cmd, check=True)

        out_mp4 = Path(str(out_prefix) + ".mp4")
        if not out_mp4.exists():
            raise RuntimeError("Generation finished but output file not found.")
        return CogPath(out_mp4)
