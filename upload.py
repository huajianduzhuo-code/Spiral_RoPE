from huggingface_hub import upload_file

# 1. 这里填你的 HF 仓库名字：<username>/<model_id>
REPO_ID = "haoyuliu00/Spiral_RoPE"

# 2. 这里填你本地某一个 ckpt 的路径（绝对路径或相对路径都行）
LOCAL_PATH = "/home/haoyuliu/rope_dit/hf_downloads/model.pth"

# 3. 这里是你希望在 HF 仓库里看到的文件名/路径
#    建议分目录管理，比如 "xl2/256/spiral_ema.pth"
REMOTE_PATH = "dit_xl2/0700000.pth"

if __name__ == "__main__":
    print(f"Uploading {LOCAL_PATH} -> {REPO_ID}:{REMOTE_PATH}")
    upload_file(
        path_or_fileobj=LOCAL_PATH,
        path_in_repo=REMOTE_PATH,
        repo_id=REPO_ID,
    )
    print("Done.")
# cat /home/haoyuliu/efs_pod/efs/haoyuliu/dit_results/DiT-XL-2/RoPE_rotate4_theta10000_freqslang_maxfreqlang1.5_minfreqlang0.0_trunc0.0_samethetaFalse_seed0/eval/fid_results.txt 