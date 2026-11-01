"""Frame-local features (no mixing across time slots): for time slot t the encoder sees only frames 2t and 2t+1, tiled 8 times to fill
the 16-frame input; the slot vector is fc_norm(mean of all tokens). 8 forward passes per video -> (8, 768), same shape as extract.py.
Usage: python extract_local.py VIDEO_ROOT OUT_DIR   (writes OUT_DIR/features/<id>.npy; resumable)"""
import sys, os, glob, time
import numpy as np, torch
import extract as ex
from transformers import AutoModel, AutoConfig

def main(video_root, out):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(f"{out}/features", exist_ok=True)
    paths = sorted(glob.glob(f"{video_root}/**/*.avi", recursive=True)); todo = [p for p in paths if not os.path.exists(f"{out}/features/{ex.vid(p)}.npy")]
    print(f"{len(paths)} videos, {len(todo)} to do, device {dev}", flush=True)
    cfg = AutoConfig.from_pretrained(ex.NAME, trust_remote_code=True)
    m = AutoModel.from_pretrained(ex.NAME, config=cfg, trust_remote_code=True).to(dev).eval().model
    ds = ex.Clips(todo); t0 = time.time()
    for i in range(len(ds)):
        v, pv = ds[i]; pv = pv.to(dev)                                    # (16,3,224,224)
        clips = torch.stack([pv[2 * t:2 * t + 2].repeat(8, 1, 1, 1) for t in range(8)])   # (8,16,3,224,224): slot t = frames 2t,2t+1 only
        with torch.no_grad():
            x = ex.tokens(m, clips); seq = m.fc_norm(x.mean(1))           # (8,768)
        np.save(f"{out}/features/{v}.npy", seq.float().cpu().numpy())
        print(f"{i+1}/{len(ds)} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
