"""Frozen VideoMAE V2 Base (self-supervised) features for UCF101.
Per video: 16 uniformly spaced frames -> one clip -> final-block tokens (8 time x 14 x 14 x 768)
-> mean over space -> the model's own fc_norm -> (8, 768).  Also saves the model's native pooled vector (768).
Usage: python extract.py VIDEO_ROOT OUT_DIR      (writes OUT_DIR/features/<id>.npy and OUT_DIR/native/<id>.npy; resumable)"""
import sys, os, glob, time
import numpy as np, torch, cv2
from transformers import AutoModel, AutoConfig, VideoMAEImageProcessor

NAME, NFR, TT, SS = "OpenGVLab/VideoMAEv2-Base", 16, 8, 196
_proc = None


def vid(p): return os.path.basename(p).replace(".avi", "")


class Clips(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        global _proc
        if _proc is None: _proc = VideoMAEImageProcessor.from_pretrained(NAME)
        cap = cv2.VideoCapture(self.paths[i]); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f[:, :, ::-1])
        cap.release()
        idx = np.linspace(0, len(fr) - 1, NFR).round().astype(int)
        return vid(self.paths[i]), _proc([fr[j] for j in idx], return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def tokens(m, pv):
    """pv (B,16,3,224,224) -> final-block tokens (B, 1568, 768); same computation as VisionTransformer.forward_features."""
    x = m.patch_embed(pv.permute(0, 2, 1, 3, 4))
    x = x + m.pos_embed.expand(x.size(0), -1, -1).type_as(x).to(x.device)
    for blk in m.blocks: x = blk(x)
    return x


def main(video_root, out):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in ("features", "native"): os.makedirs(f"{out}/{d}", exist_ok=True)
    paths = sorted(glob.glob(f"{video_root}/**/*.avi", recursive=True))
    todo = [p for p in paths if not os.path.exists(f"{out}/features/{vid(p)}.npy")]
    print(f"{len(paths)} videos, {len(todo)} to do, device {dev}", flush=True)
    cfg = AutoConfig.from_pretrained(NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(NAME, config=cfg, trust_remote_code=True).to(dev).eval()
    m = model.model
    dl = torch.utils.data.DataLoader(Clips(todo), batch_size=16, num_workers=min(4, os.cpu_count() or 1))
    t0, n, checked = time.time(), 0, False
    for ids, pv in dl:
        pv = pv.to(dev)
        x = tokens(m, pv); B = x.size(0)
        assert x.shape[1] == TT * SS, x.shape
        with torch.no_grad():
            seq = m.fc_norm(x.view(B, TT, SS, -1).mean(2))          # (B,8,768)
            native = m.fc_norm(x.mean(1))                            # (B,768), the model's own clip vector
            if not checked:                                          # sanity: our native vector equals the model's own output
                ref = model.extract_features(pv.permute(0, 2, 1, 3, 4))
                assert torch.allclose(ref, native, atol=1e-3), float((ref - native).abs().max())
                checked = True; print("native-vector check passed", flush=True)
        for i, v in enumerate(ids):
            np.save(f"{out}/features/{v}.npy", seq[i].float().cpu().numpy())
            np.save(f"{out}/native/{v}.npy", native[i].float().cpu().numpy())
        n += B
        if (n // 16) % 20 == 0: print(f"{n}/{len(todo)}  {n / (time.time() - t0):.2f} videos/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
