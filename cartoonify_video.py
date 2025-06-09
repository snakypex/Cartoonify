import argparse
import os
import cv2
import torch
import numpy as np
from contextlib import nullcontext
from threading import Thread
from queue import Queue

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

def get_model(variant="paprika"):
    global _model
    if _model is None:
        repo = os.getenv("CARTOON_MODEL_REPO", "bryandlee/animegan2-pytorch:main")
        _model = torch.hub.load(repo, "generator", pretrained=variant)
        _model.to(device).eval().half()
        # 1) tracer + compilation PyTorch 2.0
        example = torch.randn(1,3,256,256, device=device).half()
        _model = torch.jit.trace(_model, example)
        _model = torch.compile(_model)
    return _model

def preprocess_batch(frames):
    # NumPy BGR→RGB, uint8→float16[-1,1], NHWC→NCHW
    arr = np.stack(frames)[:, :, :, ::-1].astype(np.float32) / 127.5 - 1
    t = (
        torch.from_numpy(arr)
        .permute(0, 3, 1, 2)
        .to(device, non_blocking=True)
        .half()
    )
    return t

def postprocess_batch(tensor):
    # tensor→uint8 BGR list
    out = ((tensor.clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).cpu().numpy()
    return [img[:,:,::-1] for img in out]

def reader_thread(cap, batch_size, q_read):
    buf = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(frame)
        if len(buf) == batch_size:
            q_read.put(buf)
            buf = []
    if buf:
        q_read.put(buf)
    q_read.put(None)

def writer_thread(out, q_write):
    while True:
        batch = q_write.get()
        if batch is None:
            break
        for f in batch:
            out.write(f)

def process_video(in_path, out_path, batch_size=8, progress_callback=None):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {in_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc= cv2.VideoWriter_fourcc(*"avc1")
    out   = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    model = get_model().half()
    q_read  = Queue(maxsize=2)
    q_write = Queue(maxsize=2)

    # Lancer les threads I/O
    t_r = Thread(
        target=reader_thread,
        args=(cap, batch_size, q_read),
        daemon=True,
    )
    t_w = Thread(target=writer_thread, args=(out, q_write), daemon=True)
    t_r.start()
    t_w.start()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    done  = 0

    stream = torch.cuda.Stream() if device.startswith("cuda") else None

    while True:
        batch = q_read.get()
        if batch is None:
            break

        # Prétraitement asynchrone sur un stream secondaire
        if stream:
            torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.stream(stream) if stream else nullcontext():
            inp = preprocess_batch(batch)
            with torch.no_grad(), torch.cuda.amp.autocast():
                out_t = model(inp)
        if stream:
            torch.cuda.current_stream().wait_stream(stream)

        # Post-traitement + écriture
        outs = postprocess_batch(out_t)
        q_write.put(outs)

        done += len(batch)
        if progress_callback and total:
            progress_callback(done/total)

    # Fin
    q_write.put(None)
    t_r.join()
    t_w.join()
    cap.release()
    out.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CartoonGAN accéléré (batch + fp16 + JIT + pipeline)"
    )
    p.add_argument("input", help="Chemin vidéo en entrée")
    p.add_argument("output", help="Chemin vidéo en sortie")
    p.add_argument("-b","--batch-size", type=int, default=8,
                   help="Taille du batch (optimiser selon votre GPU)")
    args = p.parse_args()

    def show_prog(ratio):
        print(f"Progression : {int(ratio*100)} %", end="\r", flush=True)

    process_video(
        args.input,
        args.output,
        batch_size=args.batch_size,
        progress_callback=show_prog,
    )
    print("\nTerminé.")

