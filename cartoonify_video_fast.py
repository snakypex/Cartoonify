import argparse
import cv2
import os
import torch
import numpy as np

# Détection du device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache du modèle
_model = None

# Paramètres de normalisation ImageNet
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def get_model(style="candy", repo="pytorch/examples"):
    """
    Charge le modèle Fast Neural Style (Torch Hub) pour un style donné.
    """
    global _model
    if _model is None:
        repo = "pytorch/examples"  # on enlève ":main", torch.hub prendra la branche par défaut (main)
        _model = torch.hub.load(repo, "fast_neural_style", style, pretrained=True)
    return _model


def preprocess(frames):
    """
    Convertit une liste de frames BGR uint8 en un tensor (B,3,H,W) normalisé.
    """
    # Stack → (B,H,W,3), [0,255]→[0,1], BGR→RGB
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = arr[..., ::-1]
    # Normalisation ImageNet
    arr = (arr - _MEAN) / _STD
    # NHWC→NCHW + envoi GPU
    tensor = torch.from_numpy(arr).permute(0,3,1,2).to(DEVICE).half()
    return tensor


def postprocess(tensor_out):
    """
    Reconvertit le tensor (B,3,H,W) stylisé en liste de frames BGR uint8.
    """
    # NCHW→NHWC CPU→numpy
    arr = tensor_out.cpu().permute(0,2,3,1).numpy()
    # Dénormalisation & clamp
    arr = arr * _STD + _MEAN
    arr = np.clip(arr, 0, 1) * 255
    arr = arr.astype(np.uint8)
    # RGB→BGR
    arr = arr[..., ::-1]
    return [frame for frame in arr]


def process_video(input_path, output_path, style="candy", batch_size=4):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo {input_path}")

    # Infos vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    model = get_model(style)
    buffer = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        # Quand on a un batch complet
        if len(buffer) == batch_size:
            inp = preprocess(buffer)
            with torch.no_grad():
                out_t = model(inp)
            for styled in postprocess(out_t):
                out.write(styled)
            processed += batch_size
            buffer.clear()
            print(f"Progress: {processed}/{total}", end="\r")

    # Traiter les restes
    if buffer:
        inp = preprocess(buffer)
        with torch.no_grad():
            out_t = model(inp)
        for styled in postprocess(out_t):
            out.write(styled)
        processed += len(buffer)
        print(f"Progress: {processed}/{total}", end="\r")

    cap.release()
    out.release()
    print("\nTerminé !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stylisation vidéo Fast Neural Style (batch + FP16)"
    )
    parser.add_argument("input", help="Chemin vers la vidéo d'entrée")
    parser.add_argument("output", help="Chemin vers la vidéo de sortie")
    parser.add_argument("--style", default="candy",
                        help="Nom du style (e.g. candy, mosaic, rain_princess, udnie)")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Taille du batch pour le GPU")
    args = parser.parse_args()

    process_video(args.input, args.output, style=args.style, batch_size=args.batch_size)
