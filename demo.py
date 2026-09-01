from pathlib import Path
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw

from loma import LoMa
from loma.loma import (
    LoMaB,
    filter_matches,
    to_pixel_coords    
)
from loma.cfg import LoMaConfig

def draw_keypoints(
    im_path: str,
    kpts_xy: np.ndarray,
    save_path: Path,
    radius: int | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    img = Image.open(im_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Scale marker radius to image size so points stay visible on very large images.
    if radius is None:
        radius = max(2, round(max(img.size) / 800))
    for x, y in kpts_xy:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    print(f"Saved {len(kpts_xy)} keypoints to {save_path}")


def main(
    matcher: LoMaConfig = LoMaB(),
    im_A: str = "assets/P0634995_cm9818.jpg",
    im_B: str = "assets/P0635004_cm7430.jpg",
    save_path: str = "demo/matches_P.jpg",
):
    model = LoMa(matcher)

    # NOTE: you can also simply use the kptsA, kptsB = model.match(im_A, im_B) API
    kpts_A, desc_A, h1, w1 = model.detect_and_describe(im_A)
    kpts_B, desc_B, h2, w2 = model.detect_and_describe(im_B)
    with torch.inference_mode():
        scores = model(kpts_A, kpts_B, desc_A, desc_B)["scores"]
    m0, *_ = filter_matches(scores, model.cfg.filter_threshold)
    valid = m0[0] > -1
    matched_A = to_pixel_coords(kpts_A[0][torch.where(valid)[0]], h1, w1).cpu().numpy()
    matched_B = to_pixel_coords(kpts_B[0][m0[0][valid]], h2, w2).cpu().numpy()

    save_path_p = Path(save_path)
    kpts_A_px = to_pixel_coords(kpts_A[0], h1, w1).cpu().numpy()
    kpts_B_px = to_pixel_coords(kpts_B[0], h2, w2).cpu().numpy()
    kpts_A_save = save_path_p.with_name(f"{save_path_p.stem}_kpts_A{save_path_p.suffix}")
    kpts_B_save = save_path_p.with_name(f"{save_path_p.stem}_kpts_B{save_path_p.suffix}")
    draw_keypoints(im_A, kpts_A_px, kpts_A_save)
    draw_keypoints(im_B, kpts_B_px, kpts_B_save)

    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)))
    canvas.paste(Image.open(im_A).convert("RGB"), (0, 0))
    canvas.paste(Image.open(im_B).convert("RGB"), (w1, 0))
    draw = ImageDraw.Draw(canvas)
    rng = np.random.default_rng(0)
    for (x1, y1), (x2, y2) in zip(matched_A, matched_B):
        color = tuple(rng.integers(0, 256, 3).tolist())
        draw.line([(x1, y1), (x2 + w1, y2)], fill=color, width=1)

    save_path_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path_p)
    print(f"Saved {len(matched_A)} matches to {save_path_p}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
