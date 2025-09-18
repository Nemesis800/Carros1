# clients/grpc_client.py
from __future__ import annotations
import sys
from pathlib import Path

import grpc

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto"
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))

import vehicle_pb2 as vpb  # type: ignore
import vehicle_pb2_grpc as vpb_grpc  # type: ignore


def main(video_path: str):
    chan = grpc.insecure_channel("localhost:50051")
    stub = vpb_grpc.VehicleServiceStub(chan)

    cfg = vpb.AppConfigMsg(
        model_name="yolo11n.pt",
        conf=0.30,
        iou=0.5,
        device="",
        line_orientation="vertical",
        line_position=0.50,
        invert_direction=False,
        capacity_car=50,
        capacity_moto=50,
        enable_csv=True,
        csv_dir=str(ROOT / "reports"),
        csv_name="from_grpc",
    )

    req = vpb.ProcessVideoRequest(
        video_path=video_path,
        config=cfg,
        stream_frames=False,  # si lo pones en True, el server también enviará JPEGs
    )

    for upd in stub.ProcessVideo(req):
        if upd.error:
            print("[ERROR]", upd.error)
            break
        if upd.progress:
            print(f"[PROG] {upd.progress*100:.0f}%")
        if upd.frame_jpeg:
            print(f"[FRAME] {len(upd.frame_jpeg)} bytes")
        if upd.done:
            print("[DONE] csv:", upd.csv_path)
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python clients/grpc_client.py C:\\ruta\\video.mp4")
        sys.exit(2)
    main(sys.argv[1])
