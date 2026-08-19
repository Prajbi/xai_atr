import argparse
from ultralytics import YOLO
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--model", default="yolov10l.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=args.resume,device=args.device,workers=args.workers)
    else:
        model = YOLO(args.model)
        model.train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=args.device,workers=8,patience=20,project="runs/detect",name="yolov10l_atr",save=True,plots=True,)
        metrics = model.val()
        print("mAP50-95:", metrics.box.map)
        print("mAP50:", metrics.box.map50)
if __name__ == "__main__":
    main()
