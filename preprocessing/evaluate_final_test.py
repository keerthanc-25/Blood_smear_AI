from ultralytics import YOLO

MODEL = r"D:\Blood_Smear_AI\runs\FINAL_YOLO11N_3CLASS-3\weights\best.pt"
DATA = r"D:\Blood_Smear_AI\data\final_dataset\data.yaml"

model = YOLO(MODEL)

results = model.val(
    data=DATA,
    split="test",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    conf=0.001,
    iou=0.7,
    plots=True,
    save_json=True,
    project=r"D:\Blood_Smear_AI\runs",
    name="FINAL_YOLO11N_3CLASS_TEST",
    exist_ok=True,
    verbose=True
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")


print("\nResults saved to:")
print(r"D:\Blood_Smear_AI\runs\FINAL_YOLO11N_3CLASS_TEST")