# HemoVision AI Dashboard — Connected to the Blood Smear Models

This version connects the design-first Next.js frontend to the real blood-smear inference pipeline.

## AI models used
- Stage 1: `D:\Blood_Smear_AI\runs\FINAL_YOLO11N_3CLASS-3\weights\best.pt`
- Stage 2 RBC morphology: `D:\Blood_Smear_AI\runs\rbc_morphology\weights\best_rbc_morphology.pth`
- Stage 2 implementation: `D:\Blood_Smear_AI\inference\stage2_rbc_morphology.py`

The backend discovers these files relative to the project root. Keep the HemoVision dashboard folder directly inside `D:\Blood_Smear_AI`.

## 1. Backend
Open PowerShell:

```powershell
cd D:\Blood_Smear_AI\HemoVision_AI_Dashboard\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Test:

```text
http://localhost:8000/health
```

## 2. Frontend
Open a second PowerShell:

```powershell
cd D:\Blood_Smear_AI\HemoVision_AI_Dashboard\frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## 3. Flow

```text
Frontend upload
   -> POST /api/analyze
   -> YOLO11n
   -> RBC/WBC/Platelet
   -> RBC crops
   -> RBC morphology EfficientNet-B0
   -> JSON + annotated image
   -> Dashboard
```

## Important
The application inference threshold is currently `0.25`. This is an application setting and does not change the official held-out test metrics.

Official final detector test metrics remain:
- Precision: 78.25%
- Recall: 82.16%
- mAP@50: 84.09%
- mAP@50-95: 75.34%

This dashboard is a research/educational prototype and is not a definitive clinical diagnosis system.
