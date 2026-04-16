from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os

from src.ingestion.ingestion import run_ingestion

router = APIRouter()


@router.post("/admin/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # ✅ Validate file
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        path = f"temp_{file.filename}"

        # ✅ Save file
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Run ingestion
        result = run_ingestion(path)

        # ✅ Cleanup
        os.remove(path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))