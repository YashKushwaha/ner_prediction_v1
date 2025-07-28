from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import json
from pathlib import Path

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return '<h1>Hello, World!</h1>'