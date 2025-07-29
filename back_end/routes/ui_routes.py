from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import json
from pathlib import Path

router = APIRouter()

from back_end.config_settings import templates

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_endpoint": "/echo"
    })

@router.get("/tag_list")
async def tag_list(request: Request):
    ner_model = request.app.state.ner_model
    tag2id = ner_model.tag2id
    tag_list = list({i.split('-')[-1].upper() for i in tag2id.keys()})

    return JSONResponse(tag_list)
