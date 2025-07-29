
from fastapi import Form, File, UploadFile
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

router = APIRouter()

async def stream_results(results):
    for sent_result in results:
        for word, tag in sent_result:
            #yield f"{word}: {tag} <br>"
            tag = tag.upper()
            if '-' in tag:
                prefix, entity = tag.split('-')
            else:
                prefix = entity = tag
            yield f'<span class="ner-tag {entity}" title="{tag}" prefix="{prefix}">{word}</span> '
            await asyncio.sleep(0.1)
        yield "<br>"

@router.post("/ner_predict")
async def ner_predict(request: Request, message: str = Form(...), image: UploadFile = File(None)):
    ner_model = request.app.state.ner_model
    sentences = [message]
    results = ner_model.batch_predict(sentences)
    return StreamingResponse(stream_results(results), media_type="text/plain")
