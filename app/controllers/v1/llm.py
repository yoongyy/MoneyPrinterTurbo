from fastapi import HTTPException, Request

from app.controllers.v1.base import new_router
from app.models.schema import (
    VideoScriptRequest,
    VideoScriptResponse,
    VideoSocialMetadataRequest,
    VideoSocialMetadataResponse,
    VideoTermsRequest,
    VideoTermsResponse,
)
from app.services import content_source, dialogue, llm
from app.utils import utils

# authentication dependency
# router = new_router(dependencies=[Depends(base.verify_token)])
router = new_router()


@router.post(
    "/scripts",
    response_model=VideoScriptResponse,
    summary="Create a script for the video",
)
def generate_video_script(request: Request, body: VideoScriptRequest):
    source_content = ""
    source_title = ""
    resolved_source_url = body.source_url
    video_subject = body.video_subject
    if not video_subject and not body.source_url:
        raise HTTPException(
            status_code=400,
            detail="video_subject or source_url is required",
        )
    if body.source_url:
        try:
            resolved_source = content_source.fetch_source_content(
                body.source_type, body.source_url
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        source_content = resolved_source.content
        source_title = resolved_source.title
        resolved_source_url = resolved_source.final_url
        video_subject = video_subject or source_title or resolved_source_url

    video_script = llm.generate_script(
        video_subject=video_subject,
        language=body.video_language,
        video_duration_minutes=body.video_duration_minutes,
        paragraph_number=body.paragraph_number,
        video_script_prompt=body.video_script_prompt,
        custom_system_prompt=body.custom_system_prompt,
        script_style=body.script_style,
        source_content=source_content,
        source_title=source_title,
        source_url=resolved_source_url,
    )
    response = {"video_script": video_script}
    return utils.get_response(200, response)


@router.post(
    "/terms",
    response_model=VideoTermsResponse,
    summary="Generate video terms based on the video script",
)
def generate_video_terms(request: Request, body: VideoTermsRequest):
    video_terms = llm.generate_terms(
        video_subject=body.video_subject,
        video_script=dialogue.strip_speaker_labels(
            body.video_script, body.script_style
        ),
        amount=body.amount,
    )
    response = {"video_terms": video_terms}
    return utils.get_response(200, response)


@router.post(
    "/social-metadata",
    response_model=VideoSocialMetadataResponse,
    summary="Generate social publishing metadata",
)
def generate_video_social_metadata(request: Request, body: VideoSocialMetadataRequest):
    metadata = llm.generate_social_metadata(
        video_subject=body.video_subject,
        video_script=body.video_script,
        language=body.language,
        platform=body.platform,
    )
    return utils.get_response(200, metadata)
