"""Document upload and knowledge-base management."""

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, status

from backend.core.dependencies import get_current_user, get_document_service
from backend.models.user import User
from backend.schemas.document import DocumentList, DocumentPublic
from backend.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentPublic,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion into the knowledge base",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentPublic:
    document = await document_service.upload(current_user.id, file, background_tasks)
    return DocumentPublic.from_document(document)


@router.get("", response_model=DocumentList, summary="List uploaded documents")
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentList:
    items, total = await document_service.list_documents(
        current_user.id, limit=limit, offset=offset
    )
    return DocumentList(
        items=[DocumentPublic.from_document(d) for d in items], total=total
    )


@router.get(
    "/{document_id}", response_model=DocumentPublic, summary="Get a document's status"
)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentPublic:
    document = await document_service.get_document(current_user.id, document_id)
    return DocumentPublic.from_document(document)


@router.get(
    "/{document_id}/versions",
    response_model=list[DocumentPublic],
    summary="List the version history of a document (newest first)",
)
async def list_document_versions(
    document_id: str,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> list[DocumentPublic]:
    versions = await document_service.list_versions(current_user.id, document_id)
    return [DocumentPublic.from_document(d) for d in versions]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its embeddings",
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> None:
    await document_service.delete_document(current_user.id, document_id)
