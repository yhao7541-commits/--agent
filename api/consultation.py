"""
咨询兼容 API

旧 URL 保留，内部统一交给 OperationsAgent 走知识库工具检索。
"""
import uuid

from fastapi import APIRouter, HTTPException

from agents.operations import OperationsAgent
from .core.response_models import ConsultationRequest, DataResponse


router = APIRouter(prefix="/api/consultation", tags=["咨询服务"])
operations_agent = OperationsAgent()


@router.post("/ask", response_model=DataResponse)
async def ask_consultation(request: ConsultationRequest):
    """提交咨询问题。"""
    try:
        result = operations_agent.run_turn(
            {
                "user_id": request.user_id,
                "conversation_id": f"consultation_{uuid.uuid4().hex[:8]}",
                "message": request.question,
            }
        )

        return DataResponse(
            message="咨询处理成功",
            data={
                "answer": result.get("reply", ""),
                "question": request.question,
                "intent": result.get("intent", "unknown"),
                "rag_used": result.get("rag_used", False),
                "rag_citations": result.get("rag_citations", {}),
                "trace_id": result.get("trace_id", ""),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
