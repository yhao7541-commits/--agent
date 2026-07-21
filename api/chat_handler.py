import uuid

from agents.operations import OperationsAgent


global_conversation_id = str(uuid.uuid4())
operations_agent = OperationsAgent()


async def ProcessUserInput_stream(user_input, state=None, context=None):
    """
    兼容旧 Web 流式入口。

    内部只执行单一 OperationsAgent；context 可继续承载多轮预约槽位。
    """
    if context is None:
        context = {}

    conversation_id = context.get("conversation_id", global_conversation_id)
    result = operations_agent.run_turn(
        {
            "user_id": context.get("user_id", "local_user"),
            "conversation_id": conversation_id,
            "message": user_input,
            "booking_slots": context.get("booking_slots", {}),
            "booking_slot_sources": context.get("booking_slot_sources", {}),
        }
    )
    context["conversation_id"] = conversation_id
    context["booking_slots"] = result.get("booking_slots", {})
    context["booking_slot_sources"] = result.get("booking_slot_sources", {})

    yield result.get("reply", "")
