# Demo Script

Use `/api/operations/chat` or the `/operations` console for these demos.

## 1. Incomplete Booking

User: `我想约一个肩颈放松`

Expected: intent is booking, missing date and time, no `create_booking` execution.

## 2. Complete Booking Requires Confirmation

User: `我想明天下午3点约肩颈放松`

Expected: the agent plans read tools and `create_booking`, but the gateway returns a confirmation request before any write succeeds.

## 3. Confirmed Booking Executes Write

Send the previous `confirmation_request.tool_name` and `confirmation_request.arguments` back with message `确认`.

Expected: `create_booking` succeeds and the reply includes a booking id.

## 4. Policy Question Uses RAG

User: `如果我迟到20分钟会怎么样？`

Expected: `search_knowledge_base` runs and trace metadata includes source chunks from `booking_policy.md`.

## 5. Preference Creates Memory Proposal

User: `我以后都喜欢安静一点的房间`

Expected: a memory proposal is produced and `write_customer_preference` requires confirmation.
