# Tool Governance

The Tool Gateway is the control point for operations-agent business actions.

Each tool declares:

- permission: `read`, `write`, `external`, or `sensitive`
- whether confirmation is required
- Pydantic input schema
- Pydantic output schema
- handler function

The gateway returns structured errors for unknown tools, validation failures, confirmation-required writes, and handler exceptions. It also appends tool-level trace events so tests and evals can inspect behavior without reading logs.

## Write Policy

These tools require explicit confirmation:

- `create_booking`
- `reschedule_booking`
- `cancel_booking`
- `write_customer_preference`
- `delete_customer_memory`

Read tools such as `check_schedule`, `find_available_staff`, `lookup_customer_profile`, `search_services`, and `search_knowledge_base` can execute without confirmation.

`write_customer_preference` and `delete_customer_memory` are sensitive and route confirmed memory changes through the customer memory lifecycle. Write status reflects `created`, `updated`, or `conflict`; delete status reflects `deleted` or `not_found`.
