# Protocol API

Core public objects:

| Object | Use |
| --- | --- |
| `Tactic` | Typed callable boundary. |
| `TacticInfo` | Name, schemas, refs, capabilities, examples, metadata. |
| `CallContext` | Request id, trace id, metadata, caller context. |
| `TacticEvent` | Streaming/status/error event envelope. |
| `TacticResolver` | Resolve refs to local or remote tactics. |

Input/output types should be Pydantic-schema-compatible. `BaseModel` is the
cleanest path, but simple JSON-compatible annotations and supported Pydantic
types can also work.
