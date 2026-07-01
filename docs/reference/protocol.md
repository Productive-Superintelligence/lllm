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

Identifier fields stay plain and portable. Tactic names may contain display
spaces, but they must avoid percent escapes, `.`, `..`, `/`, `\`, and `:`.
Token-style fields such as request ids, event kinds, runtime kinds, states, and
error types must also avoid whitespace.
`package_ref` values point at `psi://.../tactics/...` tactic refs.
`service_ref` values must either point at `psi://.../services/...` service refs
or use absolute HTTP(S) service URLs without embedded credentials, URL params,
queries, or fragments.
