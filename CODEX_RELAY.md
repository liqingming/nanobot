# Central OpenAI Codex Relay

The relay centralizes only OpenAI Codex provider calls. Agent loops, sessions,
workspaces, tool execution, memory, and compaction remain on each client machine.
The Codex OAuth login is stored and used only on the relay server.

No dedicated relay command is required. The process-wide behavior comes from
`.nanobot/config.json`, and server mode is exposed by the existing `nanobot serve` command.

## Server

Log in to OpenAI Codex on the central machine, then configure:

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8900
  },
  "codexRelay": {
    "mode": "server",
    "apiKey": "replace-with-a-long-random-secret",
    "requestTimeout": 600,
    "maxConcurrency": 4,
    "allowedModels": ["openai-codex/gpt-5.6-sol"]
  }
}
```

Start the normal API listener:

```text
nanobot serve
```

The relay endpoint is `/v1/codex/relay` and uses `codexRelay.apiKey`. In server
mode, the normal `/v1/chat/completions` and `/v1/models` AgentLoop routes are not
registered, so relay clients cannot execute tools or create sessions on the OAuth host.

Terminate TLS in front of the listener when clients connect over an untrusted network.

## Client

On every client/package machine, configure:

```json
{
  "codexRelay": {
    "mode": "client",
    "endpoint": "https://relay.example.com",
    "apiKey": "replace-with-the-server-secret",
    "requestTimeout": 600
  }
}
```

All OpenAI Codex providers created from that config then use the relay. Other
providers are unchanged. The endpoint must use HTTPS unless it is loopback;
`allowInsecureHttp: true` is available only for a trusted private network.

For the current trusted-LAN deployment, clients can use:

```json
{
  "codexRelay": {
    "mode": "client",
    "endpoint": "http://172.20.76.89:8900",
    "apiKey": "replace-with-the-server-secret",
    "requestTimeout": 600,
    "allowInsecureHttp": true
  }
}
```

The central Windows firewall rule permits TCP 8900 only from `LocalSubnet` on
Domain/Private profiles. Public-profile traffic remains blocked. Use HTTPS instead
if traffic leaves the trusted LAN.

Use `mode: "off"` (the default) to restore direct local Codex OAuth calls.

## Protocol and security boundaries

Requests contain normalized model messages, tool schemas, generation settings,
and model names. Responses are NDJSON events containing content/reasoning/tool-call
deltas and the final normalized provider response. The server does not receive a
workspace path and never runs tools. Configure `allowedModels` to restrict models;
an empty list allows all Codex models.
