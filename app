import json
import sys

import requests  # make sure this is installed in your WSGI environment

CONTENT_TYPE_JSON = "application/json"

# Keep existing auth imports / path tweaks
sys.path.append("/var/www/wsgi")
from auth import (
    validate_entitlements,
    is_allowed_completion_model,
    is_allowed_embedding_model,
)  # noqa: E402


def _build_backend_url(environ, model_name, original_path):
    """
    Build URL that goes back through Apache's /model/... route so existing
    ProxyPass rules apply exactly as before.
    """
    scheme = environ.get("wsgi.url_scheme", "http")
    host = environ.get("HTTP_HOST")
    if not host:
        server_name = environ.get("SERVER_NAME", "127.0.0.1")
        server_port = environ.get("SERVER_PORT")
        host = f"{server_name}:{server_port}" if server_port else server_name

    # Example: /model/jina-embeddings-v2-base-en/v1/embeddings
    return f"{scheme}://{host}/model/{model_name}/v1{original_path}"


def _forward_to_model(environ, start_response, model_name, data, original_path):
    """
    Forward the (possibly modified) JSON body to the model-specific endpoint
    via Apache's /model/... path, and stream the response back.

    For embeddings, Triton will only ever see a 'text' field (no 'input').
    """
    url = _build_backend_url(environ, model_name, original_path)
    payload = json.dumps(data).encode("utf-8")

    # Forward all client headers except hop-by-hop and Host/Content-Length.
    headers = {}
    for key, value in environ.items():
        if not key.startswith("HTTP_"):
            continue
        name = key[5:].replace("_", "-").title()
        lower = name.lower()
        if lower in (
            "host",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        ):
            continue
        headers[name] = value

    # Force JSON; requests will set Content-Length.
    headers["Content-Type"] = CONTENT_TYPE_JSON

    method = environ.get("REQUEST_METHOD", "POST")

    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=payload,
            timeout=60,
        )

        body = resp.content
        status_line = f"{resp.status_code} {resp.reason or ''}".strip()

        response_headers = []
        for k, v in resp.headers.items():
            # Let mod_wsgi manage hop-by-hop fields
            lk = k.lower()
            if lk in ("connection", "transfer-encoding"):
                continue
            if lk == "content-length":
                continue
            response_headers.append((k, v))

        # Ensure Content-Length matches the body we return
        response_headers.append(("Content-Length", str(len(body))))

        start_response(status_line, response_headers)
        return [body]

    except requests.exceptions.RequestException as e:
        print(f"Error proxying to model backend: {e}", file=sys.stdout, flush=True)
        start_response("502 Bad Gateway", [("Content-Type", CONTENT_TYPE_JSON)])
        return [
            json.dumps(
                {"error": f"Error proxying to model backend: {str(e)}"}
            ).encode("utf-8")
        ]


def application(environ, start_response):
    """
    Main WSGI entrypoint.

    Behaviour vs your original router:

      - Only POST is allowed; others return 405 (same as before).
      - JSON body must contain 'model' (same as before).
      - Entitlements validated via validate_entitlements() (same).
      - Model access checked via is_allowed_completion_model() or
        is_allowed_embedding_model() (same).
      - If model is allowed:
          * PATH_INFO == '/embeddings' (i.e. /v1/embeddings):
              - Accepts either 'input' or 'text' from client.
              - Chooses the present one and puts it in 'text'.
              - Removes 'input' so Triton only sees 'text'.
              - Proxies internally to /model/<model>/v1/embeddings.
          * Any other PATH_INFO:
              - 307 redirect to /model/<model>/v1<PATH_INFO> (unchanged).
    """
    method = environ.get("REQUEST_METHOD", "GET")
    original_path = environ.get("PATH_INFO", "")

    # Only allow POST as before
    if method != "POST":
        start_response(
            "405 Method Not Allowed",
            [("Content-Type", CONTENT_TYPE_JSON)],
        )
        return [
            json.dumps(
                {"error": "Only POST method is supported"}
            ).encode("utf-8")
        ]

    try:
        # Read request body safely
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            content_length = 0

        raw_body = (
            environ["wsgi.input"].read(content_length) if content_length > 0 else b""
        )
        if not raw_body:
            data = {}
        else:
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                start_response(
                    "400 Bad Request",
                    [("Content-Type", CONTENT_TYPE_JSON)],
                )
                return [
                    json.dumps(
                        {"error": "Invalid JSON in request body"}
                    ).encode("utf-8")
                ]

        if not isinstance(data, dict):
            start_response(
                "400 Bad Request",
                [("Content-Type", CONTENT_TYPE_JSON)],
            )
            return [
                json.dumps(
                    {"error": "Request JSON must be an object"}
                ).encode("utf-8")
            ]

        # Extract model name
        model_name = data.get("model", "")
        print(
            f"GMGENAI Router: Path: {original_path}, Model: {model_name}",
            file=sys.stdout,
        )
        sys.stdout.flush()

        if not model_name:
            start_response(
                "400 Bad Request",
                [("Content-Type", CONTENT_TYPE_JSON)],
            )
            return [
                json.dumps(
                    {"error": "No model specified in request"}
                ).encode("utf-8")
            ]

        # Entitlement validation (unchanged semantics)
        try:
            user, entitlements = validate_entitlements(environ, model_name)
        except Exception as e:
            print(f"Entitlement validation failed: {e}", file=sys.stdout)
            sys.stdout.flush()
            start_response(
                "401 Unauthorized",
                [("Content-Type", CONTENT_TYPE_JSON)],
            )
            return [
                json.dumps(
                    {"error": f"Entitlement validation failed: {str(e)}"}
                ).encode("utf-8")
            ]

        # Model access check (unchanged)
        is_model_allowed = (
            is_allowed_completion_model(model_name, entitlements)
            or is_allowed_embedding_model(model_name, entitlements)
        )

        if not is_model_allowed:
            print(
                f"User is trying to access model {model_name} and access is forbidden",
                file=sys.stdout,
            )
            sys.stdout.flush()
            start_response(
                "403 Forbidden",
                [("Content-Type", CONTENT_TYPE_JSON)],
            )
            return [
                json.dumps(
                    {"error": "Access to the specified model is forbidden"}
                ).encode("utf-8")
            ]

        # --------- ALLOWED REQUESTS FROM HERE ---------

        # Special handling for embeddings to normalise input/text and proxy.
        if original_path == "/embeddings":
            # Clients may send either 'input' or 'text' (not both by design).
            # We enforce that Triton only ever sees 'text'.
            if "text" in data:
                text_value = data["text"]
            elif "input" in data:
                text_value = data["input"]
            else:
                start_response(
                    "400 Bad Request",
                    [("Content-Type", CONTENT_TYPE_JSON)],
                )
                return [
                    json.dumps(
                        {
                            "error": "Either 'input' or 'text' must be provided for /v1/embeddings"
                        }
                    ).encode("utf-8")
                ]

            # Ensure backend only sees 'text', never 'input'.
            data["text"] = text_value
            if "input" in data:
                del data["input"]

            print(
                f"Proxying embeddings request for model {model_name} to backend with 'text' only",
                file=sys.stdout,
            )
            sys.stdout.flush()

            return _forward_to_model(
                environ, start_response, model_name, data, original_path
            )

        # For all non-embeddings paths, keep original 307 redirect behaviour.
        redirect_url = f"/model/{model_name}/v1{original_path}"
        print(
            f"Redirecting request to model endpoint: {redirect_url}",
            file=sys.stdout,
        )
        sys.stdout.flush()

        start_response(
            "307 Temporary Redirect",
            [
                ("Location", redirect_url),
                ("Content-Type", "text/plain"),
            ],
        )
        return [b"Redirecting to model endpoint"]

    except Exception as e:
        # Preserve general error semantics
        print(f"Error processing request: {e}", file=sys.stdout)
        sys.stdout.flush()
        start_response(
            "500 Internal Server Error",
            [("Content-Type", CONTENT_TYPE_JSON)],
        )
        return [
            json.dumps(
                {"error": f"Failed to process request: {str(e)}"}
            ).encode("utf-8")
        ]
