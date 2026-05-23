from datetime import UTC

import httpx


class InferenceApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def fetch_model_info(self) -> dict:
        response = self._client.get("/model/info")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Failed to fetch model info: {exc}") from exc
        return response.json()

    def predict(self, message: dict) -> dict:
        body = {
            "transaction_id": message["transaction_id"],
            "user_id": message["user_id"],
            "merchant_id": message["merchant_id"],
            "merchant_category": message["merchant_category"],
            "amount": message["amount"],
            "country": message["country"],
            "timestamp": message["timestamp"].astimezone(UTC).isoformat(),
            "device_type": message["device_type"],
            "ip_hash": message["ip_hash"],
            "features": message["features"],
        }
        response = self._client.post("/predict", json=body)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()


__all__ = ["InferenceApiClient"]
