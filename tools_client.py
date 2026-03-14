
from __future__ import annotations
import os
import json
import requests
from typing import Dict, List, Optional, Any


class ToolClient:
    """
    Calls test6 (running on 127.0.0.1:8006) as "tools".
    Assumes test6 provides:
      - POST /vqa_batch  (multipart): image + questions_json or questions_text
      - POST /vqa        (multipart): image + question
      - (optional) POST /caption     (multipart): image
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8006", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_multipart(self, path: str, image_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": f"Non-JSON response (status={resp.status_code})", "text": resp.text[:500]}

        if resp.status_code >= 400:
            # Normalize error shape
            msg = payload.get("error") or payload.get("detail") or str(payload)
            raise RuntimeError(f"{path} failed ({resp.status_code}): {msg}")
        return payload

    def caption(self, image_path: str, model_dir: Optional[str] = None, offline: bool = True) -> Dict[str, Any]:
        """
        Try /caption first; if not available, fallback to VQA question.
        """
        # Try /caption
        try:
            data = {}
            if model_dir:
                data["model_dir"] = model_dir
            data["offline"] = str(bool(offline)).lower()
            return self._post_multipart("/caption", image_path, data)
        except Exception:
            # fallback via /vqa
            q = "What is the image about? Answer in one short sentence."
            out = self.vqa(image_path, q, model_dir=model_dir, offline=offline)

            ans = (out.get("answer") or "").lower()
            if any(x in ans for x in ["don't know", "dont know", "no idea", "none", "unknown"]):
                # second fallback
                out = self.vqa(image_path, "Is this indoor or outdoor? What room/place is it?", model_dir=model_dir, offline=offline)

            return out
        
        
    def vqa(self, image_path: str, question: str, model_dir: Optional[str] = None, offline: bool = True) -> Dict[str, Any]:
        data = {"question": question, "offline": str(bool(offline)).lower()}
        if model_dir:
            data["model_dir"] = model_dir
        return self._post_multipart("/vqa", image_path, data)

    def vqa_batch(
        self,
        image_path: str,
        questions: List[str],
        model_dir: Optional[str] = None,
        offline: bool = True,
        max_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """
        Uses /vqa_batch with questions_json (JSON array string).
        """
        data = {
            "questions_json": json.dumps(questions, ensure_ascii=False),
            "offline": str(bool(offline)).lower(),
            "max_new_tokens": str(int(max_new_tokens)),
        }
        if model_dir:
            data["model_dir"] = model_dir
        return self._post_multipart("/vqa_batch", image_path, data)