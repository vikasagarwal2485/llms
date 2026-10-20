"""Local HTTP API that runs every trained task on the text it is given.

Usage: python -m classifier_model.server [--artifacts artifacts] [--host 127.0.0.1] [--port 8000] [--device auto]
"""

import argparse
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from classifier_model.multitask import MultiTaskClassifier

MAX_BATCH = 64


class ClassifyRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def exactly_one_input(self):
        if (self.text is None) == (self.texts is None):
            raise ValueError("Provide exactly one of 'text' or 'texts'")
        texts = [self.text] if self.text is not None else self.texts
        if not texts:
            raise ValueError("'texts' must contain at least one text")
        if len(texts) > MAX_BATCH:
            raise ValueError(f"At most {MAX_BATCH} texts per request; got {len(texts)}")
        if any(not t.strip() for t in texts):
            raise ValueError("Texts must be non-empty strings")
        return self


def _filter(results, min_confidence):
    if min_confidence is None:
        return results
    return {task: r for task, r in results.items() if r["confidence"] >= min_confidence}


def create_app(artifacts_dir="artifacts", device="auto"):
    classifier = MultiTaskClassifier.load(artifacts_dir, device)
    app = FastAPI(title="GPT-2 text classifiers")

    @app.get("/health")
    def health():
        return {"status": "ok", "tasks_loaded": len(classifier.tasks)}

    @app.get("/tasks")
    def tasks():
        return {"tasks": [
            {"task": t.name, "description": t.description, "labels": t.label_names}
            for t in classifier.tasks.values()
        ]}

    @app.post("/classify")
    def classify(request: ClassifyRequest):
        if not classifier.tasks:
            raise HTTPException(status_code=503, detail="No tasks are loaded")
        texts = [request.text] if request.text is not None else request.texts
        results = [_filter(r, request.min_confidence) for r in classifier.predict(texts)]
        return {"results": results[0] if request.text is not None else results}

    return app


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default="artifacts", help="Directory containing model bundles")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(args.artifacts, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
