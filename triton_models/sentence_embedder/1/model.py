import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                texts_raw = pb_utils.get_input_tensor_by_name(request, 'text').as_numpy().flatten()
                texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts_raw]

                embeddings = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

                out = pb_utils.Tensor('embedding', embeddings)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
        return responses

    def finalize(self):
        del self.model
