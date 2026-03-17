import triton_python_backend_utils as pb_utils
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import tempfile
import base64
import os


class TritonPythonModel:
    def initialize(self, args):
        self.pipe = KPipeline(lang_code='a', device='cpu')

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                text_raw = pb_utils.get_input_tensor_by_name(request, 'text').as_numpy().flatten()[0]
                voice_raw = pb_utils.get_input_tensor_by_name(request, 'voice').as_numpy().flatten()[0]

                text = text_raw.decode('utf-8') if isinstance(text_raw, bytes) else str(text_raw)
                voice = voice_raw.decode('utf-8') if isinstance(voice_raw, bytes) else str(voice_raw)

                samples = []
                for _, _, audio in self.pipe(text, voice=voice, speed=1.0):
                    samples.append(audio)

                if samples:
                    audio_np = np.concatenate(samples)
                    tmp = tempfile.mktemp(suffix='.wav')
                    sf.write(tmp, audio_np, 24000)
                    with open(tmp, 'rb') as f:
                        data = f.read()
                    os.unlink(tmp)
                    audio_b64 = base64.b64encode(data).decode('utf-8')
                else:
                    audio_b64 = ''

                out = pb_utils.Tensor('audio_b64', np.array([audio_b64.encode('utf-8')], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
        return responses

    def finalize(self):
        del self.pipe
