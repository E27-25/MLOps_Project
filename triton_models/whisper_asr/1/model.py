import triton_python_backend_utils as pb_utils
import whisper
import soundfile as sf
import numpy as np
import io
import json
from transformers import pipeline as hf_pipeline


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        params = model_config.get('parameters', {})
        size = params.get('whisper_size', {}).get('string_value', 'base')
        self.model = whisper.load_model(size, device='cpu')
        self.model_th = hf_pipeline(
            'automatic-speech-recognition',
            model='nectec/Pathumma-whisper-th-medium',
            device='cpu',
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                raw = pb_utils.get_input_tensor_by_name(request, 'wav_bytes').as_numpy().flatten()[0]
                wav_bytes = raw if isinstance(raw, bytes) else bytes(raw)

                audio, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')

                # Language detection
                audio_trim = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio_trim).to(self.model.device)
                _, lang_probs = self.model.detect_language(mel)
                detected_lang = max(lang_probs, key=lang_probs.get)

                if detected_lang == 'th':
                    result = self.model_th({'raw': audio, 'sampling_rate': sr})
                    transcript = result['text'].strip()
                else:
                    result = self.model.transcribe(audio)
                    transcript = result['text'].strip()

                out = pb_utils.Tensor('transcript', np.array([transcript.encode('utf-8')], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
        return responses

    def finalize(self):
        del self.model
        del self.model_th
