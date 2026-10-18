"""Frozen checkpoint interfaces and supported prefix normalization."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .readers import normalize

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
MODELS = json.loads((CONFIGS / 'models.json').read_text())


def truncate(x, model, width):
    if width not in MODELS[model]['widths'] or width > x.shape[-1]:
        raise ValueError(f'Unsupported {model} width: {width}')
    return normalize(x[..., :width])


class VideoProcessor:
    def __init__(self, processor):
        self.processor = processor
        self.metadata = None

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get('videos') is not None:
            kwargs['videos_kwargs'] = {**(kwargs.get('videos_kwargs') or {}),
                                      'do_sample_frames': False, 'cap_pixels_per_frame': True}
            kwargs['video_metadata'] = [self.metadata]
        return self.processor(*args, **kwargs)


def runtime_view(source, target, *, ovis=False, vision_model=None, image_processor=None):
    """Use local component paths without changing checkpoint weight files."""
    import torch
    source, target = Path(source).resolve(), Path(target).resolve()
    base = source / 'model' if ovis else source
    target.mkdir(parents=True, exist_ok=False)
    for p in base.iterdir():
        if p.name.startswith('.') or p.name == 'README.md':
            continue
        if not ovis and p.name in ('config.json', 'processor_config.json'):
            doc = json.loads(p.read_text())
            if p.name == 'config.json':
                doc['backbone_model'] = str(source / 'backbone')
            if doc.get('vision_mode') == 'custom':
                if vision_model is None:
                    raise ValueError('Omni Mini 0.9B requires --vision-model')
                doc['vision_model'] = str(Path(vision_model).resolve())
            elif p.name == 'processor_config.json':
                if image_processor is None:
                    raise ValueError('Omni Mini 2.3B requires --image-processor')
                doc['backbone_model'] = str(Path(image_processor).resolve())
            (target/p.name).write_text(json.dumps(doc))
        else:
            (target/p.name).symlink_to(p)
    if ovis and not (target/'spk_dict.pt').exists():
        # Generation-only sidecar is missing from the embedding release.
        torch.save({}, target/'spk_dict.pt')
    return target


class Encoder:
    def __init__(self, name, checkpoint, *, device='cuda:0', runtime_dir=None,
                 vision_model=None, image_processor=None):
        import torch
        self.name, self.spec = name, MODELS[name]
        self.device = torch.device(device)
        path = str(Path(checkpoint).resolve())
        if name in ('gemma2', 'qwen2b', 'qwen8b', 'jina'):
            from sentence_transformers import SentenceTransformer
            options = {'model_kwargs': {'default_task': 'retrieval', 'modality': 'omni'}} if name == 'jina' else {
                'model_kwargs': {'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'}}
            self.model = SentenceTransformer(path, device=device, trust_remote_code=name == 'jina',
                                             local_files_only=True, **options).eval()
            if name == 'jina':
                self.video_processor = VideoProcessor(self.model[0].processor)
                self.model[0].processor = self.video_processor
        elif name == 'ovis':
            from transformers import AutoModelForMultimodalLM, AutoProcessor
            if runtime_dir is None:
                raise ValueError('Ovis requires an empty --runtime-dir')
            path = runtime_view(path, runtime_dir, ovis=True)
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                path, dtype=torch.bfloat16, device_map={'': device}, local_files_only=True).eval()
        else:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            if runtime_dir is None:
                raise ValueError('Omni Mini requires an empty --runtime-dir')
            path = str(runtime_view(path, runtime_dir, vision_model=vision_model, image_processor=image_processor))
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16,
                                                   local_files_only=True).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                                      fix_mistral_regex=True, local_files_only=True)
            self.processor.tokenizer = self.model.tokenizer = tokenizer
            self.model.set_pad_ids(**{f'{m}_pad': tokenizer.convert_tokens_to_ids(f'<|{m}_pad|>')
                                      for m in ('audio', 'image', 'video')})
            self.model._load_weights(path)
        self.model.requires_grad_(False)

    def encode(self, value, modality, *, role='document', instruction='', metadata=None):
        import torch
        if modality not in self.spec['modalities']:
            raise ValueError(f'{self.name} does not support {modality}')
        is_query = role == 'query'
        if self.name == 'jina':
            if modality == 'video':
                if metadata is None:
                    raise ValueError('Video source timing is required')
                self.video_processor.metadata = metadata
                value = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in value])
            fn = self.model.encode_query if is_query else self.model.encode_document
            with torch.inference_mode():
                result = fn([value], batch_size=1, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)[0]
        elif self.name in ('gemma2', 'qwen2b', 'qwen8b'):
            processing = {'text': {'truncation': False}}
            if modality == 'video':
                frames = [Image.open(p).convert('RGB') for p in value]
                value = {'video': {'array': frames, 'video_metadata': {
                    'fps': 1., 'total_num_frames': len(frames), 'duration': float(len(frames)),
                    'frames_indices': list(range(len(frames)))}}}
                processing['video'] = {'do_sample_frames': False}
            elif modality == 'audio':
                import soundfile as sf
                from scipy.signal import resample_poly
                samples, rate = sf.read(value, dtype='float32')
                if samples.ndim == 2:
                    samples = samples.mean(1)
                if rate != 16000:
                    divisor = math.gcd(rate, 16000)
                    samples = resample_poly(samples, 16000//divisor, rate//divisor).astype(np.float32)
                value = {'audio': {'array': samples, 'sampling_rate': 16000}}
            elif modality == 'image':
                value = {'image': str(value)}
            prompt = ('Represent the user\'s input.' if self.name.startswith('qwen') else
                      self.model.prompts['SearchQuery' if is_query else 'Document'] if modality == 'text' else '')
            features = self.model.preprocess([value], prompt=prompt, processing_kwargs=processing)
            limit = 8192 if self.name == 'gemma2' else 32768
            if int(features['attention_mask'].sum(-1).max()) > limit:
                raise ValueError('Input exceeds context; refusing silent truncation')
            with torch.inference_mode():
                result = self.model.encode([value], prompt=prompt, batch_size=1, convert_to_numpy=True,
                                           normalize_embeddings=True, processing_kwargs=processing)[0]
        else:
            content = []
            if self.name == 'ovis':
                text = f'Instruction: {instruction}\nQuery: {value}' if is_query else (
                    f'Instruction: {instruction}\nDocument: {value}' if modality == 'text' else
                    'Instruction: Represent this input as a retrieval candidate.')
                content.append({'type': 'text', 'text': text})
            elif modality == 'text':
                content = value
            if modality != 'text':
                media = [Image.open(p).convert('RGB') for p in value] if modality == 'video' and self.name == 'ovis' else value
                content.append({'type': modality, modality: media})
            messages = [{'role': 'user', 'content': content}]
            if self.name == 'ovis':
                options = {'padding': True}
                if modality == 'video':
                    if metadata is None:
                        raise ValueError('Video source timing is required')
                    options['videos_kwargs'] = {'fps': metadata['fps'], 'do_sample_frames': False,
                        'min_frames': len(value), 'max_frames': len(value), 'cap_pixels_per_frame': True,
                        'size': {'shortest_edge': 3136, 'longest_edge': 1310720}}
                inputs = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                            return_tensors='pt', processor_kwargs=options)
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                with torch.inference_mode():
                    output = self.model.thinker(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                    mask = inputs['attention_mask']
                    last = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(1)
                    result = output.hidden_states[-1][0, last[0]].float().cpu().numpy()
            else:
                options = {'text_recipe': 'chat'} if is_query else {}
                inputs = self.processor.apply_chat_template(messages, role='query' if is_query else 'passage',
                            tokenize=True, return_tensors='pt', **options).to(self.device)
                # These custom backbones retain position state across calls.
                backbone = getattr(self.model, 'backbone', None)
                if backbone is not None and hasattr(backbone, 'rope_deltas'):
                    backbone.rope_deltas = None
                with torch.inference_mode():
                    result = self.model(**inputs).pooler_output[0].float().cpu().numpy()
        result = normalize(np.asarray(result, dtype=np.float32))
        if result.shape != (max(self.spec['widths']),):
            raise ValueError(f'Unexpected {self.name} embedding shape: {result.shape}')
        return result
