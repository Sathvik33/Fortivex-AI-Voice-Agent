# Piper TTS Voice Models

This directory stores the downloaded Piper TTS voice model files.

## Expected files after download:
- `en_US-lessac-medium.onnx` — English voice model (ONNX format)
- `en_US-lessac-medium.onnx.json` — Voice configuration

## Download automatically:
The app downloads these automatically on first TTS synthesis.

## Manual download:
```bash
# English voice (recommended)
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -o models/en_US-lessac-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -o models/en_US-lessac-medium.onnx.json
```

## Note on Hindi TTS:
Piper TTS does not currently have a high-quality Hindi voice.
For Hindi input, the agent responds in English TTS.
This is a documented limitation, not a bug.
