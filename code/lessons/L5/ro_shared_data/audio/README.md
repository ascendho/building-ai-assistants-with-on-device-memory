# Voice-note audio

Five short voice memos, one per voice capture in `ro_shared_data/memories.json`, plus `question.wav`, a spoken question Lesson 4 asks the assistant in section 5. Lesson 4 transcribes them on-device with a small Whisper model, so the "voice" modality runs a real speech-to-text step. Each memo maps to its note through the `audio_file` field in `ro_shared_data/memories.json`. `question.wav` stands on its own and belongs to no memory.
