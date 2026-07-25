import pygame
import numpy as np

pygame.mixer.init(frequency=44100, size=-16, channels=2)

def _generate_tone(frequency, duration_ms, volume=0.3, fade_ms=20):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    tone = np.sin(frequency * t * 2 * np.pi)

    fade_samples = int(sample_rate * fade_ms / 1000)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    tone[:fade_samples] *= fade_in
    tone[-fade_samples:] *= fade_out

    audio = (tone * volume * 32767).astype(np.int16)
    stereo_audio = np.ascontiguousarray(np.column_stack((audio, audio)))
    return pygame.sndarray.make_sound(stereo_audio)

_letter_confirm = _generate_tone(880, 90, volume=0.25)
_word_complete  = _generate_tone(660, 140, volume=0.3)
_error_sound    = _generate_tone(220, 100, volume=0.2)

_muted = False
_volume = 0.7

def set_muted(muted: bool):
    global _muted
    _muted = muted

def set_volume(vol: float):
    global _volume
    _volume = max(0.0, min(1.0, vol))

def play_letter_confirm():
    if not _muted:
        _letter_confirm.set_volume(_volume)
        _letter_confirm.play()

def play_word_complete():
    if not _muted:
        _word_complete.set_volume(_volume)
        _word_complete.play()

def play_error():
    if not _muted:
        _error_sound.set_volume(_volume)
        _error_sound.play()