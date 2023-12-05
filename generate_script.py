from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("/home/eddiediller/audiocraft/checkpoints/my_audio_lm/")
#model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=8)  # generate 8 seconds.

descriptions = ["Synth-heavy psychedelic and disco-infused pop.", "Synth-heavy psychedelic and disco-infused pop."]

wav = model.generate(descriptions)  # generates 1 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")