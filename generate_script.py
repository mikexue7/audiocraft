from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("/home/michaelxue/audiocraft/checkpoints/my_audio_lm") # finetuned model
#model = MusicGen.get_pretrained("facebook/musicgen-small") # baseline model
model.set_generation_params(duration=15, temperature=1.0)  # generate 15 seconds.

base_descriptions = ["Synth-heavy psychedelic and disco-infused pop.",
                     "A jazzy masterpiece with bluesy vibes and psychedelic hues.",
                     "Reggae rhythms meet the dreamy sounds of a psychedelic island groove.",
                     "Chillhop beats in a kaleidoscopic world, making it groove and soothe simultaneously.",
                     "Country twang with a psychedelic echo in a dusty saloon.",
                     "A classical symphony, as if conducting an orchestra on a cosmic voyage.",
                     "Rock and roll with cosmic energy, colliding eras in a psychedelic frenzy.",
                     "Hip-hop beats dripping with psychedelia, making it a mind trip.",
                     "Indie folk takes a psychedelic detour, letting the acoustic vibes mingle with the cosmic.",
                     "An experimental noise adventure, pushing the boundaries in a psychedelic cacophony."
                    ]

descriptions = ["Synth-heavy psychedelic and disco-infused pop like <TI>.",
                "A jazzy masterpiece with bluesy vibes and psychedelic hues in the style of <TI>.",
                "Reggae rhythms meet the dreamy sounds of a psychedelic island groove with a <TI> touch.",
                "Chillhop beats in a kaleidoscopic world, making it groove and soothe simultaneously with the vibes of <TI>.",
                "Country twang with a psychedelic echo in a dusty saloon, as performed by <TI>.",
                "A classical symphony, as if <TI> were conducting an orchestra on a cosmic voyage.",
                "Rock and roll with <TI>'s cosmic energy, colliding eras in a psychedelic frenzy.",
                "Hip-hop beats dripping with <TI>'s signature psychedelia, making it a mind trip.",
                "Indie folk takes a psychedelic detour with <TI>, letting the acoustic vibes mingle with the cosmic.",
                "An experimental noise adventure, in the spirit of <TI>, pushing the boundaries in a psychedelic cacophony."
               ]

names = ["pop", "jazz", "reggae", "chillhop", "country", "classical", "rock", "hiphop", "indie", "noise"]

wav = model.generate(base_descriptions)  # generates samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'model_0_out/{names[idx]}', one_wav.cpu(), model.sample_rate, strategy="loudness")