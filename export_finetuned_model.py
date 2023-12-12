from audiocraft.utils import export
from audiocraft import train
xp = train.main.get_xp_from_sig('f151ffd2')
export.export_lm(xp.folder / 'checkpoint.th', '/home/michaelxue/audiocraft/checkpoints/my_audio_lm/state_dict.bin')
# You also need to bundle the EnCodec model you used !!

## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
## This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model('facebook/encodec_32khz', '/home/michaelxue/audiocraft/checkpoints/my_audio_lm/compression_state_dict.bin')