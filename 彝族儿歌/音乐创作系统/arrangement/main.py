import pretty_midi
import numpy as np
from espnet2.bin.tts_inference import Text2Speech

class ArrangementGenerator:
    def __init__(self):
        self.instruments = {
            'piano': pretty_midi.Instrument(program=0),
            'guitar': pretty_midi.Instrument(program=24),
            'bass': pretty_midi.Instrument(program=32)
        }
        self.chord_progressions = {
            'happy': ['C-G-Am-F', 'I-V-vi-IV'],
            'sad': ['Am-F-C-G', 'vi-IV-I-V'],
            'summer': ['C-G-Am-F', 'I-V-vi-IV'],
            'ocean': ['C-G-Am-F', 'I-V-vi-IV']
        }
    
    def generate_chord_progression(self, key='C', style='pop', emotion=None):
        """生成和弦进行"""
        if emotion and emotion in self.chord_progressions:
            return random.choice(self.chord_progressions[emotion])
        
        # 示例和弦进行
        if style == 'pop':
            return ['C', 'G', 'Am', 'F']
        elif style == 'rock':
            return [f'{key}5', f'F5', f'G5', f'{key}5']
        else:
            return [f'{key}maj7', f'Dm7', f'G7', f'Cmaj7']
    
    def create_midi(self, chords, output_file='output.mid', bpm=120):
        """创建MIDI文件"""
        midi = pretty_midi.PrettyMIDI()
        midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
        midi.tempo_changes.append(pretty_midi.TempoChange(bpm, 0))
        
        # 为每个和弦添加音符
        for i, chord in enumerate(chords):
            for instrument in self.instruments.values():
                note = pretty_midi.Note(
                    velocity=100, 
                    pitch=60 + i*5, 
                    start=i, 
                    end=i+1)
                instrument.notes.append(note)
        
        # 添加乐器到MIDI文件
        for instrument in self.instruments.values():
            midi.instruments.append(instrument)
        
        midi.write(output_file)
        return output_file
        
    def text_to_speech(self, text, output_file='output.wav'):
        """Convert text to speech using ESPnet's VITS model"""
        tts = Text2Speech.from_pretrained("vits")
        speech = tts(text)["wav"]
        import soundfile as sf
        sf.write(output_file, speech.numpy(), 22050)
        return output_file

if __name__ == '__main__':
    generator = ArrangementGenerator()
    chords = generator.generate_chord_progression()
    generator.create_midi(chords)