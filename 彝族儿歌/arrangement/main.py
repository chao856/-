import pretty_midi
import numpy as np

class ArrangementGenerator:
    def generate_chord_progression(self, key='C', progression_type='simple'):
        """生成基础和弦进行
        :param key: 调式（默认C大调）
        :param progression_type: 和弦类型（simple/complex）
        :return: 和弦进行列表
        """
        # 基础和弦进行模板（C大调示例）
        chord_templates = {
            'simple': [f'{key}', f'{key}7', f'{key}m', f'{key}m7'],
            'complex': [f'{key}', f'{key}maj7', f'{key}m7', f'{key}7', f'{key}m', f'{key}dim']
        }
        return chord_templates.get(progression_type, chord_templates['simple'])

    def create_midi(self, chords, output_path='chords.mid', tempo=120):
        """将和弦进行转换为MIDI文件
        :param chords: 和弦列表
        :param output_path: 输出MIDI路径
        :param tempo: 速度(BPM)
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # 钢琴音色

        # 定义和弦音符映射（简化版）
        chord_notes = {
            'C': [60, 64, 67],    # C和弦: C4, E4, G4
            'C7': [60, 64, 67, 70], # C7和弦
            'Cm': [60, 63, 67],   # Cm和弦
            'Cm7': [60, 63, 67, 70],# Cm7和弦
            'Cmaj7': [60, 64, 67, 71] # Cmaj7和弦
        }

        # 添加和弦到MIDI（每个和弦持续2拍）
        for i, chord in enumerate(chords):
            notes = chord_notes.get(chord, [60, 64, 67])  # 默认C和弦
            start_time = i * 2
            end_time = start_time + 2
            for pitch in notes:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                instrument.notes.append(note)

        midi.instruments.append(instrument)
        midi.write(output_path)
        return output_path

if __name__ == '__main__':
    generator = ArrangementGenerator()
    chords = generator.generate_chord_progression(progression_type='simple')
    generator.create_midi(chords, output_path='simple_chords.mid')
    print(f"MIDI文件已生成: simple_chords.mid")