import nltk
from nltk.corpus import cmudict
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from magenta.models.melody_rnn import melody_rnn_sequence_generator
import pretty_midi

class LyricsGenerator:
    def __init__(self):
        self.pronunciation_dict = cmudict.dict()
        self.rhyme_dict = {'A': ['love', 'dove'], 'B': ['sea', 'free']}
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-lyrics')
        self.model = GPT2LMHeadModel.from_pretrained('my-trained-lyric-model')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.melody_generator = melody_rnn_sequence_generator(config='attention_rnn')
        
    def generate_lyrics(self, theme, style='pop', length=16):
        """
        根据主题和风格生成歌词
        :param theme: 歌词主题
        :param style: 音乐风格
        :param length: 歌词行数
        :return: 生成的歌词
        """
        # 使用自定义GPT-2模型生成歌词
        input_ids = self.tokenizer.encode(theme, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=50)
        expanded_theme = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 分析情感倾向
        emotion = self.sentiment_analyzer(expanded_theme)[0]['label']
        
        # 根据扩展后的主题和情感生成歌词
        lines = []
        rhyme_group = random.choice(list(self.rhyme_dict.keys()))
        for i in range(length):
            if i % 2 == 0:
                rhyme_word = random.choice(self.rhyme_dict[rhyme_group])
                lines.append(f"{expanded_theme.strip()} {rhyme_word} ({emotion})")
            else:
                lines.append(f"{expanded_theme.strip()} ({emotion})")
        return '\n'.join(lines)
    
    def generate_melody(self, emotion_label):
        """
        Generate melody based on emotion label from lyrics
        :param emotion_label: Emotion label from sentiment analysis
        :return: pretty_midi.PrettyMIDI object
        """
        return self.melody_generator.generate(emotion_label)

if __name__ == '__main__':
    generator = LyricsGenerator()
    print(generator.generate_lyrics('爱情'))